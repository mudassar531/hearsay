"""Optional speaker diarization for single-voice TTS datasets (the ``[diarize]`` extra).

Whisper/Parakeet produce one unlabeled transcript, so concatenating multi-speaker
audio splices every voice into one "speaker" — fatal for single-voice TTS. This
module adds an *optional* speaker timeline and assigns each clip a speaker.

The design is pluggable and offline-testable: the pure ``assign_speaker`` takes a
normalized list of :class:`SpeakerTurn` (speaker, start, end) — from *any* diarizer
— and assigns a clip its **maximum-total-overlap** speaker plus a purity score
(dominant overlap / total overlapped speech), so cross-speaker clips can be dropped.

The real backend is :class:`PyannoteDiarizer` (lazy-imported, so a plain install
never pays the torch import cost). It is gated: the user must accept the model's
conditions on Hugging Face and provide a read token (``HF_TOKEN``). When the extra
is not installed the build degrades cleanly to a mixed-speaker dataset with a warning.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from hearsay.dataset.models import DiarizeConfig
from hearsay.errors import DiarizationError

# Hugging Face honours HF_TOKEN first, then the deprecated HUGGING_FACE_HUB_TOKEN.
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class SpeakerTurn(NamedTuple):
    """A diarized stretch of speech attributed to one speaker."""

    speaker: str
    start_s: float
    end_s: float


# A diarizer maps an audio file to a speaker timeline. Injectable for offline tests.
Diarizer = Callable[[Path], list[SpeakerTurn]]


def assign_speaker(
    start_s: float, end_s: float, turns: list[SpeakerTurn]
) -> tuple[str | None, float]:
    """Assign a clip ``[start_s, end_s]`` its dominant speaker by maximum total overlap.

    Returns ``(speaker, purity)`` where purity = dominant speaker's overlap / total
    overlapped speech (1.0 = pure, lower = more cross-talk). ``(None, 0.0)`` when no
    turn overlaps the clip (the clip should then be treated as unknown/dropped, not
    guessed). Matches whisperX/pyannote's argmax-overlap rule.
    """
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        overlap = min(end_s, turn.end_s) - max(start_s, turn.start_s)
        if overlap > 0:
            overlap_by_speaker[turn.speaker] = overlap_by_speaker.get(turn.speaker, 0.0) + overlap
    if not overlap_by_speaker:
        return None, 0.0
    total = sum(overlap_by_speaker.values())
    speaker = max(overlap_by_speaker, key=lambda s: overlap_by_speaker[s])
    return speaker, overlap_by_speaker[speaker] / total


def dominant_speaker(turns: list[SpeakerTurn]) -> str | None:
    """The speaker with the most total speech time across all turns, or None."""
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + max(0.0, turn.end_s - turn.start_s)
    return max(totals, key=lambda s: totals[s]) if totals else None


def resolve_token(config: DiarizeConfig) -> str | None:
    """The HF access token: explicit config value, else HF_TOKEN / HUGGING_FACE_HUB_TOKEN."""
    if config.hf_token:
        return config.hf_token
    for var in _TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


_GATING_HINT = (
    "Speaker diarization needs a gated Hugging Face model and a token. (1) Accept the "
    "conditions at https://hf.co/{model} (and any model it depends on); (2) create a READ "
    "token at https://hf.co/settings/tokens; (3) set HF_TOKEN=... (or pass --hf-token). "
    "First run downloads ~GBs, then it works offline."
)


class PyannoteDiarizer:
    """A pyannote.audio 4.x diarization pipeline, loaded lazily and reused across calls."""

    def __init__(self, config: DiarizeConfig) -> None:
        self._config = config
        self._pipeline: Any = None  # pyannote Pipeline, built lazily on first call

    def _load(self) -> Any:
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed.",
                hint='Install the diarization extra: uv tool install "hearsay[diarize]" '
                "(pulls in torch; ~2-3 GB).",
            ) from exc
        token = resolve_token(self._config)
        try:
            pipeline = Pipeline.from_pretrained(self._config.model, token=token)
        except Exception as exc:  # gated/unauthorized/offline/etc.
            raise DiarizationError(
                f"Could not load the diarization model '{self._config.model}': {exc}",
                hint=_GATING_HINT.format(model=self._config.model),
            ) from exc
        if pipeline is None:  # pyannote returns None on a silent auth failure
            raise DiarizationError(
                f"Diarization model '{self._config.model}' could not be loaded (not authorized?).",
                hint=_GATING_HINT.format(model=self._config.model),
            )
        return pipeline

    def __call__(self, audio_path: Path) -> list[SpeakerTurn]:
        if self._pipeline is None:
            self._pipeline = self._load()
        kwargs: dict[str, int] = {}
        if self._config.min_speakers is not None:
            kwargs["min_speakers"] = self._config.min_speakers
        if self._config.max_speakers is not None:
            kwargs["max_speakers"] = self._config.max_speakers
        try:
            result = self._pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            raise DiarizationError(
                f"Diarization failed for {audio_path}: {exc}",
                hint="Check the audio is valid and ffmpeg is installed (torchcodec needs it).",
            ) from exc
        # pyannote 4.x returns a DiarizeOutput dataclass; 3.x returned a bare Annotation.
        # Wrap parsing too, so a future API shape-shift degrades to an actionable error.
        try:
            annotation = getattr(result, "speaker_diarization", result)
            turns = [
                SpeakerTurn(
                    speaker=str(label), start_s=float(segment.start), end_s=float(segment.end)
                )
                for segment, _track, label in annotation.itertracks(yield_label=True)
            ]
        except (AttributeError, TypeError) as exc:
            raise DiarizationError(
                f"Unexpected diarization output {type(result).__name__!r} "
                "(expected a pyannote 4.x DiarizeOutput or 3.x Annotation).",
                hint="Check your installed pyannote.audio version is supported.",
            ) from exc
        return turns


def load_diarizer(config: DiarizeConfig) -> Diarizer:
    """Build a diarizer from config (currently pyannote). Raises DiarizationError if absent."""
    return PyannoteDiarizer(config)
