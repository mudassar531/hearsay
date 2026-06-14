"""Local speech-to-text with a pluggable backend.

Two engines are supported and selected by model name:

* **Parakeet** (``parakeet`` / ``parakeet-en``) — NVIDIA Parakeet TDT running on
  Apple's MLX. On Apple Silicon it transcribes ~3x faster than ``whisper-small``
  on CPU (~24x vs ~7x realtime on an M1 Pro), with comparable accuracy.
  Multilingual (v3) or English-only (v2).
* **Whisper** (``tiny``…``large-v3``) — faster-whisper / ctranslate2 on CPU,
  int8. Portable everywhere; the fallback when Parakeet is unavailable.

``auto`` (the default) picks Parakeet when it is installed and runnable, else
``whisper-small`` — so every platform gets the fastest engine it has.

Both backends are imported lazily inside the functions here: the captions-only
path, and the engine you are not using, never pay the (heavy, slow) import cost.
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from hearsay.errors import TranscriptionError
from hearsay.models import Segment

# Whisper sizes, smallest (fastest) to largest (most accurate).
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")
# Back-compat alias (older imports referenced MODEL_SIZES).
MODEL_SIZES = WHISPER_SIZES

# Parakeet model aliases → MLX-community repo ids. ``parakeet`` is multilingual
# (25 European languages); ``parakeet-en`` is English-only. Override the repo
# for either alias with HEARSAY_PARAKEET_MODEL.
PARAKEET_MODELS = {
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
    "parakeet-en": "mlx-community/parakeet-tdt-0.6b-v2",
}

# Whisper size used when ``auto`` falls back off Parakeet.
DEFAULT_WHISPER = "small"
# Default model: resolve the fastest available engine at transcription time.
DEFAULT_MODEL = "auto"

# Long files are transcribed in overlapping windows so progress can be reported
# and memory stays bounded; shorter files run in a single pass.
_PARAKEET_CHUNK_S = 120.0

# Reports (processed_seconds, total_seconds) as transcription advances.
ProgressCallback = Callable[[float, float], None]


class TranscriptionResult(NamedTuple):
    """Engine output mapped onto hearsay's segment model.

    ``method`` is the document-facing label (e.g. ``whisper-small`` or
    ``parakeet-tdt-0.6b-v3``); ``model_size`` is the requested model identifier.
    """

    segments: list[Segment]
    language: str
    duration_s: float
    model_size: str
    method: str


def transcribe_audio(
    path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    local_files_only: bool = False,
    vad_filter: bool = True,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Transcribe an audio/video file into timed segments.

    Args:
        path: Path to a local media file (any container ffmpeg can decode).
        model_size: ``auto`` (default), a Parakeet alias (``parakeet`` /
            ``parakeet-en``), or a Whisper size (``tiny``…``large-v3``).
        language: Force a language code, or ``None`` to auto-detect (Whisper).
            Parakeet auto-detects internally; this only labels the output.
        local_files_only: If True, never reach the network — load the model
            from the local cache or fail. Used by the offline test suite.
        vad_filter: Voice-activity detection (Whisper only). ``True`` (default)
            skips silence/music beds — right for speech. Set ``False`` for
            music/songs. Parakeet ignores this (it has no VAD pre-filter).
        on_progress: Optional callback invoked with (processed_s, total_s).

    Raises:
        TranscriptionError: the model could not be loaded (e.g. not cached and
            offline, or Parakeet requested but not installed) or the file could
            not be decoded.
    """
    engine, identifier = _resolve_engine(model_size)
    if engine == "parakeet":
        return _transcribe_parakeet(
            path,
            repo_id=identifier,
            language=language,
            local_files_only=local_files_only,
            on_progress=on_progress,
        )
    return _transcribe_whisper(
        path,
        model_size=identifier,
        language=language,
        local_files_only=local_files_only,
        vad_filter=vad_filter,
        on_progress=on_progress,
    )


def _resolve_engine(model_size: str) -> tuple[str, str]:
    """Map a model name to an ``(engine, identifier)`` pair.

    ``auto`` resolves to Parakeet when it is importable, else Whisper. An
    explicit Parakeet alias stays Parakeet (and fails loudly later if it cannot
    load), so an explicit choice is never silently downgraded.
    """
    if model_size == "auto":
        if _parakeet_available():
            return "parakeet", PARAKEET_MODELS["parakeet"]
        return "whisper", DEFAULT_WHISPER
    if model_size in PARAKEET_MODELS:
        repo = os.environ.get("HEARSAY_PARAKEET_MODEL") or PARAKEET_MODELS[model_size]
        return "parakeet", repo
    if model_size in WHISPER_SIZES:
        return "whisper", model_size
    raise TranscriptionError(
        f"Unknown transcription model: {model_size!r}",
        hint=f"Choose one of: auto, {', '.join(PARAKEET_MODELS)}, {', '.join(WHISPER_SIZES)}",
    )


def _parakeet_available() -> bool:
    """True when parakeet-mlx can be imported (Apple Silicon + extra installed)."""
    from importlib.util import find_spec

    try:
        return find_spec("parakeet_mlx") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


# --- Parakeet (MLX) backend ----------------------------------------------


def _transcribe_parakeet(
    path: Path,
    *,
    repo_id: str,
    language: str | None,
    local_files_only: bool,
    on_progress: ProgressCallback | None,
) -> TranscriptionResult:
    """Transcribe via NVIDIA Parakeet on MLX (Apple Silicon GPU/Neural Engine)."""
    model = _load_parakeet(repo_id, local_files_only=local_files_only)
    sample_rate = model.preprocessor_config.sample_rate

    def _chunk_cb(end_samples: float, total_samples: float) -> None:
        if on_progress is not None and total_samples > 0:
            on_progress(min(end_samples, total_samples) / sample_rate, total_samples / sample_rate)

    try:
        result = model.transcribe(
            str(path),
            chunk_duration=_PARAKEET_CHUNK_S,
            chunk_callback=_chunk_cb,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Could not transcribe {path}: {exc}",
            hint="Check the file is a valid audio/video file and ffmpeg is installed.",
        ) from exc

    segments: list[Segment] = []
    for sent in result.sentences:
        text = sent.text.strip()
        if not text:
            continue
        start_s = max(0.0, float(sent.start))
        end_s = max(float(sent.end), start_s)
        segments.append(Segment(text=text, start_s=start_s, end_s=end_s))

    duration_s = segments[-1].end_s if segments else 0.0
    if on_progress is not None:
        on_progress(duration_s, duration_s)
    label = repo_id.split("/")[-1]
    return TranscriptionResult(
        segments=segments,
        language=language or "en",
        duration_s=duration_s,
        model_size=label,
        method=label,
    )


def _load_parakeet(repo_id: str, *, local_files_only: bool):
    """Load a Parakeet MLX model, mapping failures to friendly errors."""
    try:
        from parakeet_mlx import from_pretrained
    except ImportError as exc:
        raise TranscriptionError(
            "parakeet-mlx is not installed.",
            hint=(
                "Install the fast Apple-Silicon engine with "
                'uv tool install "hearsay[parakeet]" (macOS arm64), '
                "or use --model small for CPU Whisper."
            ),
        ) from exc
    # parakeet-mlx has no offline-only switch; force Hugging Face into offline
    # mode so a cached model loads without any network round-trip. The env var
    # alone is read at import time, so we also flip the live module constant.
    with _hf_offline(local_files_only):
        try:
            return from_pretrained(repo_id)
        except Exception as exc:
            raise TranscriptionError(
                f"Could not load the Parakeet model '{repo_id}': {exc}",
                hint=(
                    "The model downloads once (~600MB). Check your network on first "
                    "use, then it is cached for offline runs."
                ),
            ) from exc


@contextmanager
def _hf_offline(enabled: bool):
    """Temporarily force ``huggingface_hub`` offline when ``enabled``.

    The ``HF_HUB_OFFLINE`` env var is read into a module constant at import
    time, so setting it now is too late if hf_hub is already imported — we flip
    the live constant as well, then restore both on exit.
    """
    if not enabled:
        yield
        return
    prior_env = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from huggingface_hub import constants

        prior_const = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = True
    except Exception:  # pragma: no cover - hf_hub always present with parakeet
        constants = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if prior_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior_env
        if constants is not None:
            constants.HF_HUB_OFFLINE = prior_const


# --- Whisper (faster-whisper) backend ------------------------------------


def _transcribe_whisper(
    path: Path,
    *,
    model_size: str,
    language: str | None,
    local_files_only: bool,
    vad_filter: bool,
    on_progress: ProgressCallback | None,
) -> TranscriptionResult:
    """Transcribe via faster-whisper on CPU (int8)."""
    model = _load_whisper(model_size, local_files_only=local_files_only)
    # whisper's segment iterator is lazy — the actual decode happens as we
    # consume it, so live progress is reported here. Only this loop can fail on
    # a bad file; Segment construction is done afterward so a model invariant
    # slip (e.g. end < start) surfaces honestly, not as a bogus "bad file".
    raw: list[tuple[str, float, float]] = []
    try:
        segment_iter, info = model.transcribe(str(path), language=language, vad_filter=vad_filter)
        for seg in segment_iter:
            raw.append((seg.text, seg.start, seg.end))
            if on_progress is not None:
                on_progress(min(seg.end, info.duration), info.duration)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not transcribe {path}: {exc}",
            hint="Check the file is a valid audio/video file and ffmpeg is installed.",
        ) from exc

    segments: list[Segment] = []
    for text, start, end in raw:
        text = text.strip()
        if not text:
            continue
        start_s = max(0.0, start)
        end_s = max(end, start_s)  # whisper occasionally yields end < start
        segments.append(Segment(text=text, start_s=start_s, end_s=end_s))
    if on_progress is not None:
        on_progress(info.duration, info.duration)
    return TranscriptionResult(
        segments=segments,
        language=info.language or "en",
        duration_s=float(info.duration),
        model_size=model_size,
        method=f"whisper-{model_size}",
    )


def _load_whisper(model_size: str, *, local_files_only: bool):
    """Load a faster-whisper model on CPU, mapping failures to friendly errors."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TranscriptionError(
            "faster-whisper is not installed.",
            hint="Reinstall hearsay so transcription support is available.",
        ) from exc
    try:
        return WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load the '{model_size}' whisper model: {exc}",
            hint=(
                "The model downloads once (~tens of MB to ~1.5GB). Check your "
                "network on first use, then it is cached for offline runs."
            ),
        ) from exc
