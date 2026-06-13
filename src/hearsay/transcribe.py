"""Local speech-to-text via faster-whisper (CPU).

faster-whisper (and its ctranslate2 backend) is heavy and slow to import, so
it is imported lazily inside the functions here — the captions-only path never
pays that cost. Audio decoding is handled by faster-whisper's bundled PyAV, so
common containers (mp3/m4a/wav/mp4/webm) are read directly without shelling out
to ffmpeg.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from hearsay.errors import TranscriptionError
from hearsay.models import Segment

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "small"

# Reports (processed_seconds, total_seconds) as transcription advances.
ProgressCallback = Callable[[float, float], None]


class TranscriptionResult(NamedTuple):
    """Whisper output mapped onto hearsay's segment model."""

    segments: list[Segment]
    language: str
    duration_s: float
    model_size: str


def transcribe_audio(
    path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    local_files_only: bool = False,
    vad_filter: bool = True,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Transcribe an audio/video file into timed segments (CPU, int8).

    Args:
        path: Path to a local media file (any container PyAV can decode).
        model_size: One of ``MODEL_SIZES``; larger is more accurate but slower.
        language: Force a language code, or ``None`` to auto-detect.
        local_files_only: If True, never reach the network — load the model
            from the local cache or fail. Used by the offline test suite.
        vad_filter: Voice-activity detection. ``True`` (default) skips silence
            and music beds — right for speech (podcasts, talks, meetings). Set
            ``False`` for music/songs, where VAD would otherwise discard the
            sung audio as "non-speech" (much slower, less hallucination-proof).
        on_progress: Optional callback invoked with (processed_s, total_s).

    Raises:
        TranscriptionError: the model could not be loaded (e.g. not cached and
            offline) or the file could not be decoded.
    """
    if model_size not in MODEL_SIZES:
        raise TranscriptionError(
            f"Unknown whisper model: {model_size!r}",
            hint=f"Choose one of: {', '.join(MODEL_SIZES)}",
        )

    model = _load_model(model_size, local_files_only=local_files_only)
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
    )


def _load_model(model_size: str, *, local_files_only: bool):
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
