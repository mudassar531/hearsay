"""Slice and probe audio for dataset clips via ffmpeg/ffprobe (subprocess).

hearsay already documents ffmpeg as a system requirement and shells out to yt-dlp
the same way (``youtube.py``), so dataset mode reuses that pattern rather than
adding a Python audio dependency. Clips are produced by **re-encoding** (never
``-c copy``, which snaps cuts to keyframes); for 16-bit PCM WAV re-encoding is
lossless, so there is no quality cost. We use **input seeking** (``-ss`` before
``-i``), which is frame-accurate when transcoding (FFmpeg >= 2.1) and far faster
on long sources than output seeking — paired with ``-t DURATION`` because input
seeking resets the output timeline to zero.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hearsay.errors import AudioExportError

_SLICE_TIMEOUT_S = 300
_PROBE_TIMEOUT_S = 60


def ensure_tools() -> None:
    """Verify ffmpeg and ffprobe are on PATH, with an actionable error if not."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise AudioExportError(
                f"{tool} was not found on your PATH.",
                hint=(
                    "Dataset export needs ffmpeg. Install it (macOS: brew install ffmpeg; "
                    "Debian/Ubuntu: sudo apt install ffmpeg) — see the README Requirements section."
                ),
            )


def slice_clip(
    source: Path,
    start_s: float,
    end_s: float,
    dest: Path,
    *,
    sample_rate: int = 22050,
) -> None:
    """Write ``source``'s ``[start_s, end_s]`` as a mono 16-bit PCM WAV at ``sample_rate``.

    Re-encodes (lossless for PCM) using input seeking + a duration, so the cut is
    frame-accurate and the timestamps map directly to the source. Raises
    AudioExportError on failure.
    """
    # Clamp the start once and derive the duration from the clamped start, so a
    # negative start can never make -ss and -t disagree (which would over-run the clip).
    start = max(0.0, start_s)
    duration = max(0.0, end_s - start)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SLICE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExportError(
            f"ffmpeg timed out slicing {dest.name} after {_SLICE_TIMEOUT_S}s.",
            hint="Try a shorter source, or check ffmpeg is healthy.",
        ) from exc
    if proc.returncode != 0 or not dest.exists():
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no error output"
        raise AudioExportError(
            f"ffmpeg could not slice clip {dest.name}: {tail}",
            hint="Check the source file is valid audio/video and ffmpeg supports it.",
        )


def probe_duration(path: Path) -> float:
    """Return the duration (seconds) of an audio file via ffprobe, or 0.0 if unknown."""
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExportError(
            f"ffprobe timed out reading {path.name}.",
            hint="Check ffprobe is healthy and the file is valid.",
        ) from exc
    out = proc.stdout.strip()
    try:
        return max(0.0, float(out))
    except ValueError:
        return 0.0
