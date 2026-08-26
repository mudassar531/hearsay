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

import json
import re
import shutil
import subprocess
from pathlib import Path

from hearsay.errors import AudioExportError

_SLICE_TIMEOUT_S = 300
_PROBE_TIMEOUT_S = 60


# ffmpeg prints its version, build flags and every library line before the real error,
# and faster-whisper surfaces that whole banner too. Showing the last line is usually
# right, but these lines are never the error — drop them so the message a user reads is
# the thing that actually went wrong.
_FFMPEG_NOISE = re.compile(
    r"^\s*(ffmpeg version|built with|configuration:|lib[a-z]+\s+\d|Input #|Output #|"
    r"Stream #|Metadata:|Duration:|\s+encoder|\s+creation_time)",
)


def _error_tail(stderr: str) -> str:
    """The last meaningful line of ffmpeg stderr, minus its banner."""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    useful = [ln.strip() for ln in lines if not _FFMPEG_NOISE.match(ln)]
    return (useful or lines or ["no error output"])[-1][:300]


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


# EBU R128 loudness target for --normalize: -23 LUFS integrated, -1.5 dBTP true-peak
# ceiling (conservative; EBU max is -1.0), 7 LU loudness range.
_LOUDNORM = "loudnorm=I=-23:TP=-1.5:LRA=7"
_JSON_BLOCK = re.compile(r"\{[^{}]*\}")


def ensure_filter(name: str) -> None:
    """Verify the running ffmpeg build includes filter ``name`` (e.g. for --normalize)."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise AudioExportError(
            f"Could not query ffmpeg filters: {exc}", hint="Check ffmpeg is healthy."
        ) from exc
    if f" {name} " not in proc.stdout:
        raise AudioExportError(
            f"Your ffmpeg build lacks the '{name}' filter, needed for --normalize.",
            hint="Install a full ffmpeg build (the official static builds), or drop --normalize.",
        )


def slice_clip(
    source: Path,
    start_s: float,
    end_s: float,
    dest: Path,
    *,
    sample_rate: int = 22050,
    normalize: bool = False,
    fade_s: float = 0.0,
) -> None:
    """Write ``source``'s ``[start_s, end_s]`` as a mono 16-bit PCM WAV at ``sample_rate``.

    Re-encodes (lossless for PCM) using input seeking + a duration, so the cut is
    frame-accurate and the timestamps map directly to the source. ``normalize`` applies
    two-pass EBU R128 loudness normalization (``loudnorm``, length-preserving). ``fade_s``
    (when > 0 and the clip is longer than two fades) applies a short in/out fade that
    removes the click/pop from cutting on a non-zero sample, without changing the clip's
    length. Raises AudioExportError on failure.
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
    ]
    # Build the -af chain: loudnorm first (measure -> linear apply, length-preserving),
    # then a short edge fade to de-click the boundaries. Both run before the output -ar.
    filters: list[str] = []
    if normalize:
        filters.append(_loudnorm_filter(source, start, duration))
    if fade_s > 0 and duration > 2 * fade_s:
        filters.append(f"afade=t=in:st=0:d={fade_s:.3f}")
        filters.append(f"afade=t=out:st={duration - fade_s:.3f}:d={fade_s:.3f}")
    if filters:
        args += ["-af", ",".join(filters)]
    args += [
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
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SLICE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExportError(
            f"ffmpeg timed out slicing {dest.name} after {_SLICE_TIMEOUT_S}s.",
            hint="Try a shorter source, or check ffmpeg is healthy.",
        ) from exc
    if proc.returncode != 0 or not dest.exists():
        tail = _error_tail(proc.stderr)
        raise AudioExportError(
            f"ffmpeg could not slice clip {dest.name}: {tail}",
            hint="Check the source file is valid audio/video and ffmpeg supports it.",
        )


def _loudnorm_filter(source: Path, start: float, duration: float) -> str:
    """Build a loudnorm filter for the segment, two-pass (measure then linear-apply).

    Single-pass dynamic loudnorm buffers/look-aheads and trims ~tens of ms off the
    clip; the two-pass form (measure on pass 1, ``linear=true`` with the measured
    values on pass 2) is length-preserving and more accurate. Falls back to plain
    single-pass only if the measurement can't be parsed.
    """
    args = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-af",
        f"{_LOUDNORM}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SLICE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return _LOUDNORM
    measured: dict[str, str] = {}
    for block in _JSON_BLOCK.findall(proc.stderr):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        if "input_i" in data:
            measured = data
    keys = ("input_i", "input_tp", "input_lra", "input_thresh")
    if not all(k in measured for k in keys):
        return _LOUDNORM  # measurement failed — fall back to single-pass
    i, tp, lra, thresh = (measured[k] for k in keys)
    return (
        f"{_LOUDNORM}:measured_I={i}:measured_TP={tp}:measured_LRA={lra}"
        f":measured_thresh={thresh}:linear=true"
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
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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


def probe_sample_rate(path: Path) -> int:
    """Return the audio sample rate (Hz) of the first audio stream, or 0 if unknown."""
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return 0
    try:
        return max(0, int(proc.stdout.strip()))
    except ValueError:
        return 0


# Diarization models (pyannote) run on 16 kHz mono. Decoding to that once, up front,
# is both what they want and a way around lossy-container decode drift.
DIARIZE_SAMPLE_RATE = 16000


def decode_to_wav(source: Path, dest: Path, *, sample_rate: int = DIARIZE_SAMPLE_RATE) -> Path:
    """Decode ``source`` in full to a mono 16-bit PCM WAV at ``sample_rate``.

    Compressed containers (MP3/M4A) decode to a frame-quantized number of samples,
    which can disagree with what a fixed-window consumer computes from the reported
    duration — pyannote 4.x raises on exactly that mismatch ("resulted in N samples
    instead of the expected M"). Handing it a PCM WAV makes the sample count exact,
    so diarization works on podcast audio, not just on WAV input. Returns ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
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
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SLICE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExportError(
            f"ffmpeg timed out decoding {source.name} after {_SLICE_TIMEOUT_S}s.",
            hint="Try a shorter source, or check ffmpeg is healthy.",
        ) from exc
    if proc.returncode != 0 or not dest.exists():
        tail = _error_tail(proc.stderr)
        raise AudioExportError(
            f"ffmpeg could not decode {source.name}: {tail}",
            hint="Check the source file is valid audio/video and ffmpeg supports it.",
        )
    return dest
