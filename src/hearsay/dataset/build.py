"""Build a TTS/STT dataset from one source: transcribe -> segment -> slice -> index.

Orchestrates the dataset pipeline for a single audio source (a local file or one
YouTube video). Network/transcription steps are injected as callables so the
build logic is testable offline against fixtures. Phase D4 layers batch merging
(playlists/feeds) on top of ``build_dataset``.
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from hearsay import __version__
from hearsay.dataset.audio import ensure_tools, probe_duration, slice_clip
from hearsay.dataset.filters import detect_clipping_drops, filter_segments
from hearsay.dataset.formats import write_dataset_card, write_dropped, write_indices
from hearsay.dataset.models import (
    DATASET_FORMATS,
    BuildReport,
    DatasetClip,
    DatasetConfig,
    DatasetSegment,
    DropRecord,
)
from hearsay.dataset.segmentation import segment_words
from hearsay.errors import AudioExportError, InvalidSourceError
from hearsay.models import SourceMetadata, Word
from hearsay.transcribe import DEFAULT_MODEL, TranscriptionResult, transcribe_audio
from hearsay.youtube import download_audio, fetch_raw_metadata, parse_metadata

Transcriber = Callable[..., TranscriptionResult]
MetadataFetcher = Callable[[str], dict]
AudioDownloader = Callable[[str, Path], Path]

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
# A clip shorter than this (after slicing) is a near-empty ffmpeg artifact, not audio.
_MIN_CLIP_S = 0.01


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_id(text: str) -> str:
    """A filesystem-safe, collision-resistant id stem from a video id or title."""
    cleaned = _UNSAFE.sub("-", text).strip("-")
    return cleaned[:60] or "clip"


def _validate_formats(formats: list[str]) -> None:
    unknown = [f for f in formats if f not in DATASET_FORMATS]
    if unknown:
        raise AudioExportError(
            f"Unknown dataset format(s): {', '.join(unknown)}.",
            hint=f"Choose from: {', '.join(DATASET_FORMATS)}.",
        )


def build_dataset(
    source_audio: Path,
    words: list[Word],
    meta: SourceMetadata,
    *,
    config: DatasetConfig,
    language: str = "en",
    source_platform: str = "local",
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
) -> BuildReport:
    """Segment ``words``, slice ``source_audio`` into clips, and write the dataset.

    Writes ``wavs/<id>_NNNN.wav`` plus the requested index formats and a
    ``dataset_card.md`` into ``config.out_dir``. Returns a :class:`BuildReport`.
    Raises AudioExportError when ffmpeg/ffprobe are missing or a slice fails.
    """
    ensure_tools()
    _validate_formats(config.formats)
    segments = segment_words(words, min_s=config.segment_min_s, max_s=config.segment_max_s)

    out_dir = config.out_dir
    try:
        (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioExportError(
            f"Could not create the dataset directory {out_dir}: {exc.strerror or exc}",
            hint="Pick a writable --out path whose parent is not an existing file.",
        ) from exc

    # Clamp clip ends to the true media length so a segment whose last word ends a
    # hair past the audio (common) yields a real, shorter clip — and one entirely
    # past EOF is skipped rather than written as a 0-second "ghost" clip.
    source_duration = probe_duration(source_audio)
    source_id = _safe_id(meta.video_id or meta.title)

    # Pre-slice candidates (clamped span); Tier-1 filters run here so we never
    # waste an ffmpeg slice on a clip we are going to drop.
    candidates: list[tuple[str, DatasetSegment, float]] = []
    spans: dict[str, tuple[float, float]] = {}
    for seg_index, seg in enumerate(segments, start=1):
        start = seg.start_s
        end = min(seg.end_s, source_duration) if source_duration > 0 else seg.end_s
        if end - start <= 0:
            continue  # nothing left to slice (segment lies past the source end)
        ref = f"{source_id}_seg{seg_index:04d}"
        candidates.append((ref, seg, end - start))
        spans[ref] = (start, end)

    # Tier-1 quality filters (default on). Note an "oversized" clip (longer than
    # segment_max_s) is dropped here by the duration filter whenever
    # filters.max_duration_s == segment_max_s (the defaults) — intended.
    kept, drops = filter_segments(candidates, config.filters)

    clips: list[DatasetClip] = []
    index = 0
    for ref, seg, _duration in kept:
        start, end = spans[ref]
        index += 1
        clip_id = f"{source_id}_{index:04d}"
        rel = f"wavs/{clip_id}.wav"
        dest = out_dir / rel
        slice_clip(source_audio, start, end, dest, sample_rate=config.sample_rate)
        duration_s = probe_duration(dest)
        if duration_s <= _MIN_CLIP_S:
            dest.unlink(missing_ok=True)  # empty/near-empty slice — don't record it
            index -= 1
            drops.append(
                DropRecord(
                    clip=ref,
                    filter="empty_slice",
                    value="0.00s",
                    threshold=">0s",
                    text=seg.text[:80],
                )
            )
            continue
        clips.append(
            DatasetClip(
                id=clip_id,
                audio_path=rel,
                text=seg.text,
                start_s=start,
                end_s=end,
                duration_s=duration_s,
                oversized=seg.oversized,
            )
        )

    # Tier-2 (opt-in): drop hard-clipped clips and remove their WAVs.
    clips, clip_drops = detect_clipping_drops(clips, out_dir, config.filters)
    for drop in clip_drops:
        (out_dir / "wavs" / f"{drop.clip}.wav").unlink(missing_ok=True)
    drops.extend(clip_drops)

    files = write_indices(out_dir, clips, config.formats)
    files.append(write_dropped(out_dir, drops))
    report = BuildReport(
        out_dir=str(out_dir),
        source=meta.source,
        clip_count=len(clips),
        total_duration_s=sum(c.duration_s for c in clips),
        oversized_count=sum(1 for c in clips if c.oversized),
        sample_rate=config.sample_rate,
        language=language,
        formats=list(config.formats),
        files=files,
        clips=clips,
        dropped_count=len(drops),
        drops_by_reason=dict(Counter(d.filter for d in drops)),
        drops=drops,
    )
    card = write_dataset_card(
        out_dir, report, meta, version=version, generated_at=now(), source_platform=source_platform
    )
    report.files.append(card)
    return report


def build_dataset_from_file(
    path: Path,
    *,
    config: DatasetConfig,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    vad_filter: bool = True,
    transcriber: Transcriber = transcribe_audio,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
) -> BuildReport:
    """Transcribe a local file (with word timings) and build a dataset from it."""
    if not path.exists():
        raise InvalidSourceError(
            f"File not found: {path}", hint="Check the path, or pass a YouTube URL."
        )
    result = transcriber(
        path, model_size=model_size, language=language, vad_filter=vad_filter, word_timestamps=True
    )
    words = _require_words(result)
    meta = SourceMetadata(
        title=path.stem,
        source=str(path),
        channel="Local file",
        duration_s=result.duration_s,
        video_id=path.stem,
    )
    return build_dataset(
        path,
        words,
        meta,
        config=config,
        language=result.language,
        source_platform="local",
        version=version,
        now=now,
    )


def build_dataset_from_youtube(
    url: str,
    *,
    config: DatasetConfig,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    vad_filter: bool = True,
    metadata_fetcher: MetadataFetcher = fetch_raw_metadata,
    audio_downloader: AudioDownloader = download_audio,
    transcriber: Transcriber = transcribe_audio,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
) -> BuildReport:
    """Download a YouTube video's audio, transcribe it, and build a dataset.

    The downloaded audio is always deleted; the dataset is built before the temp
    directory is removed (clips are sliced from that audio).
    """
    raw = metadata_fetcher(url)
    meta = parse_metadata(raw, url)
    with tempfile.TemporaryDirectory(prefix="hearsay-ds-") as tmp:
        audio_path = audio_downloader(url, Path(tmp))
        result = transcriber(
            audio_path,
            model_size=model_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
        )
        words = _require_words(result)
        return build_dataset(
            audio_path,
            words,
            meta,
            config=config,
            language=result.language,
            source_platform="youtube",
            version=version,
            now=now,
        )


def _require_words(result: TranscriptionResult) -> list[Word]:
    """Return the result's words, or raise a friendly error if there are none."""
    if not result.words:
        raise AudioExportError(
            "Transcription produced no word-level timings, so no clips can be cut.",
            hint="The audio may be silent or non-speech; try a clearer file or another --model.",
        )
    return result.words
