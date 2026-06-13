"""Orchestrate ingestion: YouTube captions, YouTube transcription, local files.

Ties together metadata, captions or whisper transcription, paragraph grouping,
sectioning, and document assembly. Network/transcription steps are injected as
callables so the assembly logic can be tested offline against fixtures.
"""

import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from hearsay.captions import CaptionResult, fetch_captions
from hearsay.errors import AudioDownloadError, InvalidSourceError, NoCaptionsError
from hearsay.feeds import Episode, download_episode
from hearsay.grouping import group_segments
from hearsay.models import Document, Segment, SourceMetadata
from hearsay.sectioning import sectionize
from hearsay.transcribe import DEFAULT_MODEL, TranscriptionResult, transcribe_audio
from hearsay.youtube import download_audio, fetch_raw_metadata, parse_metadata

MetadataFetcher = Callable[[str], dict]
CaptionFetcher = Callable[[str, str], CaptionResult]
AudioDownloader = Callable[[str, Path], Path]
EpisodeDownloader = Callable[[str, Path], Path]
Transcriber = Callable[..., TranscriptionResult]

_AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".mkv",
    ".mov",
}


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def assemble_document(
    meta: SourceMetadata,
    segments: list[Segment],
    *,
    method: str,
    language: str,
    ingested_at: str,
) -> Document:
    """Group segments into paragraphs and section them into a Document (pure)."""
    paragraphs = group_segments(segments)
    sections = sectionize(paragraphs, meta.chapters)
    return Document(
        meta=meta,
        method=method,
        language=language,
        ingested_at=ingested_at,
        sections=sections,
    )


def build_document(meta: SourceMetadata, captions: CaptionResult, *, ingested_at: str) -> Document:
    """Assemble a Document from metadata and fetched captions (pure)."""
    method = "captions-auto" if captions.is_generated else "captions"
    return assemble_document(
        meta,
        captions.segments,
        method=method,
        language=captions.language_code,
        ingested_at=ingested_at,
    )


def ingest_youtube(
    url: str,
    *,
    language: str = "en",
    metadata_fetcher: MetadataFetcher = fetch_raw_metadata,
    caption_fetcher: CaptionFetcher = fetch_captions,
    now: Callable[[], str] = _utc_now_iso,
) -> Document:
    """Ingest a YouTube URL via the captions path.

    Raises NoCaptionsError when the video has no captions (the CLI falls back
    to transcription), plus the other documented youtube/caption errors.
    """
    raw = metadata_fetcher(url)
    meta = parse_metadata(raw, url)
    captions = caption_fetcher(meta.video_id, language)
    return build_document(meta, captions, ingested_at=now())


def ingest_youtube_transcribe(
    url: str,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    metadata_fetcher: MetadataFetcher = fetch_raw_metadata,
    audio_downloader: AudioDownloader = download_audio,
    transcriber: Transcriber = transcribe_audio,
    on_progress: Callable[[float, float], None] | None = None,
    now: Callable[[], str] = _utc_now_iso,
) -> Document:
    """Ingest a YouTube URL by downloading its audio and transcribing it.

    Keeps the video's real metadata (title/channel/chapters) from yt-dlp, then
    transcribes the audio with whisper. The downloaded audio is always deleted.
    """
    raw = metadata_fetcher(url)
    meta = parse_metadata(raw, url)
    with tempfile.TemporaryDirectory(prefix="hearsay-") as tmp:
        audio_path = audio_downloader(url, Path(tmp))
        result = transcriber(
            audio_path, model_size=model_size, language=language, on_progress=on_progress
        )
    return assemble_document(
        meta,
        result.segments,
        method=f"whisper-{result.model_size}",
        language=result.language,
        ingested_at=now(),
    )


def ingest_file(
    path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    transcriber: Transcriber = transcribe_audio,
    on_progress: Callable[[float, float], None] | None = None,
    now: Callable[[], str] = _utc_now_iso,
) -> Document:
    """Ingest a local audio/video file by transcribing it with whisper."""
    if not path.exists():
        raise InvalidSourceError(
            f"File not found: {path}",
            hint="Check the path, or pass a YouTube URL.",
        )
    if path.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise InvalidSourceError(
            f"Unsupported file type: {path.suffix or '(none)'}",
            hint=f"Supported audio/video extensions: {', '.join(sorted(_AUDIO_EXTENSIONS))}",
        )
    result = transcriber(path, model_size=model_size, language=language, on_progress=on_progress)
    meta = SourceMetadata(
        title=path.stem,
        source=str(path),
        channel="Local file",
        duration_s=result.duration_s,
        video_id=path.stem,
    )
    return assemble_document(
        meta,
        result.segments,
        method=f"whisper-{result.model_size}",
        language=result.language,
        ingested_at=now(),
    )


def ingest_episode(
    episode: Episode,
    show_title: str,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    episode_downloader: EpisodeDownloader = download_episode,
    transcriber: Transcriber = transcribe_audio,
    on_progress: Callable[[float, float], None] | None = None,
    now: Callable[[], str] = _utc_now_iso,
) -> Document:
    """Ingest one podcast episode: download its audio and transcribe it.

    The downloaded audio is always deleted. Raises AudioDownloadError when the
    episode has no usable media URL.
    """
    if not episode.audio_url:
        raise AudioDownloadError(
            f"Episode '{episode.title}' has no audio enclosure.",
            hint="This feed entry has no downloadable media; try another episode.",
        )
    with tempfile.TemporaryDirectory(prefix="hearsay-") as tmp:
        audio_path = episode_downloader(episode.audio_url, Path(tmp))
        result = transcriber(
            audio_path, model_size=model_size, language=language, on_progress=on_progress
        )
    meta = SourceMetadata(
        title=episode.title,
        source=episode.audio_url,
        channel=show_title,
        duration_s=episode.duration_s or result.duration_s,
        video_id=episode.title,
    )
    return assemble_document(
        meta,
        result.segments,
        method=f"whisper-{result.model_size}",
        language=result.language,
        ingested_at=now(),
    )


__all__ = [
    "NoCaptionsError",
    "assemble_document",
    "build_document",
    "ingest_episode",
    "ingest_file",
    "ingest_youtube",
    "ingest_youtube_transcribe",
]
