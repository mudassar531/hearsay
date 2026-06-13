"""Orchestrate the YouTube → markdown ingestion pipeline.

Ties together metadata fetching, caption retrieval, paragraph grouping,
sectioning, and document assembly. The network-touching steps are injected
as callables so the assembly logic can be tested offline against fixtures.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from hearsay.captions import CaptionResult, fetch_captions
from hearsay.grouping import group_segments
from hearsay.models import Document, SourceMetadata
from hearsay.sectioning import sectionize
from hearsay.youtube import fetch_raw_metadata, parse_metadata

MetadataFetcher = Callable[[str], dict]
CaptionFetcher = Callable[[str, str], CaptionResult]


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_document(
    meta: SourceMetadata,
    captions: CaptionResult,
    *,
    ingested_at: str,
) -> Document:
    """Assemble a Document from metadata and captions (pure).

    Groups caption segments into paragraphs, sections them by chapter (or by
    ~5-minute windows when there are no chapters), and records the method as
    ``captions`` (manual) or ``captions-auto`` (auto-generated).
    """
    paragraphs = group_segments(captions.segments)
    sections = sectionize(paragraphs, meta.chapters)
    method = "captions-auto" if captions.is_generated else "captions"
    return Document(
        meta=meta,
        method=method,
        language=captions.language_code,
        ingested_at=ingested_at,
        sections=sections,
    )


def ingest_youtube(
    url: str,
    *,
    language: str = "en",
    metadata_fetcher: MetadataFetcher = fetch_raw_metadata,
    caption_fetcher: CaptionFetcher = fetch_captions,
    now: Callable[[], str] = _utc_now_iso,
) -> Document:
    """Ingest a YouTube URL into a Document via the captions path.

    The network steps default to the real fetchers but are injectable for
    offline testing. Raises the hearsay errors documented on the fetchers
    (VideoUnavailableError, NoCaptionsError, MetadataError, ...).
    """
    raw = metadata_fetcher(url)
    meta = parse_metadata(raw, url)
    captions = caption_fetcher(meta.video_id, language)
    return build_document(meta, captions, ingested_at=now())
