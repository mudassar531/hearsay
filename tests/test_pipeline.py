"""Tests for the ingestion pipeline (offline, fixture-driven via injection)."""

import json
from pathlib import Path

from hearsay.captions import CaptionResult, normalize_snippets
from hearsay.models import Document
from hearsay.pipeline import build_document, ingest_youtube
from hearsay.youtube import parse_metadata

FIXTURES = Path(__file__).parent / "fixtures"


def load_meta(video_id: str) -> dict:
    return json.loads((FIXTURES / f"{video_id}.meta.json").read_text())


def load_captions(video_id: str) -> CaptionResult:
    data = json.loads((FIXTURES / f"{video_id}.transcript.json").read_text())
    return CaptionResult(
        segments=normalize_snippets(data["snippets"]),
        language_code=data["language_code"],
        is_generated=data["is_generated"],
    )


def test_build_document_chapter_video() -> None:
    meta = parse_metadata(load_meta("zjkBMFhNj_g"), "url")
    doc = build_document(meta, load_captions("zjkBMFhNj_g"), ingested_at="2026-06-13T00:00:00Z")
    assert isinstance(doc, Document)
    # Auto-generated captions are recorded as captions-auto.
    assert doc.method == "captions-auto"
    assert doc.language == "en"
    # 21 chapters in the fixture; every section title should be a chapter title.
    chapter_titles = {c.title for c in meta.chapters}
    assert len(doc.sections) > 1
    assert all(s.title in chapter_titles for s in doc.sections)
    # Lossless through the whole pipeline.
    doc_words = " ".join(p.text for s in doc.sections for p in s.paragraphs).split()
    src_words = " ".join(seg.text for seg in load_captions("zjkBMFhNj_g").segments).split()
    assert doc_words == src_words


def test_build_document_no_chapters_uses_time_sections() -> None:
    meta = parse_metadata(load_meta("rStL7niR7gs"), "url")
    doc = build_document(meta, load_captions("rStL7niR7gs"), ingested_at="2026-06-13T00:00:00Z")
    assert doc.method == "captions"  # manual captions
    assert len(doc.sections) >= 1
    # Time-based section titles look like "[00:00:00 – 00:04:58]".
    assert all(s.title.startswith("[") for s in doc.sections)


def test_ingest_youtube_with_injected_fetchers() -> None:
    captured: dict[str, str] = {}

    def fake_metadata(url: str) -> dict:
        captured["url"] = url
        return load_meta("rStL7niR7gs")

    def fake_captions(video_id: str, language: str) -> CaptionResult:
        captured["video_id"] = video_id
        captured["language"] = language
        return load_captions("rStL7niR7gs")

    doc = ingest_youtube(
        "https://www.youtube.com/watch?v=rStL7niR7gs",
        language="en",
        metadata_fetcher=fake_metadata,
        caption_fetcher=fake_captions,
        now=lambda: "2026-06-13T10:00:00Z",
    )
    assert captured["video_id"] == "rStL7niR7gs"
    assert captured["language"] == "en"
    assert doc.ingested_at == "2026-06-13T10:00:00Z"
    assert doc.meta.title == "You Would Be a Terrible Leader"
