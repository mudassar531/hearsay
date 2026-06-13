"""Tests for the ingestion pipeline (offline, fixture-driven via injection)."""

import json
from pathlib import Path

import pytest

from hearsay.captions import CaptionResult, normalize_snippets
from hearsay.errors import InvalidSourceError
from hearsay.models import Document, Segment
from hearsay.pipeline import (
    build_document,
    ingest_file,
    ingest_youtube,
    ingest_youtube_transcribe,
)
from hearsay.render import render_markdown
from hearsay.transcribe import TranscriptionResult
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


def test_render_full_chain_from_chapter_fixture() -> None:
    # Rendering, exercised against real fixture data end to end.
    meta = parse_metadata(load_meta("zjkBMFhNj_g"), "https://www.youtube.com/watch?v=zjkBMFhNj_g")
    doc = build_document(meta, load_captions("zjkBMFhNj_g"), ingested_at="2026-06-13T10:00:00Z")
    markdown = render_markdown(doc)

    lines = markdown.splitlines()
    assert lines[0] == "---"
    assert 'title: "[1hr Talk] Intro to Large Language Models"' in lines
    assert 'method: "captions-auto"' in lines
    assert "# [1hr Talk] Intro to Large Language Models" in lines
    # Every chapter that received paragraphs renders as a `##` heading.
    heading_count = sum(1 for line in lines if line.startswith("## "))
    assert heading_count == len(doc.sections)
    # Each paragraph carries a bold HH:MM:SS timestamp marker.
    assert markdown.count("**[") == sum(len(s.paragraphs) for s in doc.sections)
    # Lossless: the body words equal the source caption words.
    body = markdown.split("\n---\n", 1)[-1]
    body_words = [w for w in body.split() if not w.startswith(("#", "**[", "[")) and ":" not in w]
    assert "language" in body_words and "models" in body_words


# --- Whisper paths (offline via injected transcriber / downloader) --------


def _fake_transcription(model_size: str) -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            Segment(text="Hello and welcome to the show.", start_s=0.0, end_s=4.0),
            Segment(text="Today we talk about audio.", start_s=4.0, end_s=8.0),
        ],
        language="en",
        duration_s=8.0,
        model_size=model_size,
    )


def test_ingest_file_transcribes_with_injected_transcriber(tmp_path: Path) -> None:
    clip = tmp_path / "my recording.mp3"
    clip.write_bytes(b"not really audio")

    def fake_transcriber(path, **kwargs):
        assert path == clip
        return _fake_transcription(kwargs["model_size"])

    doc = ingest_file(
        clip,
        model_size="tiny",
        transcriber=fake_transcriber,
        now=lambda: "2026-06-13T10:00:00Z",
    )
    assert doc.method == "whisper-tiny"
    assert doc.meta.title == "my recording"
    assert doc.meta.channel == "Local file"
    assert doc.meta.duration_s == 8.0
    assert doc.meta.video_id == "my recording"
    body = " ".join(p.text for s in doc.sections for p in s.paragraphs)
    assert "welcome to the show" in body


def test_ingest_file_threads_vad_filter(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x")
    seen: dict[str, object] = {}

    def fake_transcriber(path, **kwargs):
        seen["vad_filter"] = kwargs.get("vad_filter")
        return _fake_transcription(kwargs["model_size"])

    # Default is VAD on (speech)...
    ingest_file(clip, transcriber=fake_transcriber, now=lambda: "t")
    assert seen["vad_filter"] is True
    # ...and it can be turned off for music.
    ingest_file(clip, vad_filter=False, transcriber=fake_transcriber, now=lambda: "t")
    assert seen["vad_filter"] is False


def test_ingest_file_missing_path() -> None:
    with pytest.raises(InvalidSourceError) as excinfo:
        ingest_file(Path("/no/such/file.mp3"))
    assert "not found" in excinfo.value.message.lower()


def test_ingest_file_unsupported_extension(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    with pytest.raises(InvalidSourceError) as excinfo:
        ingest_file(notes)
    assert "unsupported" in excinfo.value.message.lower()
    assert excinfo.value.hint


def test_ingest_youtube_transcribe_keeps_metadata_and_cleans_temp(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_metadata(url: str) -> dict:
        return load_meta("zjkBMFhNj_g")  # has chapters

    def fake_downloader(url: str, dest_dir: Path) -> Path:
        seen["dest_dir"] = dest_dir
        assert dest_dir.exists()  # a real temp dir during the call
        audio = dest_dir / "audio.m4a"
        audio.write_bytes(b"x")
        return audio

    def fake_transcriber(path, **kwargs):
        return _fake_transcription(kwargs["model_size"])

    doc = ingest_youtube_transcribe(
        "https://www.youtube.com/watch?v=zjkBMFhNj_g",
        model_size="small",
        metadata_fetcher=fake_metadata,
        audio_downloader=fake_downloader,
        transcriber=fake_transcriber,
        now=lambda: "2026-06-13T10:00:00Z",
    )
    assert doc.method == "whisper-small"
    assert doc.meta.title == "[1hr Talk] Intro to Large Language Models"
    assert doc.meta.chapters  # chapters preserved from metadata
    # The temp directory is deleted once the context manager exits.
    assert not Path(str(seen["dest_dir"])).exists()
