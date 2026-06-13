"""Tests for the Transcript JSON model, the sidecar, and the exported schema."""

import json
from pathlib import Path

import pytest

from hearsay.models import (
    Document,
    Paragraph,
    Section,
    SourceMetadata,
    Transcript,
    transcript_schema_json,
)

REPO = Path(__file__).resolve().parent.parent


def _doc() -> Document:
    meta = SourceMetadata(
        title="A Talk",
        source="https://www.youtube.com/watch?v=abc123def45",
        channel="Chan",
        duration_s=3723.0,
        video_id="abc123def45",
    )
    return Document(
        meta=meta,
        method="captions",
        language="en",
        ingested_at="2026-06-13T10:00:00Z",
        sections=[
            Section(
                title="Intro",
                start_s=0,
                paragraphs=[
                    Paragraph(text="First paragraph.", start_s=0.0, end_s=5.0),
                    Paragraph(text="Second paragraph.", start_s=5.0, end_s=12.0),
                ],
            ),
            Section(
                title="[00:05:00 – 00:09:58]",
                start_s=300,
                paragraphs=[Paragraph(text="Later.", start_s=300.0, end_s=305.0)],
            ),
        ],
    )


def test_transcript_from_document_flattens_chunks() -> None:
    transcript = Transcript.from_document(_doc())
    assert transcript.title == "A Talk"
    assert transcript.duration == "01:02:03"  # formatted from duration_s
    assert transcript.method == "captions"
    assert len(transcript.chunks) == 3
    first, second, third = transcript.chunks
    assert first.section == "Intro" and first.text == "First paragraph."
    assert second.section == "Intro"
    assert third.section == "[00:05:00 – 00:09:58]"
    # Chunks carry timing and stay in document order.
    assert [c.start_s for c in transcript.chunks] == [0.0, 5.0, 300.0]


def test_transcript_json_roundtrips_and_validates() -> None:
    transcript = Transcript.from_document(_doc())
    payload = transcript.model_dump_json(indent=2)
    reparsed = Transcript.model_validate_json(payload)
    assert reparsed == transcript


def test_committed_schema_matches_model() -> None:
    committed = (REPO / "docs" / "schema.json").read_text()
    assert committed == transcript_schema_json(), (
        "docs/schema.json is stale — run: uv run python scripts/export_schema.py"
    )


def test_schema_documents_chunk_fields() -> None:
    schema = json.loads(transcript_schema_json())
    chunk = schema["$defs"]["Chunk"]["properties"]
    assert set(chunk) == {"start_s", "end_s", "section", "text"}
    assert set(schema["required"]) >= {
        "title",
        "source",
        "channel",
        "duration",
        "method",
        "language",
    }


def test_negative_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        Transcript.model_validate(
            {
                "title": "t",
                "source": "s",
                "channel": "c",
                "duration": "00:00:00",
                "ingested": "t",
                "method": "captions",
                "language": "en",
                "chunks": [{"start_s": -1.0, "end_s": 1.0, "section": "x", "text": "y"}],
            }
        )
