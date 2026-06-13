"""Pydantic data models shared across the ingestion pipeline."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from hearsay.timefmt import format_timestamp


class Chapter(BaseModel):
    """A chapter marker from the source video."""

    title: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)


class SourceMetadata(BaseModel):
    """Metadata about an ingested source, used for the frontmatter."""

    title: str
    source: str
    channel: str
    duration_s: float = Field(ge=0)
    video_id: str
    chapters: list[Chapter] = Field(default_factory=list)


class Segment(BaseModel):
    """A raw timed text fragment (caption snippet or whisper segment)."""

    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)


class Paragraph(BaseModel):
    """A readable paragraph grouped from segments; the unit of output."""

    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated words in the paragraph."""
        return len(self.text.split())


class Section(BaseModel):
    """A `##` section of the document: a chapter or a time-based slice."""

    title: str
    start_s: float = Field(ge=0)
    paragraphs: list[Paragraph] = Field(default_factory=list)


class Document(BaseModel):
    """Everything the renderer needs to produce the markdown file."""

    meta: SourceMetadata
    method: str
    language: str
    ingested_at: str
    sections: list[Section] = Field(default_factory=list)


class Chunk(BaseModel):
    """One paragraph of the transcript, tagged with its section, for the JSON sidecar."""

    start_s: float = Field(ge=0, description="Start time of the chunk, in seconds.")
    end_s: float = Field(ge=0, description="End time of the chunk, in seconds.")
    section: str = Field(
        description="Title of the section (chapter or time span) this chunk is in."
    )
    text: str = Field(description="The paragraph text.")


class Transcript(BaseModel):
    """The JSON sidecar: source metadata plus the ordered transcript chunks.

    This is the stable, machine-readable contract for RAG pipelines and agents;
    its JSON Schema is exported to docs/schema.json.
    """

    title: str = Field(description="Title of the video / episode / file.")
    source: str = Field(description="Original URL or file path.")
    channel: str = Field(description="Channel, show, or 'Local file'.")
    duration: str = Field(description="Total duration as HH:MM:SS.")
    ingested: str = Field(description="UTC ingestion time, ISO-8601 with trailing Z.")
    method: str = Field(
        description='How the text was obtained: "captions", "captions-auto", or "whisper-<model>".'
    )
    language: str = Field(description="BCP-47-ish language code of the transcript.")
    chunks: list[Chunk] = Field(default_factory=list, description="Ordered transcript paragraphs.")

    @classmethod
    def from_document(cls, doc: Document) -> Transcript:
        """Flatten a Document's sections into a chunk list (pure)."""
        chunks = [
            Chunk(start_s=p.start_s, end_s=p.end_s, section=section.title, text=p.text)
            for section in doc.sections
            for p in section.paragraphs
        ]
        return cls(
            title=doc.meta.title,
            source=doc.meta.source,
            channel=doc.meta.channel,
            duration=format_timestamp(doc.meta.duration_s),
            ingested=doc.ingested_at,
            method=doc.method,
            language=doc.language,
            chunks=chunks,
        )


def transcript_schema_json() -> str:
    """The Transcript JSON Schema as pretty-printed, newline-terminated JSON.

    The canonical source for both scripts/export_schema.py and the test that
    asserts docs/schema.json stays in sync.
    """
    return json.dumps(Transcript.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
