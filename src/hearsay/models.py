"""Pydantic data models shared across the ingestion pipeline."""

from pydantic import BaseModel, Field


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
