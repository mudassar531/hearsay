"""Pydantic models for the dataset-export mode."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hearsay.models import Word


class DatasetSegment(BaseModel):
    """One training clip: a contiguous run of words with a repaired span.

    Produced by ``segment_words``. ``words`` are the original, unmodified
    :class:`~hearsay.models.Word` objects (verbatim, never split); ``text`` is
    their whitespace-normalized join. ``start_s``/``end_s`` are the *extent* of
    the segment — the minimum word start and maximum word end after span
    repair — so the span always contains every member word's audio.

    ``oversized`` is ``True`` exactly when the clip's duration exceeds the
    configured ``max_s`` (an unbreakable run, e.g. a single very long word):
    the segmenter never splits a word to satisfy the duration bound, it flags
    the clip instead. A clip is never both within ``max_s`` and ``oversized``.
    """

    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    words: list[Word] = Field(default_factory=list)
    oversized: bool = False

    @property
    def duration_s(self) -> float:
        """Clip duration in seconds (``end_s - start_s``; always >= 0)."""
        return self.end_s - self.start_s
