"""Pydantic models for the dataset-export mode."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from hearsay.models import Word

# The dataset index formats hearsay can export.
DATASET_FORMATS = ("ljspeech", "jsonl", "hf")
DEFAULT_FORMATS = ("ljspeech", "jsonl")


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


class DatasetClip(BaseModel):
    """One exported clip: a sliced WAV plus its verbatim transcript and timing.

    ``audio_path`` is relative to the dataset root (e.g. ``wavs/<id>.wav``) so the
    dataset is portable. ``start_s``/``end_s`` are the clip's span in the source
    media; ``duration_s`` is the *actual* duration of the written WAV (probed with
    ffprobe), which is what a training pipeline reads from the manifest.
    """

    id: str
    audio_path: str
    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    oversized: bool = False
    speaker: str | None = None  # set by optional diarization (Phase D5)


class DatasetConfig(BaseModel):
    """Knobs for a dataset build."""

    out_dir: Path
    formats: list[str] = Field(default_factory=lambda: list(DEFAULT_FORMATS))
    sample_rate: int = Field(default=22050, gt=0)
    segment_min_s: float = Field(default=1.0, gt=0)
    segment_max_s: float = Field(default=15.0, gt=0)


class BuildReport(BaseModel):
    """Summary of one dataset build: counts, totals, and the clips written."""

    out_dir: str
    source: str
    clip_count: int
    total_duration_s: float
    oversized_count: int
    sample_rate: int
    language: str
    formats: list[str]
    files: list[str] = Field(default_factory=list)  # index/card files written, relative
    clips: list[DatasetClip] = Field(default_factory=list)
