"""Pydantic models for the dataset-export mode."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

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


class FilterConfig(BaseModel):
    """Quality-filter thresholds (Tier-1 default-on, dep-free; Tier-2 opt-in).

    All operate on data hearsay already has — clip duration, word timings, the
    transcript — so the default set needs no audio decode and no new dependency.
    ``min_avg_confidence`` is an engine-agnostic stand-in for Whisper's
    ``avg_logprob``/``no_speech_prob`` (linear word probability; ~0.30 ≈
    ``avg_logprob`` -1.0), since clips are re-grouped from words and Parakeet
    exposes no segment-level decoding signals (see DECISIONS).
    """

    enabled: bool = True
    min_duration_s: float = 1.0
    max_duration_s: float = 15.0
    max_internal_gap_s: float = 2.0  # an internal silence this long = a join of two utterances
    max_compression_ratio: float = 2.4  # gzip-ratio repetition guard (Whisper's threshold)
    min_avg_confidence: float = 0.30  # mean word confidence floor (~avg_logprob -1.0)
    min_chars_per_s: float = 5.0
    max_chars_per_s: float = 25.0
    target_language: str = "en"
    require_target_script: bool = True  # drop clips whose text is mostly the wrong script
    detect_clipping: bool = False  # Tier-2: read the WAV and drop clipped clips (opt-in)


DIARIZE_MODES = ("tag", "dominant", "per_speaker")


class DiarizeConfig(BaseModel):
    """Optional speaker-diarization knobs (the ``hearsay[diarize]`` extra).

    ``mode``: ``tag`` labels every clip with its speaker; ``dominant`` keeps only
    each source's most-spoken speaker (single-voice TTS); ``per_speaker`` also emits
    a per-speaker index. A clip is dropped as cross-speaker when its dominant
    speaker covers less than ``min_purity`` of its overlapped speech.
    """

    enabled: bool = False
    mode: str = "tag"
    model: str = "pyannote/speaker-diarization-community-1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    min_purity: float = 0.85
    hf_token: str | None = None

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, value: str) -> str:
        if value not in DIARIZE_MODES:
            raise ValueError(f"mode must be one of {DIARIZE_MODES}, got {value!r}")
        return value


class DatasetConfig(BaseModel):
    """Knobs for a dataset build."""

    out_dir: Path
    formats: list[str] = Field(default_factory=lambda: list(DEFAULT_FORMATS))
    sample_rate: int = Field(default=22050, gt=0)
    segment_min_s: float = Field(default=1.0, gt=0)
    segment_max_s: float = Field(default=15.0, gt=0)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    diarize: DiarizeConfig = Field(default_factory=DiarizeConfig)


class DropRecord(BaseModel):
    """One filtered-out clip, with the failing filter and the measured value."""

    clip: str  # Tier-1: a segment-order ref ("<id>_segNNNN"); Tier-2: the final clip id
    filter: str  # the failing filter name — the drop reason
    value: str  # the measured value, formatted
    threshold: str  # the threshold it failed against, formatted
    text: str  # short transcript preview (for eyeballing)


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
    dropped_count: int = 0
    drops_by_reason: dict[str, int] = Field(default_factory=dict)
    drops: list[DropRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceResult(BaseModel):
    """The outcome of one source (playlist video / feed episode) in a combined build."""

    source_id: str
    label: str
    ok: bool
    clip_count: int = 0
    duration_s: float = 0.0
    dropped: int = 0
    error: str | None = None


class CombinedReport(BaseModel):
    """Summary of a combined build merging many sources into one dataset."""

    out_dir: str
    source: str  # the playlist/feed URL (or a label)
    title: str
    clip_count: int
    total_duration_s: float
    oversized_count: int
    dropped_count: int
    drops_by_reason: dict[str, int] = Field(default_factory=dict)
    sample_rate: int
    language: str
    formats: list[str]
    files: list[str] = Field(default_factory=list)
    sources: list[SourceResult] = Field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    warnings: list[str] = Field(default_factory=list)
