"""Tests for local whisper transcription.

The integration test runs the real `tiny` model with ``local_files_only`` so it
stays offline: it transcribes the committed sample clip if the model is cached,
and skips cleanly otherwise (e.g. fresh CI before a model-cache step).
"""

import os
from pathlib import Path

import pytest

from hearsay.errors import TranscriptionError
from hearsay.models import Segment, Word
from hearsay.transcribe import TranscriptionResult, _ensure_download_timeout, transcribe_audio

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"


def test_unknown_model_raises_friendly_error() -> None:
    with pytest.raises(TranscriptionError) as excinfo:
        transcribe_audio(SAMPLE, model_size="enormous")
    assert "enormous" in excinfo.value.message
    assert excinfo.value.hint


def test_ensure_download_timeout_raises_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    # With nothing set, the floor is raised above huggingface_hub's 10s default
    # (which caused real "read operation timed out" failures on multi-GB weights).
    from huggingface_hub import constants

    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    default = getattr(constants, "DEFAULT_DOWNLOAD_TIMEOUT", 10)
    monkeypatch.setattr(constants, "HF_HUB_DOWNLOAD_TIMEOUT", default)

    _ensure_download_timeout()

    assert int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]) >= 60
    assert constants.HF_HUB_DOWNLOAD_TIMEOUT >= 60


def test_ensure_download_timeout_preserves_user_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # A user-provided, higher value must never be lowered.
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    monkeypatch.setattr(constants, "HF_HUB_DOWNLOAD_TIMEOUT", 300)

    _ensure_download_timeout()

    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"
    assert constants.HF_HUB_DOWNLOAD_TIMEOUT == 300


def test_transcribe_sample_with_tiny() -> None:
    try:
        result = transcribe_audio(SAMPLE, model_size="tiny", local_files_only=True)
    except TranscriptionError:
        pytest.skip("tiny whisper model not cached; run hearsay once online to fetch it")

    text = " ".join(s.text for s in result.segments).lower()
    # The clip says "...the quick brown fox jumps over the lazy dog...".
    assert "fox" in text
    assert "dog" in text
    assert result.model_size == "tiny"
    assert result.method == "whisper-tiny"
    assert result.language == "en"
    assert result.duration_s == pytest.approx(4.7, abs=1.0)
    assert all(s.end_s >= s.start_s for s in result.segments)


def test_transcribe_sample_with_parakeet() -> None:
    pytest.importorskip("parakeet_mlx", reason="parakeet-mlx not installed (non-Apple-Silicon)")
    try:
        result = transcribe_audio(SAMPLE, model_size="parakeet", local_files_only=True)
    except TranscriptionError:
        pytest.skip("parakeet model not cached; run hearsay once online to fetch it")

    text = " ".join(s.text for s in result.segments).lower()
    # The clip says "...the quick brown fox jumps over the lazy dog...".
    assert "fox" in text
    assert "dog" in text
    assert result.method.startswith("parakeet")
    assert all(s.end_s >= s.start_s for s in result.segments)


def test_transcribe_reports_progress() -> None:
    calls: list[tuple[float, float]] = []
    try:
        transcribe_audio(
            SAMPLE,
            model_size="tiny",
            local_files_only=True,
            on_progress=lambda done, total: calls.append((done, total)),
        )
    except TranscriptionError:
        pytest.skip("tiny whisper model not cached")
    assert calls, "progress callback was never invoked"
    done, total = calls[-1]
    assert total > 0
    assert done == pytest.approx(total, abs=0.01)  # finishes at 100%


def test_transcription_result_words_default_none() -> None:
    # The markdown/JSON path never asks for words, so the field defaults to None
    # and is purely additive (does not affect the JSON schema).
    result = TranscriptionResult(
        segments=[Segment(text="hi", start_s=0.0, end_s=1.0)],
        language="en",
        duration_s=1.0,
        model_size="tiny",
        method="whisper-tiny",
    )
    assert result.words is None


def test_word_timestamps_off_by_default() -> None:
    try:
        result = transcribe_audio(SAMPLE, model_size="tiny", local_files_only=True)
    except TranscriptionError:
        pytest.skip("tiny whisper model not cached")
    assert result.words is None  # default: no word-level timings captured


def test_word_timestamps_captured_when_requested() -> None:
    try:
        result = transcribe_audio(
            SAMPLE, model_size="tiny", local_files_only=True, word_timestamps=True
        )
    except TranscriptionError:
        pytest.skip("tiny whisper model not cached")
    assert result.words, "expected word-level timings"
    assert all(isinstance(w, Word) for w in result.words)
    text = " ".join(w.text for w in result.words).lower()
    assert "fox" in text and "dog" in text  # the clip says "...brown fox...lazy dog..."
    assert all(w.end_s >= w.start_s for w in result.words)
    assert all(0.0 <= w.confidence <= 1.0 for w in result.words)
