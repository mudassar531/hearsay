"""Tests for local whisper transcription.

The integration test runs the real `tiny` model with ``local_files_only`` so it
stays offline: it transcribes the committed sample clip if the model is cached,
and skips cleanly otherwise (e.g. fresh CI before a model-cache step).
"""

from pathlib import Path

import pytest

from hearsay.errors import TranscriptionError
from hearsay.transcribe import transcribe_audio

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"


def test_unknown_model_raises_friendly_error() -> None:
    with pytest.raises(TranscriptionError) as excinfo:
        transcribe_audio(SAMPLE, model_size="enormous")
    assert "enormous" in excinfo.value.message
    assert excinfo.value.hint


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
