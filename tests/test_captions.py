"""Tests for caption selection and normalization (offline, fixture-driven)."""

import json
from pathlib import Path

import pytest

from hearsay import captions
from hearsay.captions import TranscriptInfo, normalize_snippets, select_transcript

FIXTURES = Path(__file__).parent / "fixtures"


def available_from_fixture(video_id: str) -> list[TranscriptInfo]:
    data = json.loads((FIXTURES / f"{video_id}.transcript.json").read_text())
    return [
        TranscriptInfo(t["language_code"], t["is_generated"]) for t in data["available_transcripts"]
    ]


def test_selects_manual_en_when_available() -> None:
    available = available_from_fixture("rStL7niR7gs")  # CGP Grey: manual en exists
    choice = select_transcript(available, "en")
    assert choice is not None
    assert choice.language_code == "en"
    assert choice.is_generated is False


def test_falls_back_to_generated_when_no_manual_in_language() -> None:
    available = available_from_fixture("zjkBMFhNj_g")  # Karpathy: auto en only
    choice = select_transcript(available, "en")
    assert choice is not None
    assert choice.language_code == "en"
    assert choice.is_generated is True


def test_prefers_manual_over_generated_same_language() -> None:
    available = [TranscriptInfo("en", True), TranscriptInfo("en", False)]
    choice = select_transcript(available, "en")
    assert choice == TranscriptInfo("en", False)


def test_matches_regional_and_named_variants() -> None:
    # Real case: Stanford's Steve Jobs address has a manual track "en-eEY6OEpapPo".
    available = [TranscriptInfo("ja", False), TranscriptInfo("en-eEY6OEpapPo", False)]
    choice = select_transcript(available, "en")
    assert choice == TranscriptInfo("en-eEY6OEpapPo", False)


def test_falls_back_to_first_available_language() -> None:
    available = [TranscriptInfo("de", True), TranscriptInfo("fr", False)]
    choice = select_transcript(available, "en")
    assert choice == TranscriptInfo("fr", False)  # any manual beats any generated


def test_returns_none_when_nothing_available() -> None:
    assert select_transcript([], "en") is None


def test_normalize_unescapes_and_collapses_whitespace() -> None:
    segments = normalize_snippets(
        [{"text": "it&#39;s a\ntest   of &amp; entities", "start": 1.0, "duration": 2.5}]
    )
    assert len(segments) == 1
    assert segments[0].text == "it's a test of & entities"
    assert segments[0].start_s == 1.0
    assert segments[0].end_s == 3.5


def test_normalize_drops_noise_cues_and_empties() -> None:
    snippets = [
        {"text": "[Music]", "start": 0.0, "duration": 1.0},
        {"text": "(applause)", "start": 1.0, "duration": 1.0},
        {"text": "  ", "start": 2.0, "duration": 1.0},
        {"text": "[Music] but real words too", "start": 3.0, "duration": 1.0},
        {"text": "plain text", "start": 4.0, "duration": 1.0},
    ]
    segments = normalize_snippets(snippets)
    assert [s.text for s in segments] == ["[Music] but real words too", "plain text"]


def test_normalize_preserves_real_parentheticals() -> None:
    # Short fully-parenthesized speech survives; only cues containing a sound
    # keyword (music/applause/laughter/...) are dropped — even descriptive ones
    # like "[ominous music plays]" (real case from the CGP fixture).
    snippets = [
        {"text": "(I think so)", "start": 0.0, "duration": 1.0},
        {"text": "[for all your hard work]", "start": 1.0, "duration": 1.0},
        {"text": "(that sounds good)", "start": 2.0, "duration": 1.0},
        {"text": "[ominous music plays]", "start": 3.0, "duration": 1.0},  # noise
        {"text": "[Laughter]", "start": 4.0, "duration": 1.0},  # noise
    ]
    segments = normalize_snippets(snippets)
    assert [s.text for s in segments] == [
        "(I think so)",
        "[for all your hard work]",
        "(that sounds good)",
    ]


def test_normalize_real_fixtures_yields_clean_ordered_segments() -> None:
    for video_id in ("rStL7niR7gs", "zjkBMFhNj_g"):
        data = json.loads((FIXTURES / f"{video_id}.transcript.json").read_text())
        segments = normalize_snippets(data["snippets"])
        assert len(segments) > 100
        starts = [s.start_s for s in segments]
        assert starts == sorted(starts)
        for segment in segments:
            assert segment.text == segment.text.strip()
            assert "\n" not in segment.text
            assert segment.end_s >= segment.start_s


def test_captions_session_applies_a_default_timeout() -> None:
    """requests has no default timeout, so a stalled YouTube connection would hang
    hearsay indefinitely on its fastest, most common path."""
    session = captions._TimeoutSession()
    seen: dict = {}

    class _Adapter:
        def send(self, request: object, **kwargs: object) -> object:
            seen.update(kwargs)
            raise RuntimeError("stop before the network")

    session.adapters.clear()
    session.mount("http://", _Adapter())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        session.get("http://example.invalid")
    assert seen.get("timeout") == captions._CAPTIONS_TIMEOUT_S
