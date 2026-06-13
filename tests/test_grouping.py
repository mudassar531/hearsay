"""Tests for the paragraph-grouping algorithm — hearsay's core feature.

The grouper is the entire reason the output reads well, so it is tested
hard: the lossless invariant and word-budget targets on the two real
fixtures, plus the edge cases an adversarial review surfaced (empty and
whitespace-only segments, overlapping/zero-duration timestamps, giant
segments, and degenerate parameters).
"""

import json
import statistics
from pathlib import Path

import pytest

from hearsay.captions import normalize_snippets
from hearsay.grouping import group_segments
from hearsay.models import Segment

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_segments(video_id: str) -> list[Segment]:
    data = json.loads((FIXTURES / f"{video_id}.transcript.json").read_text())
    return normalize_snippets(data["snippets"])


def seg(text: str, start_s: float, end_s: float) -> Segment:
    return Segment(text=text, start_s=start_s, end_s=end_s)


def words_of(segments: list[Segment]) -> list[str]:
    return " ".join(s.text for s in segments).split()


def words_of_paragraphs(paragraphs: list) -> list[str]:
    return " ".join(p.text for p in paragraphs).split()


# --- The two invariants that matter most, on real data --------------------


@pytest.mark.parametrize("video_id", ["rStL7niR7gs", "zjkBMFhNj_g"])
def test_lossless_on_real_fixtures(video_id: str) -> None:
    segments = fixture_segments(video_id)
    paragraphs = group_segments(segments)
    assert words_of_paragraphs(paragraphs) == words_of(segments)


@pytest.mark.parametrize("video_id", ["rStL7niR7gs", "zjkBMFhNj_g"])
def test_word_budget_on_real_fixtures(video_id: str) -> None:
    segments = fixture_segments(video_id)
    paragraphs = group_segments(segments)
    counts = [p.word_count for p in paragraphs]
    in_range = sum(1 for c in counts if 40 <= c <= 120)
    # Manual (punctuated) and auto (unpunctuated, overlapping) both land in
    # budget; the upper bound is a hard cap from the greedy split.
    assert max(counts) <= 120
    assert in_range / len(counts) >= 0.97
    assert statistics.median(counts) >= 40


@pytest.mark.parametrize("video_id", ["rStL7niR7gs", "zjkBMFhNj_g"])
def test_paragraph_timestamps_are_ordered_and_span_segments(video_id: str) -> None:
    segments = fixture_segments(video_id)
    paragraphs = group_segments(segments)
    starts = [p.start_s for p in paragraphs]
    assert starts == sorted(starts)
    for p in paragraphs:
        assert p.end_s >= p.start_s
    assert paragraphs[0].start_s == segments[0].start_s


def test_no_empty_paragraphs_on_real_fixtures() -> None:
    for video_id in ("rStL7niR7gs", "zjkBMFhNj_g"):
        paragraphs = group_segments(fixture_segments(video_id))
        assert all(p.text.strip() for p in paragraphs)


def test_deterministic() -> None:
    segments = fixture_segments("rStL7niR7gs")
    first = group_segments(segments)
    second = group_segments(segments)
    assert [(p.text, p.start_s, p.end_s) for p in first] == [
        (p.text, p.start_s, p.end_s) for p in second
    ]


# --- Edge cases -----------------------------------------------------------


def test_empty_input() -> None:
    assert group_segments([]) == []


def test_single_segment() -> None:
    paragraphs = group_segments([seg("just one short segment here", 0, 3)])
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "just one short segment here"
    assert paragraphs[0].start_s == 0
    assert paragraphs[0].end_s == 3


def test_giant_single_segment_is_not_split() -> None:
    big = " ".join(f"w{i}" for i in range(300))
    paragraphs = group_segments([seg(big, 0, 10)])
    assert len(paragraphs) == 1
    assert paragraphs[0].word_count == 300  # over max, but irreducible


def test_empty_and_whitespace_segments_do_not_crash_or_leak() -> None:
    # Regression: a whitespace/empty segment used to raise IndexError in the
    # break scorer and could inject a word_count==0 paragraph.
    segments = [
        seg("hello there friends this is the intro part one of the talk", 0, 3),
        seg("   ", 3, 4),
        seg("", 4, 5),
        seg("and now the second part of what we will discuss here today", 5, 8),
    ]
    paragraphs = group_segments(segments)
    assert words_of_paragraphs(paragraphs) == words_of(segments)
    assert all(p.text.strip() for p in paragraphs)
    assert "  " not in " ".join(p.text for p in paragraphs)  # no double spaces


def test_overlapping_timestamps_stay_lossless() -> None:
    # Auto-captions overlap: each 3s segment starts 1s after the previous.
    segments = [
        seg(f"word{i} and some more padding text", i * 1.0, i * 1.0 + 3.0) for i in range(60)
    ]
    paragraphs = group_segments(segments)
    assert words_of_paragraphs(paragraphs) == words_of(segments)


def test_zero_duration_segments_stay_lossless() -> None:
    segments = [seg(f"token{i} filler words here now", float(i), float(i)) for i in range(60)]
    paragraphs = group_segments(segments)
    assert words_of_paragraphs(paragraphs) == words_of(segments)


def test_long_pause_forces_a_break() -> None:
    segments = [
        seg("this is the first thought finishing up now", 0, 5),
        seg("and here begins a clearly separate later thought", 100, 105),
    ]
    paragraphs = group_segments(segments, min_words=3)
    assert len(paragraphs) == 2


def test_sentence_boundary_preferred_over_mid_phrase() -> None:
    # A sentence ends at segment 1; the budget window spans both candidates.
    # The break should land on the period, not after "the".
    segments = [
        seg(w, i * 1.0, i * 1.0 + 0.9)
        for i, w in enumerate(
            ["alpha beta gamma delta epsilon.", "the next sentence keeps going onward"]
        )
    ]
    paragraphs = group_segments(segments, min_words=3, max_words=8)
    assert paragraphs[0].text.endswith(".")


# --- Degenerate parameters (clamped, never crash, always lossless) --------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_words": 0, "max_words": 0},
        {"min_words": 40, "max_words": 5},  # max < min
        {"min_words": 999},  # min exceeds total words
        {"pause_threshold_s": 0},
        {"pause_threshold_s": -10},
    ],
)
def test_degenerate_params_are_lossless(kwargs: dict) -> None:
    segments = fixture_segments("rStL7niR7gs")
    paragraphs = group_segments(segments, **kwargs)
    assert words_of_paragraphs(paragraphs) == words_of(segments)
    assert all(p.text.strip() for p in paragraphs)


def test_pause_floor_prevents_shattering_on_small_gaps() -> None:
    # Realistic caption cadence: 5-word segments, 0.2s gaps. A zero threshold
    # must not shatter this into one-word paragraphs (it is floored to 0.5s).
    segments = [seg(f"a{i} b{i} c{i} d{i} e{i}", i * 2.2, i * 2.2 + 2.0) for i in range(60)]
    paragraphs = group_segments(segments, pause_threshold_s=0)
    assert len(paragraphs) < 10
    assert words_of_paragraphs(paragraphs) == words_of(segments)


def test_performance_on_ten_thousand_segments() -> None:
    segments = [
        seg(f"word{i} more text padding here ok", i * 2.0, i * 2.0 + 1.8) for i in range(10000)
    ]
    paragraphs = group_segments(segments)
    assert words_of_paragraphs(paragraphs) == words_of(segments)
