"""Tests for the dataset segmentation core (words -> training clips).

Covers the synthesized test matrix (boundary cases, span repair, tiny-tail
handling), the adversarial critic's regressions (a glitched interior word must
not launder the ``oversized`` flag), and a property/fuzz test that asserts every
hard invariant over hundreds of noisy random inputs.
"""

import math
import random

import pytest

from hearsay.dataset.segmentation import (
    _is_clause_end,
    _is_sentence_end,
    segment_words,
)
from hearsay.models import Word


def W(text: str, start: float, end: float, conf: float = 1.0) -> Word:
    return Word(text=text, start_s=start, end_s=end, confidence=conf)


def flatten(segs: list) -> list[Word]:
    return [w for s in segs for w in s.words]


def repaired(w: Word) -> tuple[float, float]:
    """The per-word span repair the segmenter applies (for assertions)."""
    s = w.start_s if math.isfinite(w.start_s) else 0.0
    s = max(s, 0.0)
    e = w.end_s if math.isfinite(w.end_s) else s
    e = max(e, s)
    return s, e


def seg_of(segs: list, word: Word):
    """The segment whose words contain ``word`` (by identity)."""
    for s in segs:
        if any(w is word for w in s.words):
            return s
    return None


# --- empties & singletons --------------------------------------------------


def test_empty_input() -> None:
    assert segment_words([]) == []


def test_all_whitespace_words() -> None:
    segs = segment_words([W(" ", 0, 1), W("\t", 1, 2), W("", 2, 3)])
    assert segs == []  # no empty segments are ever emitted


def test_single_word_oversized() -> None:
    segs = segment_words([W("x", 0.0, 20.0)])
    assert len(segs) == 1
    assert segs[0].oversized is True
    assert segs[0].duration_s == pytest.approx(20.0)
    assert [w.text for w in segs[0].words] == ["x"]


def test_single_word_under_min() -> None:
    segs = segment_words([W("x", 0.0, 0.4)])
    assert len(segs) == 1
    assert segs[0].oversized is False
    assert segs[0].duration_s == pytest.approx(0.4)


def test_whole_transcript_under_min() -> None:
    words = [W("a", 0.0, 0.2), W("b", 0.25, 0.45), W("c", 0.5, 0.7)]
    segs = segment_words(words)
    assert len(segs) == 1
    assert segs[0].oversized is False
    assert [w is x for w, x in zip(segs[0].words, words, strict=True)] == [True, True, True]


# --- losslessness ----------------------------------------------------------


def test_lossless_partition_simple() -> None:
    words = [
        W(f"w{i}" + ("," if i == 5 else ("." if i == 11 else "")), i * 0.6, i * 0.6 + 0.55)
        for i in range(12)
    ]
    segs = segment_words(words)
    flat = flatten(segs)
    assert [id(w) for w in flat] == [id(w) for w in words]  # exact, by identity, in order


def test_duration_bounds_normal() -> None:
    words = []
    t = 0.0
    for i in range(40):
        text = f"x{i}" + ("." if (i + 1) % 10 == 0 else "")
        words.append(W(text, t, t + 0.35))
        t += 0.4
    segs = segment_words(words)
    assert len(segs) >= 2
    for s in segs:
        assert s.oversized or s.duration_s <= 15.0 + 1e-9
    assert [id(w) for w in flatten(segs)] == [id(w) for w in words]


# --- hard breaks & boundary preference -------------------------------------


def test_hard_break_on_long_pause() -> None:
    words = [W(f"a{i}", i * 0.5, i * 0.5 + 0.4) for i in range(5)]  # ends 2.4
    words += [W(f"b{i}", 5.0 + i * 0.5, 5.0 + i * 0.5 + 0.4) for i in range(5)]  # 2.6s gap
    segs = segment_words(words)
    assert len(segs) >= 2
    assert seg_of(segs, words[4]) is not seg_of(segs, words[5])  # no segment spans the gap


def test_sentence_beats_clause() -> None:
    words = [W("alpha", 0, 1), W("beta,", 1, 2), W("gamma.", 2, 3), W("delta", 3, 4)]
    segs = segment_words(words, min_s=1.0, max_s=3.0)
    assert segs[0].text == "alpha beta, gamma."  # cut on the sentence, not the comma


def test_clause_beats_plain_pause() -> None:
    words = [W("aa", 0, 1), W("bb;", 1, 2), W("cc", 2, 3), W("dd", 3, 4)]
    segs = segment_words(words, min_s=1.0, max_s=3.0)
    assert segs[0].text == "aa bb;"  # clause beats a plain, equal-gap boundary


def test_long_pause_can_outrank_clause() -> None:
    words = [W("p", 0, 1), W("q,", 1, 2), W("r", 2, 3), W("s", 4.0, 5.0)]  # 1.0s gap after r
    segs = segment_words(words, min_s=1.0, max_s=3.0)
    assert segs[0].text == "p q, r"  # a long-enough pause outranks the comma (pause-primacy)


# --- detectors -------------------------------------------------------------


def test_sentence_end_detector() -> None:
    assert _is_sentence_end("end.") is True
    assert _is_sentence_end("really?") is True
    assert _is_sentence_end("stop!") is True
    assert _is_sentence_end("middle") is False


def test_abbreviation_not_sentence_end() -> None:
    for token in ("Dr.", "etc.", "e.g.", "U.S.", "p.m."):
        assert _is_sentence_end(token) is False, token


def test_decimal_and_middle_initial_guards() -> None:
    assert _is_sentence_end("J.") is False  # a name initial
    assert _is_sentence_end("3.14.") is False  # decimal, not a full stop
    assert _is_sentence_end("apples.") is True  # a real sentence end with a digit-free token


def test_ellipsis_is_clause_not_sentence() -> None:
    assert _is_sentence_end("wait...") is False
    assert _is_clause_end("wait...") is True
    assert _is_sentence_end("done.") is True


# --- span repair / noisy timings -------------------------------------------


def test_inverted_span_repair() -> None:
    words = [W("a", 0, 1), W("b", 1.5, 1.2), W("c", 1.6, 2.0)]  # b inverted
    segs = segment_words(words)
    assert all(s.duration_s >= 0 for s in segs)
    assert all(s.end_s >= s.start_s >= 0 for s in segs)
    assert seg_of(segs, words[1]) is not None  # b is present (lossless)


def test_zero_length_span() -> None:
    words = [W("a", 0, 1), W("b", 1, 1), W("c", 1.1, 2)]  # b zero-length
    segs = segment_words(words)
    assert [id(w) for w in flatten(segs)] == [id(w) for w in words]
    assert all(s.duration_s >= 0 for s in segs)


def test_overlap_no_phantom_pause_and_faithful_start() -> None:
    # w2 starts at 1.9, overlapping w1 (ends 2.0): the gap must clamp to 0 (no
    # spurious break) and w2's segment start must stay 1.9 (NOT bumped to 2.0).
    words = [W("aa", 0, 1), W("bb.", 1, 2), W("cc", 1.9, 2.9), W("dd", 2.9, 3.9)]
    segs = segment_words(words, min_s=1.0, max_s=2.0)
    seg_cc = seg_of(segs, words[2])
    assert seg_cc.start_s == pytest.approx(1.9)


def test_out_of_order_timings() -> None:
    words = [W("a", 2.0, 2.5), W("b", 1.0, 1.5), W("c", 2.6, 3.0)]  # b before a
    segs = segment_words(words)
    assert [id(w) for w in flatten(segs)] == [id(w) for w in words]  # order preserved
    assert all(s.end_s >= s.start_s >= 0 for s in segs)


def test_nan_inf_spans() -> None:
    words = [W("a", float("nan"), 1.0), W("b", 2.0, float("inf")), W("c", 3.0, 3.5)]
    segs = segment_words(words)  # must not raise
    for s in segs:
        assert math.isfinite(s.start_s) and math.isfinite(s.end_s)
        assert s.end_s >= s.start_s >= 0


def test_unbreakable_dense_run() -> None:
    words = []
    t = 0.0
    for i in range(30):  # no punctuation, ~0.7s words, tiny gaps, ~21s total
        words.append(W(f"w{i}", t, t + 0.65))
        t += 0.7
    segs = segment_words(words)
    assert len(segs) >= 2
    for s in segs:
        assert s.oversized or s.duration_s <= 15.0 + 1e-9
    assert [id(w) for w in flatten(segs)] == [id(w) for w in words]


# --- tiny-tail handling ----------------------------------------------------


def test_tiny_tail_isolated_kept() -> None:
    # Small in-run gaps (median ~0.5s) then a 6.0s pause to a tiny outro -> isolated.
    words = [W("alpha", 0, 1), W("beta", 1.5, 2.5), W("gamma", 3, 4), W("outro", 10.0, 10.4)]
    segs = segment_words(words)
    assert len(segs) == 2
    assert segs[-1].text == "outro"


def test_tiny_tail_merged() -> None:
    # Median gap ~1.5s, a 2.5s gap to the tiny tail (<= 2x scale) -> not isolated, merges.
    words = [W("alpha", 0, 1), W("beta", 2.5, 3.5), W("gamma", 5, 6), W("delta", 8.5, 8.9)]
    segs = segment_words(words)
    assert len(segs) == 1
    assert "delta" in segs[0].text


def test_tiny_tail_merge_would_overshoot_rebalances() -> None:
    # Merging the tiny tail with its predecessor would exceed max_s -> re-split in two.
    words = [W("w0", 0.0, 1.5), W("w1", 1.5, 3.0), W("w2", 3.0, 3.4)]
    segs = segment_words(words, min_s=2.0, max_s=3.0)
    assert len(segs) == 2
    assert all(not s.oversized for s in segs)
    assert all(s.duration_s <= 3.0 + 1e-9 for s in segs)


# --- exact bounds & params -------------------------------------------------


def test_exactly_max_s_not_oversized() -> None:
    segs = segment_words([W("x", 0.0, 5.0)], min_s=1.0, max_s=5.0)
    assert len(segs) == 1
    assert segs[0].duration_s == pytest.approx(5.0)
    assert segs[0].oversized is False  # exactly max_s is in bounds


def test_degenerate_params_clamped() -> None:
    words = [W(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(5)]
    segs = segment_words(words, min_s=-5, max_s=-1, pause_break_s=0.0)  # must not raise
    assert segs  # produced something
    assert [id(w) for w in flatten(segs)] == [id(w) for w in words]


def test_determinism() -> None:
    words = [W(f"w{i}" + ("." if i % 7 == 0 else ""), i * 0.8, i * 0.8 + 0.7) for i in range(25)]
    assert segment_words(words) == segment_words(words)


def test_text_normalization_and_blank_inclusion() -> None:
    words = [W("  hello ", 0, 1), W("   ", 1, 1.2), W(" world ", 1.2, 2)]
    segs = segment_words(words)
    assert len(segs) == 1
    assert segs[0].text == "hello world"  # stripped, single-spaced, blank omitted
    assert len(segs[0].words) == 3  # ...but the blank Word is still present (lossless)


def test_words_not_mutated() -> None:
    words = [W("a", 5.0, 4.0), W("b", 1.9, 1.0), W("c", 6.0, 7.0)]  # inverted/overlap
    snapshot = [(w.text, w.start_s, w.end_s, w.confidence) for w in words]
    segs = segment_words(words)
    after = [(w.text, w.start_s, w.end_s, w.confidence) for w in words]
    assert after == snapshot  # originals untouched
    for w in words:
        assert seg_of(segs, w) is not None  # repair is visible only on the segment span


def test_blank_only_run_dropped_neighbors_lossless() -> None:
    words = [W("hi", 0, 1), W("  ", 4.0, 4.5), W("bye", 8.0, 9.0)]  # blank isolated by pauses
    segs = segment_words(words)
    assert [s.text for s in segs] == ["hi", "bye"]  # all-blank run dropped, no empty seg
    assert all(s.text.strip() for s in segs)


# --- adversarial critic regressions ----------------------------------------


def test_oversized_flag_survives_tiny_tail_merge() -> None:
    # faster-whisper can emit a glitched far-future end on an interior word. A
    # naive endpoint-only duration would let a Phase-3 merge launder the oversized
    # flag; the extent-based duration must keep it honest.
    words = [W("setup here ok", 0.0, 2.0), W("GLITCH", 2.1, 400.0), W("x", 2.2, 2.4)]
    segs = segment_words(words)
    glitch_seg = seg_of(segs, words[1])
    assert glitch_seg.oversized is True
    assert glitch_seg.end_s >= 400.0  # the reported span actually contains the word


def test_interior_glitch_end_not_masked() -> None:
    words = [
        W("a", 0.0, 0.5),
        W("b", 0.6, 1.0),
        W("GLITCH", 1.1, 500.0),
        W("d", 1.6, 2.0),
        W("e", 2.1, 2.5),
    ]
    segs = segment_words(words)
    # No emitted segment may both look in-bounds AND contain a word ending past its end_s.
    for s in segs:
        max_word_end = max(repaired(w)[1] for w in s.words)
        assert s.end_s + 1e-9 >= max_word_end
        if not s.oversized:
            assert s.duration_s <= 15.0 + 1e-9


def test_pause_break_floor_still_lossless() -> None:
    # A near-zero pause_break_s floors to 0.5 and shatters dense speech into
    # one-word clips — quality collapses, but losslessness must hold.
    words = [W(f"w{i}", i * 0.6, i * 0.6 + 0.4) for i in range(8)]  # ~0.2s gaps
    segs = segment_words(words, pause_break_s=0.0)
    assert [id(w) for w in flatten(segs)] == [id(w) for w in words]


# --- property / fuzz: every hard invariant over many noisy inputs ----------


def _random_words(rng: random.Random) -> list[Word]:
    n = rng.randint(0, 25)
    words: list[Word] = []
    t = rng.uniform(0, 2)
    vocab = ["the", "cat", "sat,", "down.", "and", "then", "Dr.", "3.14", "wait...", "ok!", " ", ""]
    for _ in range(n):
        text = rng.choice(vocab)
        dur = rng.uniform(0.0, 2.0)
        start = t + rng.uniform(-0.3, 1.0)  # occasionally overlaps / goes backwards
        end = start + dur
        roll = rng.random()
        if roll < 0.04:
            start = float("nan")
        elif roll < 0.08:
            end = float("inf")
        elif roll < 0.12:
            end = start - rng.uniform(0, 1)  # inverted
        words.append(W(text, start, end, rng.uniform(0, 1)))
        t = (end if math.isfinite(end) else t) + rng.uniform(0, 1.5)
    return words


def test_property_invariants_over_random_inputs() -> None:
    rng = random.Random(20260614)
    for _ in range(400):
        words = _random_words(rng)
        min_s = rng.choice([0.5, 1.0, 2.0])
        max_s = rng.choice([1.0, 5.0, 15.0])
        pause_break_s = rng.choice([0.5, 1.0, 2.0, 5.0])
        segs = segment_words(words, min_s=min_s, max_s=max_s, pause_break_s=pause_break_s)

        # effective clamps (mirror segment_words)
        eff_min = min_s if min_s > 0 else 1.0
        eff_max = max(max_s, eff_min)

        flat = flatten(segs)
        flat_ids = [id(w) for w in flat]

        # 1. order preserved, no duplication
        pos = {id(w): i for i, w in enumerate(words)}
        positions = [pos[i] for i in flat_ids]
        assert positions == sorted(positions)
        assert len(set(flat_ids)) == len(flat_ids)

        # 2. every NON-blank word present; anything dropped is blank
        present = set(flat_ids)
        for w in words:
            if w.text.strip():
                assert id(w) in present
        for w in words:
            if id(w) not in present:
                assert w.text.strip() == ""

        for s in segs:
            # 3. no empty segments
            assert s.text.strip() != ""
            # 4. spans non-negative and monotone
            assert s.start_s >= 0
            assert s.end_s >= s.start_s
            # 5. span is the true extent of its words (contains them; honest)
            starts = [repaired(w)[0] for w in s.words]
            ends = [repaired(w)[1] for w in s.words]
            assert s.start_s == pytest.approx(min(starts))
            assert s.end_s == pytest.approx(max(max(ends), min(starts)))
            # 6. oversized iff duration genuinely exceeds max_s
            assert s.oversized == (s.duration_s > eff_max + 1e-12)

        # 7. determinism
        assert segment_words(
            words, min_s=min_s, max_s=max_s, pause_break_s=pause_break_s
        ) == segs
