"""Group transcript segments into readable paragraphs — the core of hearsay.

Pause-primary strategy: the silences between segments are the strongest
signal of where one thought ends and the next begins.

1.  Inter-segment gaps are measured, then normalized against the *typical*
    gap of the transcript. YouTube auto-captions are rolling windows that
    systematically overlap (every raw gap is negative); subtracting the
    median gap recovers the real pauses hiding inside that overlap, while
    leaving well-behaved manual captions untouched.
2.  Pauses longer than ``pause_threshold_s`` (after normalization) are hard
    paragraph breaks — a speaker who stops that long has changed topic.
    A deliberate consequence of pause-primacy: a long mid-sentence emphasis
    pause can end a paragraph on a comma. That is intended, not a bug.
3.  Inside each pause-delimited run, paragraphs grow toward a word budget.
    When the budget window (``min_words``..``max_words``) is reached, the
    break lands on the best-scoring candidate. The score weighs pause length
    (dominant) and refines with both sides of the boundary: sentence/clause/
    ellipsis punctuation and a dangling-word penalty on the segment that ends
    the paragraph, plus a discourse-marker bonus / continuation penalty on the
    segment that would start the next one. So punctuated transcripts break at
    sentence ends and unpunctuated auto-captions still break at the most
    natural breath in the window.

The module is pure and deterministic: no I/O, no randomness, stdlib only.
"""

from __future__ import annotations

import re

from hearsay.models import Paragraph, Segment

# --- Boundary punctuation -------------------------------------------------

# An ellipsis ending: a soft trailing-off, scored between clause and sentence.
_ELLIPSIS_END = re.compile(r"(?:\.\.\.|…)[\"“”‘’')\]]*$")
# A sentence-final mark, optionally followed by closing quotes/brackets.
_SENTENCE_END = re.compile(r"[.!?][\"“”‘’')\]]*$")
# A clause-final mark (weaker, but still better than mid-phrase).
_CLAUSE_END = re.compile(r"[,;:—–][\"“”‘’')\]]*$")
# A trailing period on these tokens is an abbreviation, not a full stop.
_ABBREVIATIONS = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e.",
        "jr.", "sr.", "prof.", "no.", "inc.", "co.", "fig.", "al.",
    }
)  # fmt: skip
# A dotted initialism like "u.s." or "p.m." — also not a sentence end.
_DOTTED_INITIAL = re.compile(r"^(?:[a-z]\.)+$")

# --- Word cues ------------------------------------------------------------

# Words that should never end a paragraph: articles, conjunctions, common
# prepositions, auxiliaries, and speech fillers. Auto-captions often pause
# right after these ("...but what if I add" <long pause> "this") and a break
# there reads terribly.
_DANGLING_WORDS = frozenset(
    """
    a an the and or but so because if that which who whose whom of to in on
    at by for with from into onto as is are was were be been being am do
    does did have has had will would can could shall should may might must
    very really just quite their his her its our your my this these those
    i you we they he she it there when while than then though although
    unless until example um uh like
    """.split()  # noqa: SIM905 - a wrapped word list reads better than a 60-item literal
)
# Auxiliary contractions ("don't", "it's", "we're", "I'll") never end a thought.
_CONTRACTION = re.compile(r"(?:n['’]t|['’](?:s|re|m|ve|ll|d))$")

# Words that open a fresh thought when they START the next segment — a strong
# topic-shift cue in unpunctuated speech where the previous segment has no
# period to break on.
_STRONG_MARKERS = frozenset(
    {
        "so", "now", "okay", "ok", "but", "anyway", "anyways", "alright",
        "next", "however", "finally", "meanwhile", "first", "firstly",
        "second", "secondly", "third", "lastly", "basically",
    }
)  # fmt: skip
_WEAK_MARKERS = frozenset({"and", "then", "well", "also"})
# Hesitation fillers skipped before looking for a discourse marker.
_FILLERS = frozenset({"um", "uh", "uhm", "umm", "er", "ah", "mm", "hmm"})

# --- Scoring weights (seconds of pause are the dominant currency) ---------

_GAP_WEIGHT = 3.0  # one full second of pause = 3 points
_SENTENCE_BONUS = 2.0  # a sentence end outweighs ~0.7 s of pause
_ELLIPSIS_BONUS = 1.0
_CLAUSE_BONUS = 0.6
_DANGLING_PENALTY = 2.5  # never end on "the" if any alternative exists
_MARKER_BONUS = 0.8  # next segment opens with "so"/"now"/"but"...
_WEAK_MARKER_BONUS = 0.4  # ...or "and"/"then"/"well"
_CONTINUATION_PENALTY = 1.0  # next segment opens mid-phrase ("...the"|"of...")
_BALANCE_WEIGHT = 0.8  # gentle pull toward the target paragraph size
_TARGET_FRACTION = 0.45  # target sits just below the budget midpoint
_VERY_LONG_PAUSE_FACTOR = 2.0  # tiny tail survives only past 2x threshold
# A pause threshold below this is treated as this; speech pauses shorter than
# half a second are breaths, not paragraph boundaries. Guards against a tiny
# or zero threshold shattering the transcript into one-segment paragraphs.
_MIN_PAUSE_THRESHOLD_S = 0.5

_STRIP_CHARS = "\"“”‘’'()[]"


def group_segments(
    segments: list[Segment],
    *,
    pause_threshold_s: float = 2.0,
    min_words: int = 40,
    max_words: int = 120,
) -> list[Paragraph]:
    """Group timed transcript segments into readable paragraphs.

    Splits primarily on long inter-segment pauses (normalized for the
    systematic overlap of rolling auto-captions), then sizes paragraphs to
    the ``min_words``..``max_words`` budget, refining break placement with
    punctuation and discourse cues on both sides of each candidate boundary.

    Args:
        segments: Clean, ordered transcript segments.
        pause_threshold_s: Normalized silence (seconds) that forces a
            paragraph break regardless of word counts. Floored to
            ``0.5`` so a tiny/zero value cannot shatter the transcript.
        min_words: Soft minimum paragraph size; tiny tails are merged back
            unless isolated by a very long pause. Floored to ``1``.
        max_words: Soft maximum paragraph size; only exceeded when a span
            contains no breakable boundary (e.g. one huge segment). Raised
            to ``min_words`` if a caller passes something smaller.

    Returns:
        Paragraphs whose concatenated words exactly equal the input's
        (lossless), each spanning its first segment's ``start_s`` to its
        last segment's ``end_s``. Empty input yields ``[]``; empty or
        whitespace-only segments never produce empty paragraphs.
    """
    if not segments:
        return []

    # Clamp degenerate parameters so the rest of the pipeline is well-defined.
    min_words = max(1, min_words)
    max_words = max(min_words, max_words)
    pause_threshold_s = max(pause_threshold_s, _MIN_PAUSE_THRESHOLD_S)

    eff_gaps = _effective_gaps(segments)

    # Phase 1: hard breaks at long pauses -> runs of segments.
    runs: list[tuple[int, int]] = []  # [start, end) index spans
    run_start = 0
    for i, gap in enumerate(eff_gaps):
        if gap > pause_threshold_s:
            runs.append((run_start, i + 1))
            run_start = i + 1
    runs.append((run_start, len(segments)))

    # Phase 2: split each run on the word budget, breaks scored pause-first.
    groups: list[list[int]] = []
    for lo, hi in runs:
        groups.extend(
            _split_run(segments, eff_gaps, lo, hi, min_words=min_words, max_words=max_words)
        )

    # Phase 3: merge a tiny trailing paragraph back, unless a very long
    # pause genuinely isolates it (e.g. an outro after a long silence).
    groups = _fix_tiny_tail(
        segments,
        eff_gaps,
        groups,
        pause_threshold_s=pause_threshold_s,
        min_words=min_words,
        max_words=max_words,
    )

    paragraphs: list[Paragraph] = []
    for group in groups:
        if not group:
            continue
        paragraph = _materialize(segments, group)
        if paragraph.text:  # drop groups of only empty/whitespace segments
            paragraphs.append(paragraph)
    return paragraphs


def _effective_gaps(segments: list[Segment]) -> list[float]:
    """Pause length before each next segment, normalized and clamped >= 0.

    Raw gap is ``next.start_s - prev.end_s``. Rolling auto-captions overlap
    systematically, making every raw gap negative; subtracting the median
    gap (when negative) exposes the true pauses. Manual captions with a
    non-negative median are left untouched.
    """
    raw = [segments[i + 1].start_s - segments[i].end_s for i in range(len(segments) - 1)]
    if not raw:
        return []
    baseline = min(0.0, _median(raw))
    return [max(0.0, gap - baseline) for gap in raw]


def _median(values: list[float]) -> float:
    """Median of a non-empty list (kept local for a self-contained module)."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _split_run(
    segments: list[Segment],
    eff_gaps: list[float],
    lo: int,
    hi: int,
    *,
    min_words: int,
    max_words: int,
) -> list[list[int]]:
    """Split segments[lo:hi] (no internal hard pauses) into sized groups.

    Greedy: grow a paragraph until the word budget window is filled, then
    break at the best-scoring candidate whose left side lands within
    ``min_words``..``max_words``. A short leftover tail is merged or the
    final two groups are rebalanced.
    """
    words = [len(segments[i].text.split()) for i in range(lo, hi)]
    n = hi - lo
    groups: list[list[int]] = []
    i = 0
    while i < n:
        cum = 0
        candidates: list[tuple[int, int]] = []  # (local break index, left words)
        overshoot = None  # first index where cum reaches max_words
        j = i
        while j < n:
            cum += words[j]
            if min_words <= cum <= max_words:
                candidates.append((j, cum))
            if cum >= max_words:
                overshoot = j
                break
            j += 1

        if overshoot is None:
            # The rest of the run fits under the budget: take it all.
            groups.append(list(range(lo + i, hi)))
            break

        if candidates:
            chosen = _best_candidate(
                segments, eff_gaps, lo, candidates, min_words=min_words, max_words=max_words
            )
        else:
            # No segment boundary inside the budget window (huge segment):
            # break right after the segment that overshot.
            chosen = overshoot
        groups.append(list(range(lo + i, lo + chosen + 1)))
        i = chosen + 1

    return _fix_run_tail(segments, eff_gaps, groups, min_words=min_words, max_words=max_words)


def _best_candidate(
    segments: list[Segment],
    eff_gaps: list[float],
    lo: int,
    candidates: list[tuple[int, int]],
    *,
    min_words: int,
    max_words: int,
) -> int:
    """Pick the best break among (local_index, left_words) candidates.

    Score = pause length (dominant) + two-sided punctuation/discourse
    refinement + a gentle pull toward the target size. Later candidates win
    ties, biasing toward fuller paragraphs.
    """
    target = min_words + _TARGET_FRACTION * (max_words - min_words)
    best_idx = candidates[0][0]
    best_score = float("-inf")
    for j, left in candidates:
        score = _break_score(segments, eff_gaps, lo + j)
        score -= _BALANCE_WEIGHT * abs(left - target) / target
        if score >= best_score:
            best_score = score
            best_idx = j
    return best_idx


def _break_score(segments: list[Segment], eff_gaps: list[float], gi: int) -> float:
    """How good is a paragraph break right after global segment ``gi``?

    Combines the pause after ``gi`` with cues from both the segment that ends
    the paragraph (``gi``) and the one that would start the next (``gi + 1``).
    """
    score = _GAP_WEIGHT * (eff_gaps[gi] if gi < len(eff_gaps) else 0.0)
    text = segments[gi].text
    ended_sentence = False
    if _ELLIPSIS_END.search(text):
        score += _ELLIPSIS_BONUS
    elif _SENTENCE_END.search(text):
        if _ends_with_abbreviation(text):
            score += _CLAUSE_BONUS  # "Mr." / "etc." is not a full stop
        else:
            score += _SENTENCE_BONUS
            ended_sentence = True
    elif _CLAUSE_END.search(text):
        score += _CLAUSE_BONUS
    else:
        # Unpunctuated ending: punish breaks after words that cannot close a
        # thought ("...but what if I add" | "this" reads terribly).
        words = text.split()
        if words:
            last_word = words[-1].strip(_STRIP_CHARS).lower()
            if last_word in _DANGLING_WORDS or _CONTRACTION.search(last_word):
                score -= _DANGLING_PENALTY
            elif any(ch.isdigit() for ch in last_word):
                # Bare numbers usually sit mid-quantity ("...um 13" | "billion").
                score -= _DANGLING_PENALTY / 2.0

    # Look at the segment that would open the next paragraph.
    if gi + 1 < len(segments):
        score += _begin_score(segments[gi + 1].text, after_sentence_end=ended_sentence)
    return score


def _ends_with_abbreviation(text: str) -> bool:
    """True if the final token is an abbreviation or dotted initialism."""
    words = text.split()
    if not words:
        return False
    # Keep the trailing period (it is the abbreviation marker); strip only
    # surrounding quotes/brackets so "(Mr." and "etc.)" still match.
    last = words[-1].lstrip("\"“”‘’'([").rstrip("\"“”‘’')]").lower()
    return last in _ABBREVIATIONS or bool(_DOTTED_INITIAL.fullmatch(last))


def _begin_score(next_text: str, *, after_sentence_end: bool) -> float:
    """Score the opening of the next segment as a break cue.

    Discourse markers ("so", "now", "but", ...) reward a break; a bare
    function word ("the", "of", "to") opening mid-phrase penalizes one,
    unless a sentence just ended (starting a paragraph with "The" is fine).
    Leading hesitation fillers ("um", "uh") are skipped first.
    """
    words = next_text.lower().split()
    index = 0
    while index < len(words) and words[index].strip(_STRIP_CHARS) in _FILLERS:
        index += 1
    if index >= len(words):
        return 0.0
    head = words[index].strip(_STRIP_CHARS)
    if head in _STRONG_MARKERS:
        return _MARKER_BONUS
    if head in _WEAK_MARKERS:
        return _WEAK_MARKER_BONUS
    if head in _DANGLING_WORDS and not after_sentence_end:
        return -_CONTINUATION_PENALTY
    return 0.0


def _fix_run_tail(
    segments: list[Segment],
    eff_gaps: list[float],
    groups: list[list[int]],
    *,
    min_words: int,
    max_words: int,
) -> list[list[int]]:
    """Absorb an undersized final group of a run into its neighbour.

    Merges outright when the result stays within budget; otherwise the last
    two groups are re-split at the best near-balanced break point.
    """
    if len(groups) < 2:
        return groups
    tail = groups[-1]
    if _group_words(segments, tail) >= min_words:
        return groups
    prev = groups[-2]
    combined = prev + tail
    if _group_words(segments, combined) <= max_words:
        return [*groups[:-2], combined]
    return [
        *groups[:-2],
        *_rebalance(segments, eff_gaps, combined, min_words=min_words, max_words=max_words),
    ]


def _rebalance(
    segments: list[Segment],
    eff_gaps: list[float],
    indices: list[int],
    *,
    min_words: int,
    max_words: int,
) -> list[list[int]]:
    """Split one over-budget group into two well-sized, well-placed halves.

    Prefers break points where both sides reach ``min_words``; among those,
    pause length and punctuation compete with evenness of the split.
    """
    words = [len(segments[i].text.split()) for i in indices]
    total = sum(words)
    half = total / 2.0
    viable: list[tuple[int, int]] = []  # (position, left words)
    fallback: list[tuple[int, int]] = []
    cum = 0
    for pos in range(len(indices) - 1):
        cum += words[pos]
        fallback.append((pos, cum))
        if cum >= min_words and total - cum >= min_words:
            viable.append((pos, cum))
    pool = viable or fallback
    if not pool:  # single-segment group: nothing to split
        return [indices]
    best_pos = pool[0][0]
    best_score = float("-inf")
    for pos, left in pool:
        score = _break_score(segments, eff_gaps, indices[pos])
        score -= 2.0 * abs(left - half) / max(half, 1.0)
        if score >= best_score:
            best_score = score
            best_pos = pos
    return [indices[: best_pos + 1], indices[best_pos + 1 :]]


def _fix_tiny_tail(
    segments: list[Segment],
    eff_gaps: list[float],
    groups: list[list[int]],
    *,
    pause_threshold_s: float,
    min_words: int,
    max_words: int,
) -> list[list[int]]:
    """Merge a tiny final paragraph of the *document* into its predecessor.

    Run-internal tails are already handled; this catches a whole final run
    that is undersized. The tail is kept standalone only when a very long
    pause (``_VERY_LONG_PAUSE_FACTOR * pause_threshold_s``) isolates it.
    """
    if len(groups) < 2:
        return groups
    tail = groups[-1]
    if _group_words(segments, tail) >= min_words:
        return groups
    boundary_gap = eff_gaps[tail[0] - 1] if tail[0] >= 1 else 0.0
    if boundary_gap > _VERY_LONG_PAUSE_FACTOR * pause_threshold_s:
        return groups
    combined = groups[-2] + tail
    if _group_words(segments, combined) <= max_words:
        return [*groups[:-2], combined]
    return [
        *groups[:-2],
        *_rebalance(segments, eff_gaps, combined, min_words=min_words, max_words=max_words),
    ]


def _group_words(segments: list[Segment], indices: list[int]) -> int:
    """Total word count across the segments at ``indices``."""
    return sum(len(segments[i].text.split()) for i in indices)


def _materialize(segments: list[Segment], indices: list[int]) -> Paragraph:
    """Build a Paragraph from segment indices: joined text, outer timestamps.

    Empty/whitespace-only segment texts are skipped in the join so they never
    inject double spaces; the span still uses the first/last segment's times.
    """
    text = " ".join(segments[i].text for i in indices if segments[i].text.strip())
    return Paragraph(
        text=text,
        start_s=segments[indices[0]].start_s,
        end_s=segments[indices[-1]].end_s,
    )
