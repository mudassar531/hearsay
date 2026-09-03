"""Segment word-level transcript timings into TTS/STT training clips — the D1 core.

Pure and deterministic (no I/O, no randomness, stdlib + pydantic only): a list of
time-aligned ``Word`` objects in, a list of ``DatasetSegment`` clips out. The clip
is sized by *duration in seconds* (not word count — that is the one structural
departure from the markdown grouper in ``grouping.py``, whose three-phase shape and
scoring vocabulary this mirrors), cut on sentence/pause boundaries, clamped to
``[min_s, max_s]``, and — crucially — **never split mid-word**.

Three phases:

1. **Repair + hard-break.** Each word's span is repaired *per word* (finite,
   non-negative, ``end >= start``) without mutating the originals; the original
   ``Word`` objects are what we emit, so the lossless partition is exact. Long
   inter-word silences become hard paragraph breaks ("runs").
2. **Grow-then-cut.** Inside each run a clip grows until its duration enters the
   ``[min_s, max_s]`` window, then breaks at the best-scoring boundary: pause is
   the dominant currency, refined by sentence/clause punctuation and the same
   dangling-word / discourse-marker cues as ``grouping.py`` (the input is now
   words — the granularity those cues were designed for).
3. **Tiny-tail fix.** A trailing clip shorter than ``min_s`` merges back into its
   predecessor unless a very long pause genuinely isolates it (an outro).

**Duration is computed from the segment's *extent*** — the minimum word start and
maximum word end over the range — not just its endpoint words. ASR timings are
noisy (faster-whisper can emit a glitched far-future end on an interior word); an
endpoint-only duration would let such a word hide inside an in-bounds-looking clip
and silently violate the duration bound. The extent both keeps every clip's span
honest (it always contains its words) and is monotone as a range grows (so the
grow loop's overshoot logic is valid). Correctness invariant: requirement
"never split a word" strictly dominates "respect ``max_s``" — an unbreakable run
is emitted whole and flagged ``oversized``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from hearsay.dataset.models import DatasetSegment
from hearsay.models import Word
from hearsay.punctuation import CLAUSE_END, ELLIPSIS_END, SENTENCE_END

# --- Boundary punctuation (shared with grouping.py via hearsay.punctuation, so an
#     Urdu ``۔`` or a Chinese ``。`` counts as a sentence end in both cutters). ---
_SENTENCE_END = SENTENCE_END
_CLAUSE_END = CLAUSE_END
_ELLIPSIS_END = ELLIPSIS_END
_ABBREVIATIONS = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e.",
        "jr.", "sr.", "prof.", "no.", "inc.", "co.", "fig.", "al.",
    }
)  # fmt: skip
_DOTTED_INITIAL = re.compile(r"^(?:[a-z]\.)+$")
_MIDDLE_INITIAL = re.compile(r"^[a-z]\.$")  # "J." — a name initial, not a full stop
_DECIMAL = re.compile(r"\d\.\d")  # "3.14" — the dot is not a sentence end
_OPEN_QUOTES = "\"“”‘’'([「『《〈【"
_CLOSE_QUOTES = "\"“”‘’')]」』》〉】"

# --- Word cues (mirror grouping.py) ---------------------------------------
_DANGLING_WORDS = frozenset(
    """
    a an the and or but so because if that which who whose whom of to in on
    at by for with from into onto as is are was were be been being am do
    does did have has had will would can could shall should may might must
    very really just quite their his her its our your my this these those
    i you we they he she it there when while than then though although
    unless until example um uh like
    """.split()  # noqa: SIM905
)
_CONTRACTION = re.compile(r"(?:n['’]t|['’](?:s|re|m|ve|ll|d))$")
_STRONG_MARKERS = frozenset(
    {
        "so", "now", "okay", "ok", "but", "anyway", "anyways", "alright",
        "next", "however", "finally", "meanwhile", "first", "firstly",
        "second", "secondly", "third", "lastly", "basically",
    }
)  # fmt: skip
_WEAK_MARKERS = frozenset({"and", "then", "well", "also"})
_FILLERS = frozenset({"um", "uh", "uhm", "umm", "er", "ah", "mm", "hmm"})
_STRIP_CHARS = "\"“”‘’'()[]"

# --- Scoring weights (seconds of pause are the dominant currency) ---------
_GAP_WEIGHT = 3.0  # one full second of pause = 3 points
_SENTENCE_BONUS = 2.0
_CLAUSE_BONUS = 0.6
_DANGLING_PENALTY = 2.5
_MARKER_BONUS = 0.8
_WEAK_MARKER_BONUS = 0.4
_CONTINUATION_PENALTY = 1.0
_BALANCE_WEIGHT = 0.8  # gentle pull toward the target clip duration
_TARGET_FRACTION = 0.45  # target sits just below the window midpoint
# A hard-break threshold below this is treated as this: inter-word gaps under half
# a second are breaths, not utterance boundaries. (Note: 0.5 s is close to real
# word gaps, so callers should keep pause_break_s well above the floor; a value
# near the floor shatters dense speech into one-word clips — losslessly, but
# uselessly.)
_MIN_PAUSE_FLOOR_S = 0.5


def segment_words(
    words: list[Word],
    *,
    min_s: float = 1.0,
    max_s: float = 15.0,
    pause_break_s: float = 2.0,
    target_s: float | None = None,
    isolate_factor: float = 2.0,
) -> list[DatasetSegment]:
    """Group time-aligned words into duration-bounded training clips.

    Args:
        words: Ordered, time-aligned words (verbatim text, possibly-noisy spans).
        min_s: Soft minimum clip duration (seconds). A trailing clip shorter than
            this is merged back unless a very long pause isolates it. Floored to a
            small positive value if non-positive.
        max_s: Maximum clip duration (seconds). Only exceeded by an unbreakable run
            (e.g. a single word longer than ``max_s``), which is emitted with
            ``oversized=True``. Raised to ``min_s`` if a caller passes less.
        pause_break_s: Inter-word silence (seconds) that forces a hard clip break.
            Floored to ``0.5`` so a tiny value cannot shatter the transcript.
        target_s: Ideal clip duration the size-balance term pulls toward; defaults
            to just below the window midpoint.
        isolate_factor: A tiny trailing clip survives standalone only when the pause
            before it exceeds ``isolate_factor`` times the transcript's typical pause.

    Returns:
        Clips whose concatenated words exactly equal the non-blank input words, in
        order, by identity (lossless; never mid-word). Each clip's span is the
        extent of its words; durations fall within ``[min_s, max_s]`` or the clip is
        flagged ``oversized``. Empty input yields ``[]``; whitespace-only words
        never produce empty clips.
    """
    if not words:
        return []

    # Clamp degenerate parameters so the function is always well-defined.
    if not (min_s > 0):
        min_s = 1.0
    if max_s < min_s:
        max_s = min_s
    pause_break_s = max(pause_break_s, _MIN_PAUSE_FLOOR_S)

    n = len(words)
    r_start, r_end, gap_after, blank, sent, clause = _prepare(words)

    # Phase 1: hard-break on long pauses -> runs of [lo, hi) index spans.
    runs: list[tuple[int, int]] = []
    run_start = 0
    for i in range(n - 1):
        if gap_after[i] > pause_break_s:
            runs.append((run_start, i + 1))
            run_start = i + 1
    runs.append((run_start, n))

    # Phase 2: size each run by the seconds budget.
    pieces: list[tuple[int, int]] = []  # inclusive (a, b) index ranges
    for lo, hi in runs:
        pieces.extend(
            _split_run(
                r_start,
                r_end,
                gap_after,
                sent,
                clause,
                words,
                lo,
                hi,
                min_s=min_s,
                max_s=max_s,
                target_s=target_s,
            )
        )

    # Phase 3: merge a tiny trailing clip back unless a long pause isolates it.
    pieces = _fix_tiny_tail(
        r_start,
        r_end,
        gap_after,
        sent,
        clause,
        words,
        pieces,
        min_s=min_s,
        max_s=max_s,
        isolate_factor=isolate_factor,
    )

    out: list[DatasetSegment] = []
    for a, b in pieces:
        if all(blank[i] for i in range(a, b + 1)):
            continue  # an all-whitespace range never becomes an empty segment
        out.append(_materialize(words, r_start, r_end, blank, a, b, max_s))
    return out


def _prepare(
    words: list[Word],
) -> tuple[list[float], list[float], list[float], list[bool], list[bool], list[bool]]:
    """Repair each word's span (per word, no cross-word mutation) and derive cues.

    Returns parallel arrays: repaired start/end (finite, non-negative, end>=start),
    forward inter-word gap (clamped >= 0 so overlaps/out-of-order timings never
    produce a phantom negative pause), and blank/sentence/clause flags. A word whose
    successor carries ``joins_left`` ends neither a sentence nor a clause: the
    tokenizer has said the two are one unit. The original ``Word`` objects are left
    untouched.
    """
    n = len(words)
    r_start = [0.0] * n
    r_end = [0.0] * n
    blank = [False] * n
    sent = [False] * n
    clause = [False] * n
    for i, w in enumerate(words):
        s = w.start_s
        e = w.end_s
        if not math.isfinite(s):
            s = 0.0
        if not math.isfinite(e):
            e = s
        s = max(s, 0.0)
        e = max(e, s)  # repair inverted / zero-length spans
        r_start[i] = s
        r_end[i] = e
        t = w.text.strip()
        if t:
            # ASR tokenizers split inside a word or a number — Whisper emits "19.8" as
            # ["19.", "8"] — and mark the continuation with joins_left. A boundary in
            # the middle of one token cannot end a sentence or a clause, however much
            # the fragment looks like it does, so the cues are suppressed there.
            continues = i + 1 < n and words[i + 1].joins_left
            sent[i] = not continues and _is_sentence_end(t)
            clause[i] = not continues and _is_clause_end(t)
        else:
            blank[i] = True
    gap_after = [0.0] * n
    for i in range(n - 1):
        gap_after[i] = max(0.0, r_start[i + 1] - r_end[i])
    return r_start, r_end, gap_after, blank, sent, clause


def _extent(r_start: list[float], r_end: list[float], a: int, b: int) -> tuple[float, float]:
    """The (min start, max end) over the inclusive index range — the true extent.

    Using the range extent (not the endpoint words) keeps a clip's span honest in
    the presence of a glitched interior word, and is monotone as the range grows.
    """
    lo = r_start[a]
    hi = r_end[a]
    for i in range(a + 1, b + 1):
        if r_start[i] < lo:
            lo = r_start[i]
        if r_end[i] > hi:
            hi = r_end[i]
    return lo, hi


def _dur(r_start: list[float], r_end: list[float], a: int, b: int) -> float:
    """Duration (seconds) of the inclusive range, from its extent (always >= 0)."""
    lo, hi = _extent(r_start, r_end, a, b)
    return hi - lo


def _split_run(
    r_start: list[float],
    r_end: list[float],
    gap_after: list[float],
    sent: list[bool],
    clause: list[bool],
    words: list[Word],
    rlo: int,
    rhi: int,
    *,
    min_s: float,
    max_s: float,
    target_s: float | None,
) -> list[tuple[int, int]]:
    """Split a run (no internal hard pauses) into duration-sized clips.

    Greedy: grow a clip until its extent-duration reaches the ``[min_s, max_s]``
    window, then break at the best-scoring interior boundary. A run with no
    in-window boundary breaks just before the word that overshoots ``max_s``; a
    single word already longer than ``max_s`` is emitted alone (oversized).
    """
    out: list[tuple[int, int]] = []
    i = rlo
    while i < rhi:
        candidates: list[tuple[int, float]] = []  # (cut index k, left duration)
        overshoot: int | None = None
        lo = r_start[i]
        hi = r_end[i]
        k = i
        while k < rhi:
            if r_start[k] < lo:
                lo = r_start[k]
            if r_end[k] > hi:
                hi = r_end[k]
            dur = hi - lo
            if min_s <= dur <= max_s and k < rhi - 1:
                candidates.append((k, dur))  # a real interior boundary (k+1 exists)
            if dur >= max_s:
                overshoot = k
                break
            k += 1

        if overshoot is None:
            # The rest of the run fits under max_s: take it all (Phase 3 fixes a
            # too-short final clip).
            out.append((i, rhi - 1))
            break

        if candidates:
            cut = _best_cut(
                r_start,
                r_end,
                gap_after,
                sent,
                clause,
                words,
                i,
                candidates,
                min_s=min_s,
                max_s=max_s,
                target_s=target_s,
            )
        elif overshoot > i:
            # No in-window boundary (a long word skipped the budget window): break
            # just before the word that broke max_s, so the clip stays <= max_s.
            cut = overshoot - 1
        else:
            # A single word at i already exceeds max_s: unbreakable, emit it alone.
            cut = i
        out.append((i, cut))
        i = cut + 1
    return out


def _best_cut(
    r_start: list[float],
    r_end: list[float],
    gap_after: list[float],
    sent: list[bool],
    clause: list[bool],
    words: list[Word],
    i: int,
    candidates: list[tuple[int, float]],
    *,
    min_s: float,
    max_s: float,
    target_s: float | None,
) -> int:
    """Pick the best break among in-window candidates (pause-first, two-sided cues).

    Score = pause length (dominant) + sentence/clause bonus or dangling penalty on
    the ending word + a discourse-marker / continuation cue on the next word + a
    gentle pull toward the target duration. Later candidates win ties (``>=``),
    biasing toward fuller clips and fewer tiny tails.
    """
    if target_s is None or not math.isfinite(target_s):
        target = min_s + _TARGET_FRACTION * (max_s - min_s)
    else:
        target = target_s
    target = min(max_s, max(min_s, target))
    span = max(max_s - min_s, 1e-9)
    n = len(words)
    best_k = candidates[0][0]
    best_score = float("-inf")
    for k, left in candidates:
        score = _GAP_WEIGHT * gap_after[k]
        if sent[k]:
            score += _SENTENCE_BONUS
        elif clause[k]:
            score += _CLAUSE_BONUS
        else:
            tokens = words[k].text.split()
            last_word = tokens[-1].strip(_STRIP_CHARS).lower() if tokens else ""
            if last_word in _DANGLING_WORDS or _CONTRACTION.search(last_word):
                score -= _DANGLING_PENALTY
            elif any(ch.isdigit() for ch in last_word):
                score -= _DANGLING_PENALTY / 2.0
        if k + 1 < n:
            score += _begin_score(words[k + 1].text, after_sentence_end=sent[k])
        score -= _BALANCE_WEIGHT * abs(left - target) / span
        if score >= best_score:
            best_score = score
            best_k = k
    return best_k


def _begin_score(next_text: str, *, after_sentence_end: bool) -> float:
    """Score the opening of the next clip as a break cue (mirrors grouping.py)."""
    tokens = next_text.lower().split()
    index = 0
    while index < len(tokens) and tokens[index].strip(_STRIP_CHARS) in _FILLERS:
        index += 1
    if index >= len(tokens):
        return 0.0
    head = tokens[index].strip(_STRIP_CHARS)
    if head in _STRONG_MARKERS:
        return _MARKER_BONUS
    if head in _WEAK_MARKERS:
        return _WEAK_MARKER_BONUS
    if head in _DANGLING_WORDS and not after_sentence_end:
        return -_CONTINUATION_PENALTY
    return 0.0


def _fix_tiny_tail(
    r_start: list[float],
    r_end: list[float],
    gap_after: list[float],
    sent: list[bool],
    clause: list[bool],
    words: list[Word],
    pieces: list[tuple[int, int]],
    *,
    min_s: float,
    max_s: float,
    isolate_factor: float,
) -> list[tuple[int, int]]:
    """Merge a too-short final clip into its predecessor, unless a long pause isolates it."""
    if len(pieces) < 2:
        return pieces
    ta, tb = pieces[-1]
    if _dur(r_start, r_end, ta, tb) >= min_s:
        return pieces
    scale = _isolation_scale(gap_after)
    if ta >= 1 and gap_after[ta - 1] > isolate_factor * scale:
        return pieces  # a genuinely long pause isolates the tail (e.g. an outro)
    pa, _ = pieces[-2]
    if _dur(r_start, r_end, pa, tb) <= max_s:
        return [*pieces[:-2], (pa, tb)]  # plain merge stays in bounds
    return [
        *pieces[:-2],
        *_rebalance(r_start, r_end, gap_after, sent, clause, pa, tb, min_s=min_s),
    ]


def _rebalance(
    r_start: list[float],
    r_end: list[float],
    gap_after: list[float],
    sent: list[bool],
    clause: list[bool],
    a: int,
    b: int,
    *,
    min_s: float,
) -> list[tuple[int, int]]:
    """Split one over-budget range into two well-placed halves (mirrors grouping.py)."""
    total = _dur(r_start, r_end, a, b)
    half = total / 2.0
    viable: list[int] = []
    fallback: list[int] = []
    for k in range(a, b):  # break after word k
        fallback.append(k)
        if _dur(r_start, r_end, a, k) >= min_s and _dur(r_start, r_end, k + 1, b) >= min_s:
            viable.append(k)
    pool = viable or fallback
    if not pool:
        return [(a, b)]  # single word: nothing to split
    best_k = pool[0]
    best_score = float("-inf")
    for k in pool:
        score = _GAP_WEIGHT * gap_after[k]
        if sent[k]:
            score += _SENTENCE_BONUS
        elif clause[k]:
            score += _CLAUSE_BONUS
        score -= 2.0 * abs(_dur(r_start, r_end, a, k) - half) / max(half, 1e-9)
        if score >= best_score:
            best_score = score
            best_k = k
    return [(a, best_k), (best_k + 1, b)]


def _isolation_scale(gap_after: list[float]) -> float:
    """Transcript-relative pause scale: median of positive gaps, floored.

    Used to decide whether a long pause truly isolates a tiny tail. Transcript-
    relative (not a fixed multiple of ``pause_break_s``) so jittery word timings
    don't trip false isolation; floored so an all-zero-gap transcript can't make
    the threshold zero.
    """
    positive = sorted(g for g in gap_after if g > 0)
    if not positive:
        return _MIN_PAUSE_FLOOR_S
    mid = len(positive) // 2
    median = positive[mid] if len(positive) % 2 else (positive[mid - 1] + positive[mid]) / 2.0
    return max(median, _MIN_PAUSE_FLOOR_S)


def _join_words(words: Iterable[Word]) -> str:
    """Join words back into text, honouring tokens that continue the previous word.

    ASR tokenizers split inside words, and a blind space rejoined Uzbek
    ``["qo'shig", "'i."]`` as ``qo'shig 'i.`` — two broken tokens in the transcript
    shipped alongside the audio. ``joins_left`` carries the engine's own answer.
    """
    out: list[str] = []
    for word in words:
        text = word.text.strip()
        if out and word.joins_left:
            out[-1] += text
        else:
            out.append(text)
    return " ".join(out)


def _materialize(
    words: list[Word],
    r_start: list[float],
    r_end: list[float],
    blank: list[bool],
    a: int,
    b: int,
    max_s: float,
) -> DatasetSegment:
    """Build a DatasetSegment from an index range: verbatim text + extent span.

    ``words`` are the original objects (never copied/mutated). ``text`` joins the
    non-blank words with single spaces. ``start_s``/``end_s`` are the range extent
    so the span contains every member word; ``oversized`` is honest (extent > max_s).
    """
    text = _join_words(words[i] for i in range(a, b + 1) if not blank[i])
    lo, hi = _extent(r_start, r_end, a, b)
    return DatasetSegment(
        text=text,
        start_s=lo,
        end_s=max(hi, lo),
        words=list(words[a : b + 1]),
        oversized=(hi - lo) > max_s,
    )


def _is_sentence_end(text: str) -> bool:
    """True if ``text`` ends a sentence (``. ! ?``), excluding ellipsis/abbreviations."""
    t = text.strip()
    if _ELLIPSIS_END.search(t):
        return False  # ellipsis is a soft (clause-tier) boundary
    if not _SENTENCE_END.search(t):
        return False
    return not _ends_abbrev(t)


def _is_clause_end(text: str) -> bool:
    """True if ``text`` ends on a clause mark (``, ; :`` or a dash) or an ellipsis."""
    t = text.strip()
    return bool(_CLAUSE_END.search(t)) or bool(_ELLIPSIS_END.search(t))


def _ends_abbrev(text: str) -> bool:
    """True if the final token is an abbreviation, initialism, initial, or decimal."""
    tokens = text.split()
    if not tokens:
        return False
    last = tokens[-1].lstrip(_OPEN_QUOTES).rstrip(_CLOSE_QUOTES).lower()
    return (
        last in _ABBREVIATIONS
        or bool(_DOTTED_INITIAL.fullmatch(last))
        or bool(_MIDDLE_INITIAL.fullmatch(last))
        or bool(_DECIMAL.search(last))
    )
