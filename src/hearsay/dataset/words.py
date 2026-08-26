"""Engine-agnostic word adapters: ASR output -> normalized ``Word`` objects.

faster-whisper and Parakeet expose word/token timings under different shapes and
field names. These pure, duck-typed adapters normalize both into hearsay's
:class:`~hearsay.models.Word` so the segmenter (and the rest of dataset mode)
works identically regardless of engine. No I/O, no heavy imports — testable
offline against plain stand-in objects.

* faster-whisper: ``Word(word, start, end, probability)`` — a flat list per
  segment (populated only when ``word_timestamps=True``). Usually one token per word,
  but it splits inside a word often enough to matter (Uzbek ``qo'shig'i.`` arrives as
  ``["qo'shig", "'i."]``); a leading space marks a genuine word start, so its absence
  sets ``joins_left``.
* Parakeet (parakeet-mlx): ``AlignedToken(id, text, start, duration, confidence,
  end)`` — *subword* pieces; a new word begins wherever a token's text starts
  with a literal space (not the SentencePiece ``U+2581``). Subwords are merged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from hearsay.models import Word


def _clamp01(value: Any) -> float:
    """Coerce an engine score to a finite probability in ``[0, 1]`` (default 1.0)."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(score):
        return 1.0
    return min(1.0, max(0.0, score))


def _finite_float(value: Any, default: float) -> float:
    """Coerce a timing value to a finite float, falling back to ``default``.

    Keeps the adapters genuinely duck-typed/defensive: a missing, ``None``, or
    non-finite ``start``/``end``/``duration`` yields ``default`` instead of a
    ``TypeError``. (The supported engines always populate these, so this is a
    contract guard, not a hot path.)
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def words_from_whisper(whisper_words: Iterable[Any]) -> list[Word]:
    """Convert a flat iterable of faster-whisper ``Word`` objects to ``Word``s.

    Each input must expose ``.word``, ``.start``, ``.end`` and (optionally)
    ``.probability``. Whitespace-only words are dropped and the text is stripped, but
    the leading space is read first: it is how faster-whisper marks a genuine word
    start, and dropping that turned Uzbek ``qo'shig'i.`` into ``qo'shig 'i.``.
    """
    out: list[Word] = []
    for w in whisper_words:
        raw = getattr(w, "word", None)
        if raw is None:
            continue
        raw = str(raw)
        text = raw.strip()
        if not text:
            continue
        # faster-whisper prefixes a token with a space exactly when it begins a new
        # word; its absence means this token belongs to the one before it.
        joins_left = not raw.startswith((" ", "\u00a0"))
        start = _finite_float(getattr(w, "start", 0.0), 0.0)
        end = _finite_float(getattr(w, "end", start), start)
        out.append(
            Word(
                text=text,
                start_s=start,
                end_s=end,
                confidence=_clamp01(getattr(w, "probability", 1.0)),
                joins_left=joins_left,
            )
        )
    return out


def words_from_parakeet(tokens: Iterable[Any]) -> list[Word]:
    """Merge Parakeet subword ``AlignedToken``s into whole ``Word``s.

    Each token must expose ``.text``, ``.start`` and ``.duration`` (and
    optionally ``.end``/``.confidence``). A new word starts at the first token
    and whenever a token's raw text begins with a space; the word spans its
    first sub-token's start to its last sub-token's end, with confidence the
    *geometric* mean of its sub-tokens' (matching parakeet-mlx's own per-sentence
    aggregation). Whitespace-only merges are dropped.
    """
    words: list[Word] = []
    text = ""
    start: float | None = None
    end = 0.0
    confs: list[float] = []

    def flush() -> None:
        nonlocal text, start, end, confs
        if start is not None:
            stripped = text.strip()
            if stripped:
                words.append(
                    Word(
                        text=stripped,
                        start_s=start,
                        end_s=max(end, start),
                        confidence=_geometric_mean(confs),
                    )
                )
        text = ""
        start = None
        end = 0.0
        confs = []

    for tok in tokens:
        raw = str(getattr(tok, "text", ""))
        if start is not None and raw.startswith(" "):
            flush()
        tok_start = _finite_float(getattr(tok, "start", 0.0), 0.0)
        if start is None:
            start = tok_start
        text += raw
        tok_end = getattr(tok, "end", None)
        if tok_end is not None:
            end = _finite_float(tok_end, tok_start)
        else:
            end = tok_start + _finite_float(getattr(tok, "duration", 0.0), 0.0)
        confs.append(_clamp01(getattr(tok, "confidence", 1.0)))
    flush()
    return words


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean of confidences in ``[0, 1]`` (parakeet-mlx's aggregation)."""
    if not values:
        return 1.0
    total = sum(math.log(v + 1e-10) for v in values)
    return _clamp01(math.exp(total / len(values)))
