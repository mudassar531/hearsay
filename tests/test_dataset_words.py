"""Tests for the engine-agnostic word adapters (faster-whisper / Parakeet -> Word)."""

import math
from types import SimpleNamespace

import pytest

from hearsay.dataset.words import words_from_parakeet, words_from_whisper
from hearsay.models import Word


def _whisper_word(word: str, start: float, end: float, probability: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(word=word, start=start, end=end, probability=probability)


def _token(
    text: str, start: float | None, duration: float, confidence: float = 1.0, end=...
) -> SimpleNamespace:
    ns = SimpleNamespace(text=text, start=start, duration=duration, confidence=confidence)
    if end is not ...:  # allow omitting .end to exercise the start+duration fallback
        ns.end = end
    return ns


# --- faster-whisper adapter ------------------------------------------------


def test_whisper_words_basic_rename() -> None:
    ws = [_whisper_word(" Hello", 0.0, 0.5, 0.9), _whisper_word(" world", 0.6, 1.0, 0.8)]
    out = words_from_whisper(ws)
    assert [w.text for w in out] == ["Hello", "world"]  # leading space stripped
    assert out[0] == Word(text="Hello", start_s=0.0, end_s=0.5, confidence=0.9)
    assert out[1].start_s == 0.6 and out[1].end_s == 1.0


def test_whisper_words_drops_blank_tokens() -> None:
    ws = [_whisper_word(" ", 0.0, 0.2), _whisper_word("hi", 0.2, 0.5)]
    out = words_from_whisper(ws)
    assert [w.text for w in out] == ["hi"]


def test_whisper_word_none_dropped_not_stringified() -> None:
    # A None .word must be dropped, not turned into the literal text "None".
    ws = [SimpleNamespace(word=None, start=0.0, end=0.3), _whisper_word("hi", 0.3, 0.6)]
    assert [w.text for w in words_from_whisper(ws)] == ["hi"]


def test_adapters_tolerate_none_timings() -> None:
    # Defensive: a None/missing start/end must not raise (duck-typed contract).
    w = words_from_whisper([SimpleNamespace(word="hi", start=None, end=0.5)])
    assert w[0].start_s == 0.0 and w[0].end_s == 0.5
    tok = words_from_parakeet([_token("hi", None, 0.4)])
    assert tok[0].start_s == 0.0 and tok[0].end_s == pytest.approx(0.4)


def test_whisper_missing_probability_defaults_to_one() -> None:
    w = SimpleNamespace(word="hi", start=0.0, end=0.3)  # no .probability
    out = words_from_whisper([w])
    assert out[0].confidence == 1.0


def test_whisper_probability_clamped() -> None:
    out = words_from_whisper(
        [
            _whisper_word("a", 0.0, 0.1, 1.5),  # > 1 -> 1.0
            _whisper_word("b", 0.1, 0.2, -0.3),  # < 0 -> 0.0
            _whisper_word("c", 0.2, 0.3, float("nan")),  # nan -> 1.0
        ]
    )
    assert [w.confidence for w in out] == [1.0, 0.0, 1.0]


def test_whisper_empty_iterable() -> None:
    assert words_from_whisper([]) == []


# --- Parakeet adapter (subword token merge) --------------------------------


def test_parakeet_merges_subwords_on_leading_space() -> None:
    tokens = [
        _token("Hel", 0.0, 0.2, 0.8, end=0.2),
        _token("lo", 0.2, 0.1, 0.6, end=0.3),
        _token(" wor", 0.5, 0.2, 0.9, end=0.7),  # leading space -> new word
        _token("ld", 0.7, 0.1, 0.7, end=0.8),
    ]
    out = words_from_parakeet(tokens)
    assert [w.text for w in out] == ["Hello", "world"]
    assert out[0].start_s == 0.0 and out[0].end_s == 0.3
    assert out[1].start_s == 0.5 and out[1].end_s == 0.8
    # confidence is the geometric mean of a word's sub-token confidences
    # (matching parakeet-mlx's own per-sentence aggregation)
    assert out[0].confidence == pytest.approx(math.sqrt(0.8 * 0.6), abs=1e-6)
    assert out[1].confidence == pytest.approx(math.sqrt(0.9 * 0.7), abs=1e-6)


def test_parakeet_leading_space_on_first_token() -> None:
    out = words_from_parakeet([_token(" the", 0.0, 0.3), _token(" end", 0.4, 0.3)])
    assert [w.text for w in out] == ["the", "end"]


def test_parakeet_end_falls_back_to_start_plus_duration() -> None:
    # No .end attribute -> end = start + duration.
    out = words_from_parakeet([_token("hi", 1.0, 0.4)])
    assert out[0].start_s == 1.0
    assert out[0].end_s == 1.4


def test_parakeet_empty_iterable() -> None:
    assert words_from_parakeet([]) == []
