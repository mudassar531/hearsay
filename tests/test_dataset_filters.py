"""Tests for the quality filters (Tier-1 dep-free + opt-in Tier-2 clipping)."""

import wave
from pathlib import Path

import numpy as np

from hearsay.dataset.filters import detect_clipping_drops, filter_segments
from hearsay.dataset.models import DatasetClip, DatasetSegment, FilterConfig
from hearsay.models import Word


def _seg(words: list[Word], oversized: bool = False) -> DatasetSegment:
    text = " ".join(w.text for w in words if w.text.strip())
    start = min((w.start_s for w in words), default=0.0)
    end = max((w.end_s for w in words), default=0.0)
    return DatasetSegment(text=text, start_s=start, end_s=end, words=words, oversized=oversized)


def _drop_reason(words: list[Word], duration_s: float, **cfg) -> str | None:
    seg = _seg(words)
    kept, drops = filter_segments([("ref_0001", seg, duration_s)], FilterConfig(**cfg))
    if drops:
        assert not kept
        return drops[0].filter
    assert len(kept) == 1
    return None


def _words(
    texts: list[str], *, conf: float = 0.9, step: float = 0.5, gap: float = 0.0
) -> list[Word]:
    words, t = [], 0.0
    for text in texts:
        words.append(Word(text=text, start_s=t, end_s=t + step, confidence=conf))
        t += step + gap
    return words


# --- Tier-1 filters: each reason -------------------------------------------


def test_good_clip_passes() -> None:
    assert _drop_reason(_words(["hello", "there", "friend"]), 1.5) is None


def test_too_short_dropped() -> None:
    assert _drop_reason(_words(["hi", "yo"]), 0.5) == "duration"


def test_too_long_dropped() -> None:
    assert _drop_reason(_words(["a", "b", "c", "d"]), 20.0) == "duration"


def test_empty_text_dropped() -> None:
    assert _drop_reason([Word(text="   ", start_s=0.0, end_s=2.0)], 2.0) == "empty_text"


def test_internal_silence_dropped() -> None:
    words = [
        Word(text="hello", start_s=0.0, end_s=0.5),
        Word(text="world", start_s=4.0, end_s=4.5),  # 3.5s internal gap
    ]
    assert _drop_reason(words, 4.5) == "internal_silence"


def test_non_target_script_dropped() -> None:
    words = _words(["你好", "世界", "今天", "天气"])  # CJK, target default "en"
    assert _drop_reason(words, 2.0) == "non_target_script"


def test_non_target_script_respects_target_language() -> None:
    words = _words(["你好", "世界", "今天"])
    # With the target set to Chinese, the same text passes the script filter.
    assert _drop_reason(words, 2.0, target_language="zh") != "non_target_script"


def test_mixed_script_kept_regardless_of_word_order() -> None:
    # 50/50 Latin/Cyrillic: no strict majority -> kept, and the verdict must not
    # depend on word order (was non-deterministic before the strict-majority fix).
    assert _drop_reason(_words(["ab", "вг"]), 1.0) != "non_target_script"
    assert _drop_reason(_words(["вг", "ab"]), 1.0) != "non_target_script"


def test_char_rate_skipped_for_unmapped_language() -> None:
    # Thai is space-free and not in the script map; the Latin char-rate bounds must
    # not apply, so a slow-cps Thai clip is not falsely dropped.
    words = _words(["สวัสดีครับ"])
    assert _drop_reason(words, 10.0, target_language="th") is None


def test_internal_silence_not_masked_by_glitched_word_end() -> None:
    # "so" has a glitched far-future end (7.0); the real ~silence before "next" must
    # still be detected (the end is capped, not trusted).
    words = [
        Word(text="okay", start_s=0.0, end_s=0.5),
        Word(text="so", start_s=1.0, end_s=7.0),  # glitched end
        Word(text="next", start_s=6.0, end_s=6.5),
    ]
    assert _drop_reason(words, 6.5) == "internal_silence"


def test_short_circuit_reports_first_failure() -> None:
    # A clip that is both too short AND wrong-script reports "duration" (checked first).
    assert _drop_reason(_words(["你好", "世界"]), 0.3) == "duration"


def test_char_rate_too_low_dropped() -> None:
    # 2 chars over 10s -> ~0.2 cps, well under the floor.
    assert _drop_reason([Word(text="hi", start_s=0.0, end_s=10.0)], 10.0) == "char_rate"


def test_char_rate_too_high_dropped() -> None:
    words = [Word(text="supercalifragilisticexpialidocious", start_s=0.0, end_s=1.0)]
    assert _drop_reason(words, 1.0) == "char_rate"  # 34 chars / 1s = 34 cps


def test_compression_ratio_dropped() -> None:
    words = _words(["na"] * 40, step=0.25)  # highly repetitive
    assert _drop_reason(words, 10.0) == "compression_ratio"


def test_low_confidence_dropped() -> None:
    words = _words(["hello", "world", "today", "now"], conf=0.1)
    assert _drop_reason(words, 2.0) == "confidence"


def test_filters_disabled_keeps_everything() -> None:
    words = _words(["你好", "世界"], conf=0.05)  # would fail script AND confidence
    seg = _seg(words)
    kept, drops = filter_segments([("r", seg, 0.1)], FilterConfig(enabled=False))
    assert len(kept) == 1 and drops == []


def test_drop_record_has_context() -> None:
    seg = _seg(_words(["hi"]))
    _, drops = filter_segments([("ref_0001", seg, 0.3)], FilterConfig())
    rec = drops[0]
    assert rec.clip == "ref_0001"
    assert rec.filter == "duration"
    assert "0.30" in rec.value and "hi" in rec.text


# --- Tier-2: clipping (opt-in) ---------------------------------------------


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype("<i2").tobytes())


def test_clipping_detection(tmp_path: Path) -> None:
    (tmp_path / "wavs").mkdir()
    _write_wav(tmp_path / "wavs" / "clean.wav", np.full(22050, 8000, dtype=np.int16))
    _write_wav(tmp_path / "wavs" / "clipped.wav", np.full(22050, 32767, dtype=np.int16))
    clips = [
        DatasetClip(
            id="clean", audio_path="wavs/clean.wav", text="ok", start_s=0, end_s=1, duration_s=1.0
        ),
        DatasetClip(
            id="clipped",
            audio_path="wavs/clipped.wav",
            text="bad",
            start_s=0,
            end_s=1,
            duration_s=1.0,
        ),
    ]
    kept, drops = detect_clipping_drops(clips, tmp_path, FilterConfig(detect_clipping=True))
    assert [c.id for c in kept] == ["clean"]
    assert [d.filter for d in drops] == ["clipping"]


def test_clipping_detects_negative_rail(tmp_path: Path) -> None:
    # np.abs(int16 -32768) overflows to -32768; the detector must still catch
    # negative-rail clipping (it widens to int32 first).
    (tmp_path / "wavs").mkdir()
    _write_wav(tmp_path / "wavs" / "neg.wav", np.full(22050, -32768, dtype=np.int16))
    clip = DatasetClip(
        id="neg", audio_path="wavs/neg.wav", text="bad", start_s=0, end_s=1, duration_s=1.0
    )
    kept, drops = detect_clipping_drops([clip], tmp_path, FilterConfig(detect_clipping=True))
    assert kept == []
    assert [d.filter for d in drops] == ["clipping"]


def test_clipping_off_by_default(tmp_path: Path) -> None:
    (tmp_path / "wavs").mkdir()
    _write_wav(tmp_path / "wavs" / "clipped.wav", np.full(100, 32767, dtype=np.int16))
    clip = DatasetClip(
        id="clipped", audio_path="wavs/clipped.wav", text="bad", start_s=0, end_s=1, duration_s=1.0
    )
    kept, drops = detect_clipping_drops([clip], tmp_path, FilterConfig())  # detect_clipping=False
    assert kept == [clip] and drops == []
