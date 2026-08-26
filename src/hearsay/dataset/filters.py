"""Quality filters that drop junk clips, each logged with a reason.

Tier-1 (default on, no new dependency, engine-agnostic) runs on data hearsay
already has — clip duration, word timings, and the transcript — so it needs no
audio decode:

* **duration** — drop clips shorter than ``min_duration_s`` or longer than
  ``max_duration_s`` (an oversized clip is caught here).
* **empty_text** — defensive drop of an empty transcript.
* **internal_silence** — drop a clip with an internal inter-word gap longer than
  ``max_internal_gap_s`` (a join of two utterances).
* **char_rate** — drop clips whose characters-per-second is implausible (skipped
  for logographic and space-free scripts, where the measure is meaningless).
* **non_target_script** — drop clips whose text is mostly the wrong script for the
  target language (a dep-free proxy for a language mismatch).
* **compression_ratio** — drop repetitive/hallucinated text (Whisper's gzip-ratio
  heuristic, computed on the text directly).
* **confidence** — drop clips whose mean word confidence is below
  ``min_avg_confidence`` (an engine-agnostic stand-in for Whisper's
  ``avg_logprob``/``no_speech_prob``).

Tier-2 (``detect_clipping``, opt-in) reads the sliced WAV (stdlib ``wave`` +
numpy, both already available) and drops clips with hard digital clipping.

Each drop is recorded as a :class:`DropRecord` so the build can write ``dropped.jsonl``
and a kept/dropped-by-reason summary. Filters short-circuit on the first failure,
so each dropped clip has exactly one reason.
"""

from __future__ import annotations

import itertools
import zlib
from collections import Counter
from pathlib import Path

from hearsay.dataset.models import DatasetClip, DatasetSegment, DropRecord, FilterConfig
from hearsay.models import Word

# Unicode letter ranges per script (enough to tell mainstream scripts apart).
_SCRIPT_RANGES: list[tuple[str, list[tuple[int, int]]]] = [
    ("latin", [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)]),
    ("greek", [(0x370, 0x3FF)]),
    ("cyrillic", [(0x400, 0x4FF)]),
    ("hebrew", [(0x590, 0x5FF)]),
    ("arabic", [(0x600, 0x6FF)]),
    ("devanagari", [(0x900, 0x97F)]),
    ("bengali", [(0x980, 0x9FF)]),
    ("gurmukhi", [(0xA00, 0xA7F)]),
    ("gujarati", [(0xA80, 0xAFF)]),
    ("tamil", [(0xB80, 0xBFF)]),
    ("telugu", [(0xC00, 0xC7F)]),
    ("kannada", [(0xC80, 0xCFF)]),
    ("malayalam", [(0xD00, 0xD7F)]),
    ("sinhala", [(0xD80, 0xDFF)]),
    ("thai", [(0xE00, 0xE7F)]),
    ("lao", [(0xE80, 0xEFF)]),
    ("myanmar", [(0x1000, 0x109F)]),
    ("georgian", [(0x10A0, 0x10FF), (0x1C90, 0x1CBF)]),
    ("ethiopic", [(0x1200, 0x137F)]),
    ("khmer", [(0x1780, 0x17FF)]),
    ("armenian", [(0x530, 0x58F)]),
    ("cjk", [(0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3)]),
]
# Map a target language code to its script. Codes absent here disable the
# script/char-rate filters rather than guessing.
_LANG_SCRIPT = {
    "en": "latin", "es": "latin", "fr": "latin", "de": "latin", "it": "latin",
    "pt": "latin", "nl": "latin", "sv": "latin", "no": "latin", "da": "latin",
    "fi": "latin", "pl": "latin", "cs": "latin", "ro": "latin", "tr": "latin",
    "id": "latin", "vi": "latin",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic", "sr": "cyrillic",
    "el": "greek", "he": "hebrew", "iw": "hebrew", "ar": "arabic", "fa": "arabic",
    # Perso-Arabic script. Urdu especially needs this: Whisper auto-detects Urdu audio
    # as Hindi (measured p=0.91 on a real podcast) and then emits Devanagari, so a whole
    # dataset can come back in the wrong script while the card still says "ur". Without
    # an entry here the script filter is disabled and nothing catches it.
    "ur": "arabic", "ps": "arabic", "sd": "arabic",
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "sa": "devanagari",
    # Single-script languages. An absent entry disables the wrong-script filter
    # silently, which is worst for exactly the low-resource languages most likely to
    # come back in the wrong script: a real Bengali news bulletin returned Telugu and
    # English under `language: "bn"` with zero non_target_script drops. Uzbek stays
    # unmapped on purpose — it is written in both Latin and Cyrillic.
    "bn": "bengali", "as": "bengali", "pa": "gurmukhi", "gu": "gujarati",
    "ta": "tamil", "te": "telugu", "kn": "kannada", "ml": "malayalam",
    "si": "sinhala", "th": "thai", "lo": "lao", "my": "myanmar",
    "km": "khmer", "am": "ethiopic", "ka": "georgian", "hy": "armenian",
    "sw": "latin", "af": "latin", "ca": "latin", "hr": "latin", "hu": "latin",
    "sk": "latin", "sl": "latin", "lt": "latin", "lv": "latin", "et": "latin",
    "is": "latin", "ms": "latin", "tl": "latin", "sq": "latin", "cy": "latin",
    "zh": "cjk", "ja": "cjk", "ko": "cjk",
}  # fmt: skip
# Scripts where a characters-per-second bound tuned on spaced Latin text is
# meaningless: logographic (CJK) and the space-free abugidas/abjads.
_UNSPACED_SCRIPTS = {"cjk", "thai", "lao", "myanmar", "khmer"}


def _char_script(ch: str) -> str | None:
    """Script of a single character, or None for non-letters/unknown scripts."""
    if not ch.isalpha():
        return None
    code = ord(ch)
    for name, ranges in _SCRIPT_RANGES:
        if any(lo <= code <= hi for lo, hi in ranges):
            return name
    return "other"


def _dominant_script(text: str) -> tuple[str, float] | None:
    """The most-common script of a string's letters and its share, or None if it has none."""
    counts = Counter(s for ch in text if (s := _char_script(ch)) is not None)
    if not counts:
        return None
    script, n = counts.most_common(1)[0]
    return script, n / sum(counts.values())


def _target_script(language: str | None) -> str | None:
    """The script a language is written in, or None to disable the script filters."""
    if not language:
        return None
    return _LANG_SCRIPT.get(language.split("-", 1)[0].lower())


def _compression_ratio(text: str) -> float:
    """Whisper's repetition heuristic: original bytes / zlib-compressed bytes (>= 1)."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


# A single spoken word essentially never lasts this long; a longer word "end" is a
# glitched timestamp, so we cap it when measuring gaps — otherwise a far-future end
# would mask the real silence that follows it (the join this filter exists to catch).
_MAX_WORD_S = 2.0


def _max_internal_gap(words: list[Word]) -> float:
    """Largest silence (seconds) between consecutive words, clamped >= 0.

    A word's end is capped at ``start + _MAX_WORD_S`` so a glitched far-future end
    cannot hide a genuine internal pause.
    """
    biggest = 0.0
    for prev, nxt in itertools.pairwise(words):
        start = max(prev.start_s, 0.0)
        prev_end = min(max(prev.end_s, start), start + _MAX_WORD_S)
        gap = max(0.0, nxt.start_s - prev_end)
        biggest = max(biggest, gap)
    return biggest


def _mean_confidence(words: list[Word]) -> float:
    """Mean per-word confidence (1.0 when there are no words, i.e. never drops)."""
    if not words:
        return 1.0
    return sum(w.confidence for w in words) / len(words)


def _drop(clip: str, name: str, value: object, threshold: object, text: str) -> DropRecord:
    preview = text if len(text) <= 80 else text[:77] + "..."
    return DropRecord(
        clip=clip, filter=name, value=str(value), threshold=str(threshold), text=preview
    )


def _segment_drop_reason(
    clip: str, seg: DatasetSegment, duration_s: float, config: FilterConfig
) -> DropRecord | None:
    """The first Tier-1 filter this clip fails, or None if it passes."""
    text = seg.text.strip()
    if not text:
        return _drop(clip, "empty_text", 0, ">0 chars", seg.text)

    if duration_s < config.min_duration_s:
        return _drop(clip, "duration", f"{duration_s:.2f}s", f">={config.min_duration_s}s", text)
    if duration_s > config.max_duration_s:
        return _drop(clip, "duration", f"{duration_s:.2f}s", f"<={config.max_duration_s}s", text)

    gap = _max_internal_gap(seg.words)
    if gap > config.max_internal_gap_s:
        return _drop(
            clip, "internal_silence", f"{gap:.2f}s", f"<={config.max_internal_gap_s}s", text
        )

    target_script = _target_script(config.target_language)
    if config.require_target_script and target_script is not None:
        dominant = _dominant_script(text)
        # Only drop when a single WRONG script is a strict majority of the letters;
        # a mixed/ambiguous clip is kept (and the verdict is order-independent).
        if dominant is not None and dominant[0] != target_script and dominant[1] > 0.5:
            return _drop(clip, "non_target_script", dominant[0], target_script, text)

    # Characters-per-second only makes sense for a known, space-delimited script —
    # skip it for logographic/space-free scripts and unmapped target languages (their
    # conventions differ, so the Latin-tuned bounds would mis-fire).
    if target_script is not None and target_script not in _UNSPACED_SCRIPTS and duration_s > 0:
        cps = len(text) / duration_s
        if cps < config.min_chars_per_s:
            return _drop(clip, "char_rate", f"{cps:.1f}/s", f">={config.min_chars_per_s}/s", text)
        if cps > config.max_chars_per_s:
            return _drop(clip, "char_rate", f"{cps:.1f}/s", f"<={config.max_chars_per_s}/s", text)

    ratio = _compression_ratio(text)
    if ratio > config.max_compression_ratio:
        return _drop(
            clip, "compression_ratio", f"{ratio:.2f}", f"<={config.max_compression_ratio}", text
        )

    confidence = _mean_confidence(seg.words)
    if confidence < config.min_avg_confidence:
        return _drop(
            clip, "confidence", f"{confidence:.2f}", f">={config.min_avg_confidence}", text
        )

    return None


def filter_segments(
    candidates: list[tuple[str, DatasetSegment, float]],
    config: FilterConfig,
) -> tuple[list[tuple[str, DatasetSegment, float]], list[DropRecord]]:
    """Apply Tier-1 filters; return ``(kept, drops)``. No-op when ``config.enabled`` is False."""
    if not config.enabled:
        return candidates, []
    kept: list[tuple[str, DatasetSegment, float]] = []
    drops: list[DropRecord] = []
    for clip, seg, duration_s in candidates:
        reason = _segment_drop_reason(clip, seg, duration_s, config)
        if reason is None:
            kept.append((clip, seg, duration_s))
        else:
            drops.append(reason)
    return kept, drops


# --- Tier-2 (opt-in): hard-clipping detection -----------------------------

_CLIP_FRACTION = 0.005  # drop if > 0.5% of samples sit at full scale
_FULL_SCALE_16 = 32767


def detect_clipping_drops(
    clips: list[DatasetClip], out_dir: Path, config: FilterConfig
) -> tuple[list[DatasetClip], list[DropRecord]]:
    """Drop clips with hard digital clipping (reads each WAV; opt-in). Returns ``(kept, drops)``."""
    if not config.detect_clipping:
        return clips, []
    import wave

    import numpy as np

    kept: list[DatasetClip] = []
    drops: list[DropRecord] = []
    for clip in clips:
        path = out_dir / clip.audio_path
        try:
            with wave.open(str(path), "rb") as wav:
                if wav.getsampwidth() != 2:  # only 16-bit PCM is analyzed
                    kept.append(clip)
                    continue
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, OSError):
            kept.append(clip)  # unreadable — leave it rather than guess
            continue
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            kept.append(clip)
            continue
        # Widen to int32 before abs: np.abs(int16 -32768) overflows back to -32768,
        # which would hide negative-rail (asymmetric) clipping.
        clipped = float((np.abs(samples.astype(np.int32)) >= _FULL_SCALE_16).mean())
        if clipped > _CLIP_FRACTION:
            drops.append(
                _drop(
                    clip.id,
                    "clipping",
                    f"{clipped * 100:.2f}%",
                    f"<={_CLIP_FRACTION * 100}%",
                    clip.text,
                )
            )
        else:
            kept.append(clip)
    return kept, drops
