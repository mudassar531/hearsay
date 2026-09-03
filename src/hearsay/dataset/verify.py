"""Verify a built dataset: does the audio match the text, and would a trainer load it?

``hearsay dataset`` writes files, and files are easy. This answers the question a
training run would otherwise answer expensively and late: is clip *N*'s audio actually
clip *N*'s text, is the text in the language's real script, are the clips cut on
silence rather than through a word, and does the tree hold together. It is the sweep
behind ``docs/language-verification.md`` — the one that found five silently-wrong
outputs in 0.7.0 — run by a command instead of by hand, so every dataset ships with
its own evidence in ``verification.md`` and ``verification.json``.

**Pairing** is measured by re-transcribing a sample of clips and diffing each against
its own row — and, as a control, against a *different* row. A self-similarity alone
proves nothing in a language the model transcribes badly (Bengali scored 0.61 with a
perfect pairing); the *gap* between self and control is the pairing signal, and the
self score on its own is the accuracy signal. Nothing here trusts the build report:
every number is measured on the produced files.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import statistics
import unicodedata
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field

from hearsay.dataset.filters import _dominant_script, _target_script
from hearsay.errors import InvalidSourceError
from hearsay.timefmt import format_timestamp
from hearsay.transcribe import DEFAULT_MODEL, UNKNOWN_LANGUAGE, TranscriptionResult

Transcriber = Callable[..., TranscriptionResult]

# --- Thresholds. Measured, not guessed: docs/language-verification.md ---------------
# The pairing gap ran +0.37 (Bengali) to +0.82 (CJK) across ten languages; a broken
# pairing shows no gap at all.
_MIN_PAIRING_GAP = 0.25
# Self-similarity of a correct pairing through a model that reads the language: the
# eight well-served languages scored 0.91-0.999; Swahili at ~39% word error scored 0.94
# only because the same wrong words come back twice, so it sits with "marginal" below.
_GOOD_ACCURACY = 0.90
_MARGINAL_ACCURACY = 0.75
_GOOD_SCRIPT_SHARE = 0.95
_MARGINAL_SCRIPT_SHARE = 0.80
# ponytail: a clip whose first/last 25 ms carry energy within 6 dB of its body was cut
# through speech, not on silence. A heuristic with a known ceiling — a music bed makes
# every edge "hot" and the check says so instead of reporting.
_EDGE_MS = 25
_HOT_EDGE_RATIO = 0.5
_GOOD_HOT_EDGES = 0.10  # above: spot-check (marginal)
_BAD_HOT_EDGES = 0.50  # above: most cuts land in speech — the alignment is broken (fatal)
_DURATION_TOLERANCE_S = 0.05

TRAINABLE, MARGINAL, NOT_TRAINABLE = "trainable", "marginal", "not trainable"
EXIT_CODES = {TRAINABLE: 0, MARGINAL: 1, NOT_TRAINABLE: 2}


class ClipRow(NamedTuple):
    """One indexed clip: its id, audio path relative to the root, text, and listed duration."""

    id: str
    path: str
    text: str
    duration: float | None


class Check(BaseModel):
    """One structural check: what was checked, whether it held, and the measured detail."""

    name: str
    ok: bool
    detail: str
    fatal: bool = True  # a failed non-fatal check caps the verdict at "marginal"


class PairingSample(BaseModel):
    """One re-transcribed clip scored against its own text and against another clip's."""

    clip: str
    self_score: float
    control_score: float
    hypothesis: str


class Verification(BaseModel):
    """Everything ``hearsay verify`` measured, plus the verdict it reached and why."""

    root: str
    verified_at: str
    clips: int
    total_duration_s: float
    language: str | None
    model: str
    index: str  # which index file the clips were read from
    checks: list[Check] = Field(default_factory=list)
    pairing: list[PairingSample] = Field(default_factory=list)
    pairing_mean: float | None = None
    pairing_median: float | None = None
    control_mean: float | None = None
    pairing_gap: float | None = None
    script_share: float | None = None  # clips whose text is in the language's script
    hot_edges: float | None = None  # clips whose edges carry speech-level energy
    verdict: str = NOT_TRAINABLE
    reasons: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


# --- Reading the dataset -------------------------------------------------------------


def load_rows(root: Path) -> tuple[list[ClipRow], str]:
    """Read the clip index hearsay (or a compatible tool) wrote; returns ``(rows, index name)``.

    Prefers the NeMo manifest (it carries durations), then LJSpeech ``metadata.csv``,
    then a HuggingFace ``metadata.jsonl``.
    """
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        rows = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            path = str(obj["audio_filepath"])
            duration = obj.get("duration")
            rows.append(
                ClipRow(
                    Path(path).stem,
                    path,
                    str(obj.get("text", "")),
                    float(duration) if duration is not None else None,
                )
            )
        return rows, manifest.name
    ljspeech = root / "metadata.csv"
    if ljspeech.exists():
        rows = []
        for line in ljspeech.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            clip_id, _, rest = line.partition("|")
            text = rest.split("|")[-1] if rest else ""
            rows.append(ClipRow(clip_id, f"wavs/{clip_id}.wav", text, None))
        return rows, ljspeech.name
    hf = root / "metadata.jsonl"
    if hf.exists():
        rows = []
        for line in hf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            path = str(obj["file_name"])
            rows.append(ClipRow(Path(path).stem, path, str(obj.get("transcription", "")), None))
        return rows, hf.name
    raise InvalidSourceError(
        f"No dataset index found in {root}.",
        hint="Expected manifest.jsonl, metadata.csv or metadata.jsonl — the folder that "
        "`hearsay dataset --out` wrote.",
    )


_CARD_LANGUAGE = re.compile(r"^language:\s*\n-\s*\"?([A-Za-z-]+)\"?", re.MULTILINE)
_CARD_METHOD = re.compile(r"\*\*transcription_method:\*\*\s*(\S+)")


def card_language(root: Path) -> str | None:
    """The language the dataset card declares, or None (also for ``und``)."""
    card = root / "dataset_card.md"
    if not card.exists():
        return None
    match = _CARD_LANGUAGE.search(card.read_text(encoding="utf-8"))
    if not match or match.group(1) == UNKNOWN_LANGUAGE:
        return None
    return match.group(1)


def card_model(root: Path) -> str | None:
    """The transcription model the card records, mapped back to a ``--model`` value."""
    card = root / "dataset_card.md"
    if not card.exists():
        return None
    match = _CARD_METHOD.search(card.read_text(encoding="utf-8"))
    if not match:
        return None
    method = match.group(1)
    if method.startswith("whisper-"):
        return method.removeprefix("whisper-")
    if method.startswith("parakeet"):
        return "parakeet"
    return None


# --- Structure -------------------------------------------------------------------------

_CLIP_NAME = re.compile(r"^.+_\d{4}\.wav$")


class _WavInfo(NamedTuple):
    channels: int
    width: int
    rate: int
    duration: float


def _wav_info(path: Path) -> _WavInfo | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return _WavInfo(
                handle.getnchannels(), handle.getsampwidth(), rate, frames / rate if rate else 0.0
            )
    except (wave.Error, OSError, EOFError):
        return None


def check_structure(root: Path, rows: list[ClipRow]) -> tuple[list[Check], float]:
    """Structural checks over every row; returns ``(checks, total audio seconds)``."""
    checks: list[Check] = []
    missing = [r.id for r in rows if not (root / r.path).is_file()]
    checks.append(
        Check(
            name="every clip has its audio file",
            ok=not missing,
            detail=f"{len(rows) - len(missing)}/{len(rows)} present"
            + (f"; missing e.g. {missing[0]}" if missing else ""),
        )
    )
    ids = [r.id for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
    checks.append(
        Check(
            name="clip ids are unique",
            ok=not dupes,
            detail="no duplicates" if not dupes else f"{len(dupes)} duplicated, e.g. {dupes[0]}",
        )
    )
    empty = [r.id for r in rows if not r.text.strip()]
    control = [r.id for r in rows if any(ord(c) < 32 and c not in "\t" for c in r.text)]
    checks.append(
        Check(
            name="every clip has text",
            ok=not empty and not control,
            detail=f"{len(empty)} empty, {len(control)} with control characters",
        )
    )

    infos = {r.id: _wav_info(root / r.path) for r in rows if (root / r.path).is_file()}
    unreadable = [i for i, info in infos.items() if info is None]
    readable = {i: info for i, info in infos.items() if info is not None}
    rates = {info.rate for info in readable.values()}
    bad_shape = [i for i, info in readable.items() if info.channels != 1 or info.width != 2]
    checks.append(
        Check(
            name="audio is mono 16-bit PCM at one sample rate",
            ok=not unreadable and not bad_shape and len(rates) <= 1,
            detail=(
                f"{len(readable)} readable, {len(unreadable)} unreadable, "
                f"{len(bad_shape)} not mono/16-bit, rates: "
                + (", ".join(f"{r} Hz" for r in sorted(rates)) or "none")
            ),
        )
    )
    listed = [
        (r.id, r.duration, readable[r.id].duration)
        for r in rows
        if r.duration is not None and r.id in readable
    ]
    off = [cid for cid, want, got in listed if abs(got - want) > _DURATION_TOLERANCE_S]
    if listed:
        checks.append(
            Check(
                name="listed durations match the audio",
                ok=not off,
                detail=f"{len(listed) - len(off)}/{len(listed)} within {_DURATION_TOLERANCE_S}s"
                + (f"; off e.g. {off[0]}" if off else ""),
            )
        )

    wavs_dir = root / "wavs"
    referenced = {r.path for r in rows}
    orphans = (
        sorted(
            p.name
            for p in wavs_dir.glob("*.wav")
            if f"wavs/{p.name}" not in referenced and _CLIP_NAME.match(p.name)
        )
        if wavs_dir.is_dir()
        else []
    )
    checks.append(
        Check(
            name="no unreferenced clips in wavs/",
            ok=not orphans,
            detail="none" if not orphans else f"{len(orphans)} orphan(s), e.g. {orphans[0]}",
            fatal=False,
        )
    )

    csv_path = root / "metadata.csv"
    if csv_path.exists() and (root / "manifest.jsonl").exists():
        by_id = {r.id: r.text for r in rows}
        bad_rows = 0
        mismatched = 0
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) != 3:
                bad_rows += 1
                continue
            if fields[0] in by_id and _clean(fields[2]) != _clean(by_id[fields[0]]):
                mismatched += 1
        checks.append(
            Check(
                name="metadata.csv agrees with the manifest",
                ok=not bad_rows and not mismatched,
                detail=f"{bad_rows} malformed row(s), {mismatched} text mismatch(es)",
                fatal=False,
            )
        )
    total = sum(info.duration for info in readable.values())
    return checks, total


def _clean(text: str) -> str:
    return " ".join(text.replace("|", " ").split())


# --- Pairing ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Casefold, drop punctuation and symbols, collapse whitespace — what ASR cannot hear."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S")))
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]; works the same for spaced and unspaced scripts."""
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def measure_pairing(
    root: Path,
    rows: list[ClipRow],
    *,
    sample: int,
    seed: int,
    model_size: str,
    language: str | None,
    transcriber: Transcriber,
    local_files_only: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[PairingSample]:
    """Re-transcribe ``sample`` random clips; score each against its own text and another's.

    The control text comes from a different sampled clip (the next one, cyclically), so
    a dataset whose rows are shifted by one — the classic off-by-one pairing bug — scores
    *higher* against the control than against itself and fails loudly.
    """
    present = [r for r in rows if (root / r.path).is_file()]
    if len(present) < 2 or sample <= 0:
        return []
    picked = random.Random(seed).sample(present, min(sample, len(present)))
    out: list[PairingSample] = []
    for index, row in enumerate(picked):
        if on_progress is not None:
            on_progress(index, len(picked))
        result = transcriber(
            root / row.path,
            model_size=model_size,
            language=language,
            vad_filter=False,  # a one-sentence clip has nothing worth filtering
            local_files_only=local_files_only,
        )
        hypothesis = " ".join(seg.text for seg in result.segments).strip()
        control = picked[(index + 1) % len(picked)].text
        out.append(
            PairingSample(
                clip=row.id,
                self_score=round(similarity(hypothesis, row.text), 3),
                control_score=round(similarity(hypothesis, control), 3),
                hypothesis=hypothesis[:120],
            )
        )
    if on_progress is not None:
        on_progress(len(picked), len(picked))
    return out


# --- Script and clip edges ---------------------------------------------------------------


def measure_script_share(rows: list[ClipRow], language: str | None) -> float | None:
    """Fraction of clips whose text is (mostly) in the language's script; None if unknown."""
    target = _target_script(language)
    if target is None:
        return None
    judged = 0
    right = 0
    for row in rows:
        dominant = _dominant_script(row.text)
        if dominant is None:
            continue
        judged += 1
        right += dominant[0] == target
    return right / judged if judged else None


def _edge_ratio(path: Path) -> float | None:
    """Edge RMS over body RMS for one WAV: ~0 for a cut on silence, ~1 for a cut through speech."""
    import numpy as np

    info = _wav_info(path)
    if info is None or info.width != 2:
        return None
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if info.channels > 1:
        samples = samples.reshape(-1, info.channels).mean(axis=1)
    edge = int(info.rate * _EDGE_MS / 1000)
    if samples.size < 4 * edge:
        return None

    def rms(chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0

    body = rms(samples[edge:-edge])
    if body < 1e-3:  # a silent clip has no speech to cut through
        return None
    return max(rms(samples[:edge]), rms(samples[-edge:])) / body


def measure_hot_edges(root: Path, rows: list[ClipRow]) -> float | None:
    """Fraction of clips whose first or last 25 ms carry speech-level energy."""
    ratios = [_edge_ratio(root / r.path) for r in rows if (root / r.path).is_file()]
    judged = [r for r in ratios if r is not None]
    if not judged:
        return None
    return sum(1 for r in judged if r > _HOT_EDGE_RATIO) / len(judged)


# --- Verdict -----------------------------------------------------------------------------


def _judge(v: Verification) -> None:
    """Set ``verdict`` and ``reasons`` from what was measured. Every reason is a sentence."""
    fatal = [c for c in v.checks if not c.ok and c.fatal]
    soft = [c for c in v.checks if not c.ok and not c.fatal]
    marginal = False
    reasons: list[str] = []
    for check in fatal:
        reasons.append(f"structure: {check.name} — {check.detail}.")
    for check in soft:
        reasons.append(f"tidy up: {check.name} — {check.detail}.")
        marginal = True

    if v.pairing_gap is None:
        reasons.append("pairing was not measured (no clips re-transcribed).")
        marginal = True
    elif v.pairing_gap < _MIN_PAIRING_GAP:
        reasons.append(
            f"audio and text do not belong together: re-transcribed clips scored "
            f"{v.pairing_mean:.2f} against their own text and {v.control_mean:.2f} against "
            f"another clip's (gap {v.pairing_gap:+.2f}, need {_MIN_PAIRING_GAP:+.2f})."
        )
        fatal.append(Check(name="pairing", ok=False, detail=""))
    if (
        v.pairing_mean is not None
        and v.pairing_gap is not None
        and v.pairing_gap >= _MIN_PAIRING_GAP
    ):
        if v.pairing_mean < _MARGINAL_ACCURACY:
            reasons.append(
                f"the pairing holds but the model barely reads this language: self-similarity "
                f"{v.pairing_mean:.2f} (below {_MARGINAL_ACCURACY}). The text is largely wrong; "
                "build with a fine-tune for this language (--model)."
            )
            fatal.append(Check(name="accuracy", ok=False, detail=""))
        elif v.pairing_mean < _GOOD_ACCURACY:
            reasons.append(
                f"transcripts are approximate: self-similarity {v.pairing_mean:.2f} "
                f"(below {_GOOD_ACCURACY}) — usable as weak STT supervision, not for TTS."
            )
            marginal = True

    if v.script_share is not None:
        if v.script_share < _MARGINAL_SCRIPT_SHARE:
            reasons.append(
                f"only {v.script_share:.0%} of clips are written in the script of "
                f"'{v.language}' — the model transliterated or switched language."
            )
            fatal.append(Check(name="script", ok=False, detail=""))
        elif v.script_share < _GOOD_SCRIPT_SHARE:
            reasons.append(
                f"{1 - v.script_share:.0%} of clips are not in the script of '{v.language}'."
            )
            marginal = True

    if v.hot_edges is not None:
        if v.hot_edges > _BAD_HOT_EDGES:
            reasons.append(
                f"{v.hot_edges:.0%} of clips start or end inside speech (edge energy within 6 dB "
                "of the body) — cuts landed through words, or a music bed runs under the voice."
            )
            fatal.append(Check(name="edges", ok=False, detail=""))
        elif v.hot_edges > _GOOD_HOT_EDGES:
            reasons.append(
                f"{v.hot_edges:.0%} of clips start or end inside speech; spot-check them."
            )
            marginal = True

    v.reasons = reasons
    v.verdict = NOT_TRAINABLE if fatal else MARGINAL if marginal else TRAINABLE


# --- Report ------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_report(root: Path, v: Verification) -> list[str]:
    """Write ``verification.md`` and ``verification.json`` into ``root``; return their names."""
    (root / "verification.json").write_text(v.model_dump_json(indent=2) + "\n", encoding="utf-8")
    mark = {TRAINABLE: "✅", MARGINAL: "⚠️", NOT_TRAINABLE: "❌"}[v.verdict]
    checks = "\n".join(
        f"| {'✅' if c.ok else '❌' if c.fatal else '⚠️'} | {c.name} | {c.detail} |"
        for c in v.checks
    )
    samples = "\n".join(
        f"| `{s.clip}` | {s.self_score:.2f} | {s.control_score:.2f} | {s.hypothesis[:60]} |"
        for s in v.pairing
    )
    reasons = "\n".join(f"- {r}" for r in v.reasons) or "- Nothing to flag."
    measures = [
        (
            "pairing gap",
            _num(v.pairing_gap),
            "self-similarity minus control; a correct pairing shows +0.3 or more, a broken one ~0",
        ),
        (
            "self-similarity mean / median",
            f"{_num(v.pairing_mean)} / {_num(v.pairing_median)}",
            "how well the model reads this language on these clips; 0.90 and up is accurate",
        ),
        (
            "control mean",
            _num(v.control_mean),
            "the same clips scored against *another* clip's text",
        ),
        ("script", _pct(v.script_share), f"clips written in the script of `{v.language or '?'}`"),
        (
            "hot edges",
            _pct(v.hot_edges),
            "clips whose first or last 25 ms carry speech-level energy (a cut through a word)",
        ),
    ]
    measure_rows = "\n".join(f"| {m} | {val} | {meaning} |" for m, val, meaning in measures)
    md = f"""# Verification — {mark} {v.verdict}

Measured by `hearsay verify` on {v.verified_at.split("T")[0]}, on the files in
`{Path(v.root).name}/` ({v.clips} clips, {format_timestamp(v.total_duration_s)} of audio,
index `{v.index}`). Language `{v.language or "unknown"}`, re-transcribed with `{v.model}`.

## Verdict

{reasons}

## Measurements

| measure | value | what it means |
| --- | --- | --- |
{measure_rows}

## Structure

| | check | detail |
| --- | --- | --- |
{checks}

## Pairing sample

Each clip was re-transcribed and diffed against its own text (self) and against the
next sampled clip's text (control). Character-level, case- and punctuation-insensitive.

| clip | self | control | heard |
| --- | --- | --- | --- |
{samples or "| — | — | — | pairing was not measured |"}

## Method

This report trusts nothing the build reported: every number above was measured on the
produced files. Pairing isolates what hearsay controls (does clip *N*'s audio go with
clip *N*'s text) from ASR accuracy, which it does not; the gap is the pairing signal and
the self score is the accuracy signal. The edge check is a heuristic — a music bed under
the voice makes every edge "hot" — so read it with the source in mind. Exit codes:
0 {TRAINABLE}, 1 {MARGINAL}, 2 {NOT_TRAINABLE}.
"""
    (root / "verification.md").write_text(md, encoding="utf-8")
    return ["verification.md", "verification.json"]


# --- Entry point -------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def verify_dataset(
    root: Path,
    *,
    sample: int = 8,
    seed: int = 0,
    model_size: str | None = None,
    language: str | None = None,
    transcriber: Transcriber | None = None,
    local_files_only: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    now: Callable[[], str] = _utc_now_iso,
) -> Verification:
    """Measure a dataset folder and write ``verification.md`` + ``.json`` into it.

    ``model_size`` defaults to the model the dataset card records (the pairing check is
    most telling with the same model), else ``auto``; ``language`` defaults to the card's.
    ``transcriber`` is injectable for offline tests.
    """
    from hearsay import transcribe as _transcribe  # looked up at call time: monkeypatchable

    root = Path(root).expanduser()
    if not root.is_dir():
        raise InvalidSourceError(
            f"Not a directory: {root}", hint="Pass the folder `hearsay dataset --out` wrote."
        )
    rows, index = load_rows(root)
    if not rows:
        raise InvalidSourceError(
            f"The index {index} in {root} lists no clips.", hint="Build the dataset first."
        )
    language = language or card_language(root)
    model = model_size or card_model(root) or DEFAULT_MODEL
    checks, total = check_structure(root, rows)
    pairing = measure_pairing(
        root,
        rows,
        sample=sample,
        seed=seed,
        model_size=model,
        language=language,
        transcriber=transcriber or _transcribe.transcribe_audio,
        local_files_only=local_files_only,
        on_progress=on_progress,
    )
    v = Verification(
        root=str(root),
        verified_at=now(),
        clips=len(rows),
        total_duration_s=total,
        language=language,
        model=model,
        index=index,
        checks=checks,
        pairing=pairing,
        script_share=measure_script_share(rows, language),
        hot_edges=measure_hot_edges(root, rows),
    )
    if pairing:
        selfs = [s.self_score for s in pairing]
        controls = [s.control_score for s in pairing]
        v.pairing_mean = round(statistics.fmean(selfs), 3)
        v.pairing_median = round(statistics.median(selfs), 3)
        v.control_mean = round(statistics.fmean(controls), 3)
        v.pairing_gap = round(v.pairing_mean - v.control_mean, 3)
    _judge(v)
    v.files = write_report(root, v)
    return v
