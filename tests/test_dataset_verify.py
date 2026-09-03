"""Tests for `hearsay verify` — offline, on synthetic datasets written with the stdlib.

A fake transcriber "hears" each clip as some row's text, so the pairing check can be
driven to a perfect pairing (shift 0) or the classic off-by-one manifest bug (shift 1).
Structural, script and edge checks run on real WAV bytes written with ``wave``.
"""

import json
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import hearsay.transcribe as transcribe
from hearsay.cli import app
from hearsay.dataset import verify as dv
from hearsay.dataset.verify import MARGINAL, NOT_TRAINABLE, TRAINABLE, verify_dataset
from hearsay.errors import InvalidSourceError
from hearsay.models import Segment
from hearsay.transcribe import TranscriptionResult

RATE = 16000
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Sphinx of black quartz, judge my vow.",
    "Two driven jocks help fax my big quiz.",
]


def _tone(seconds: float, *, hot: bool = False) -> np.ndarray:
    """A 220 Hz tone; silent for 100 ms at each edge unless ``hot`` (cut through the sound)."""
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    if not hot:
        pad = int(RATE * 0.1)
        x[:pad] = 0.0
        x[-pad:] = 0.0
    return (x * 32767).astype(np.int16)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.tobytes())


def _dataset(
    root: Path,
    texts: list[str],
    *,
    language: str = "en",
    hot: frozenset[int] | set[int] = frozenset(),
    method: str = "whisper-small",
) -> list[dict[str, Any]]:
    (root / "wavs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts, start=1):
        clip_id = f"src_{index:04d}"
        samples = _tone(1.5, hot=index in hot)
        _write_wav(root / "wavs" / f"{clip_id}.wav", samples)
        rows.append(
            {
                "audio_filepath": f"wavs/{clip_id}.wav",
                "duration": round(len(samples) / RATE, 3),
                "text": text,
                "offset": 0.0,
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (root / "metadata.csv").write_text(
        "".join(f"{Path(r['audio_filepath']).stem}|{r['text']}|{r['text']}\n" for r in rows),
        encoding="utf-8",
    )
    (root / "dataset_card.md").write_text(
        f'---\nlicense: unknown\nlanguage:\n- "{language}"\n---\n\n'
        f"- **transcription_method:** {method}\n",
        encoding="utf-8",
    )
    return rows


def _echo_transcriber(root: Path, rows: list[dict[str, Any]], *, shift: int = 0) -> Callable:
    """A fake ASR that hears clip N as clip N+shift's text; shift 0 is a perfect pairing."""
    ordered = [root / r["audio_filepath"] for r in rows]
    texts = {root / r["audio_filepath"]: r["text"] for r in rows}

    def transcribe_fake(path: Path, *, model_size: str, language: str | None, **kwargs: object):
        index = ordered.index(Path(path))
        heard = texts[ordered[(index + shift) % len(ordered)]]
        return TranscriptionResult(
            segments=[Segment(text=heard, start_s=0.0, end_s=1.0)],
            language=language or "en",
            duration_s=1.5,
            model_size=model_size,
            method=f"whisper-{model_size}",
        )

    return transcribe_fake


def test_a_correct_dataset_is_trainable(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS)
    v = verify_dataset(tmp_path, sample=5, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.verdict == TRAINABLE and v.reasons == []
    assert v.pairing_mean == 1.0 and v.pairing_gap is not None and v.pairing_gap > 0.3
    assert v.script_share == 1.0 and v.hot_edges == 0.0
    assert v.model == "small"  # taken from the card's transcription_method
    assert all(c.ok for c in v.checks)
    assert (tmp_path / "verification.md").exists()
    data = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert data["verdict"] == "trainable" and len(data["pairing"]) == 5
    assert "✅ trainable" in (tmp_path / "verification.md").read_text(encoding="utf-8")


def test_an_off_by_one_pairing_is_caught(tmp_path: Path) -> None:
    """Clip N's audio with clip N+1's text — the classic manifest bug — passes every
    structural check and is only visible by listening. This is the check that matters."""
    rows = _dataset(tmp_path, TEXTS)
    v = verify_dataset(tmp_path, sample=5, transcriber=_echo_transcriber(tmp_path, rows, shift=1))
    assert v.verdict == NOT_TRAINABLE
    assert v.pairing_gap is not None and v.pairing_gap < 0.25
    assert any("do not belong together" in r for r in v.reasons)


def test_a_missing_wav_is_fatal(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS)
    (tmp_path / rows[2]["audio_filepath"]).unlink()
    v = verify_dataset(tmp_path, sample=4, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.verdict == NOT_TRAINABLE
    assert any("every clip has its audio file" in r for r in v.reasons)


def test_cuts_through_speech_are_flagged(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS, hot={1, 2, 3})  # 3 of 5 clips ring right up to the edge
    v = verify_dataset(tmp_path, sample=5, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.hot_edges == pytest.approx(0.6)
    assert v.verdict == NOT_TRAINABLE and any("inside speech" in r for r in v.reasons)
    # a minority of hot edges is a spot-check, not a rejection
    few = tmp_path / "few"
    rows = _dataset(few, TEXTS, hot={1})
    v = verify_dataset(few, sample=5, transcriber=_echo_transcriber(few, rows))
    assert v.hot_edges == pytest.approx(0.2) and v.verdict == MARGINAL


def test_wrong_script_is_flagged(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS, language="ur")  # the card says Urdu; the text is Latin
    v = verify_dataset(tmp_path, sample=5, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.script_share == 0.0 and v.verdict == NOT_TRAINABLE
    assert any("script" in r for r in v.reasons)


def test_ljspeech_only_dataset_is_readable(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS)
    (tmp_path / "manifest.jsonl").unlink()
    v = verify_dataset(tmp_path, sample=3, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.index == "metadata.csv" and v.clips == 5 and v.verdict == TRAINABLE


def test_an_orphan_wav_caps_the_verdict_at_marginal(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS)
    _write_wav(tmp_path / "wavs" / "src_0099.wav", _tone(1.0))
    v = verify_dataset(tmp_path, sample=5, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.verdict == MARGINAL and any("unreferenced" in r for r in v.reasons)


def test_unmeasured_pairing_is_never_trainable(tmp_path: Path) -> None:
    rows = _dataset(tmp_path, TEXTS)
    v = verify_dataset(tmp_path, sample=0, transcriber=_echo_transcriber(tmp_path, rows))
    assert v.verdict == MARGINAL and any("not measured" in r for r in v.reasons)


def test_no_index_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidSourceError, match="No dataset index"):
        verify_dataset(tmp_path, sample=0)


def test_similarity_ignores_what_asr_cannot_hear() -> None:
    assert dv.similarity("Hello, World!", "hello world") == 1.0
    assert dv.similarity("یہ ایک جملہ ہے۔", "یہ ایک جملہ ہے") == 1.0
    assert dv.similarity("这是一个句子。", "这是一个句子") == 1.0
    assert dv.similarity("completely different words", "nothing alike in here") < 0.6


def test_cli_verify_prints_a_verdict_and_exits_by_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _dataset(tmp_path, TEXTS)
    monkeypatch.setattr(transcribe, "transcribe_audio", _echo_transcriber(tmp_path, rows))
    result = CliRunner().invoke(app, ["verify", str(tmp_path), "--sample", "3"])
    assert result.exit_code == 0, result.output
    assert "TRAINABLE" in result.output and "verification.md" in result.output
    monkeypatch.setattr(transcribe, "transcribe_audio", _echo_transcriber(tmp_path, rows, shift=1))
    result = CliRunner().invoke(app, ["verify", str(tmp_path), "--sample", "5"])
    assert result.exit_code == 2, result.output
    assert "NOT TRAINABLE" in result.output


def test_dataset_verify_flag_runs_after_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hearsay.dataset.build as ds_build
    from hearsay.dataset.models import BuildReport

    out = tmp_path / "ds"
    state: dict = {}

    def fake_build(path, *, config, **kwargs):
        state["rows"] = _dataset(config.out_dir, TEXTS)
        monkeypatch.setattr(
            transcribe, "transcribe_audio", _echo_transcriber(config.out_dir, state["rows"])
        )
        return BuildReport(
            out_dir=str(config.out_dir),
            source=str(path),
            clip_count=5,
            total_duration_s=7.5,
            oversized_count=0,
            sample_rate=RATE,
            language="en",
            formats=config.formats,
        )

    monkeypatch.setattr(ds_build, "build_dataset_from_file", fake_build)
    source = tmp_path / "talk.wav"
    _write_wav(source, _tone(2.0))
    result = CliRunner().invoke(app, ["dataset", str(source), "--out", str(out), "--verify"])
    assert result.exit_code == 0, result.output
    assert (out / "verification.md").exists() and "TRAINABLE" in result.output


def _tiny_cached() -> bool:
    try:
        transcribe._load_whisper("tiny", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tiny_cached(), reason="tiny model not cached; verify end-to-end skipped")
def test_verify_end_to_end_with_a_real_model(tmp_path: Path) -> None:
    """Build a real dataset from the fixture clip and verify it with the tiny model."""
    from hearsay.dataset.build import build_dataset
    from hearsay.dataset.models import DatasetConfig, FilterConfig
    from hearsay.models import SourceMetadata, Word

    sample = Path(__file__).parent / "fixtures" / "sample.wav"
    words = [
        Word(text="The", start_s=0.0, end_s=0.4),
        Word(text="quick", start_s=0.4, end_s=0.9),
        Word(text="brown", start_s=0.9, end_s=1.4),
        Word(text="fox.", start_s=1.4, end_s=1.9),
        Word(text="Jumps", start_s=2.0, end_s=2.5),
        Word(text="over", start_s=2.5, end_s=3.0),
        Word(text="the", start_s=3.0, end_s=3.4),
        Word(text="lazy", start_s=3.4, end_s=3.9),
        Word(text="dog.", start_s=3.9, end_s=4.4),
    ]
    meta = SourceMetadata(
        title="sample", source=str(sample), channel="Local file", duration_s=4.71, video_id="sample"
    )
    config = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.5,
        filters=FilterConfig(enabled=False),
    )
    build_dataset(
        sample, words, meta, config=config, language="en", transcription_method="whisper-tiny"
    )
    v = verify_dataset(tmp_path, sample=2, local_files_only=True)
    assert v.model == "tiny" and len(v.pairing) == 2
    assert v.pairing_mean is not None and 0.0 <= v.pairing_mean <= 1.0
    assert all(c.ok for c in v.checks)  # hearsay's own output passes its own structure checks
    assert (tmp_path / "verification.md").exists()
