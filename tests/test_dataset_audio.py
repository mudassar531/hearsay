"""Tests for the ffmpeg audio layer: loudness normalization, probes, filter check."""

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from hearsay.dataset.audio import ensure_filter, probe_duration, probe_sample_rate, slice_clip
from hearsay.dataset.build import build_dataset
from hearsay.dataset.models import DatasetConfig, FilterConfig
from hearsay.models import SourceMetadata, Word

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"  # ~4.71s, 16 kHz mono


def _stream(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    sr, ch = proc.stdout.strip().split(",")
    return int(sr), int(ch)


def test_probe_sample_rate() -> None:
    assert probe_sample_rate(SAMPLE) == 16000
    assert probe_sample_rate(Path("/nope/missing.wav")) == 0


def test_ensure_filter_loudnorm_present() -> None:
    ensure_filter("loudnorm")  # ffmpeg ships it — must not raise
    from hearsay.errors import AudioExportError

    with pytest.raises(AudioExportError):
        ensure_filter("definitely_not_a_real_filter_xyz")


def test_slice_normalize_preserves_length_and_format(tmp_path: Path) -> None:
    plain = tmp_path / "plain.wav"
    norm = tmp_path / "norm.wav"
    slice_clip(SAMPLE, 0.0, 2.0, plain, sample_rate=16000, normalize=False)
    slice_clip(SAMPLE, 0.0, 2.0, norm, sample_rate=16000, normalize=True)
    # two-pass loudnorm is length-preserving (single-pass dynamic mode would trim ~tens of ms)
    assert probe_duration(norm) == pytest.approx(2.0, abs=0.03)
    assert _stream(norm) == (16000, 1)  # still mono at the target rate
    assert norm.read_bytes() != plain.read_bytes()  # loudness was actually changed


def _meta() -> SourceMetadata:
    return SourceMetadata(
        title="s", source=str(SAMPLE), channel="c", duration_s=4.71, video_id="sample"
    )


def _words() -> list[Word]:
    return [
        Word(text="Hello", start_s=0.0, end_s=0.5),
        Word(text="there", start_s=0.5, end_s=1.0),
        Word(text="friend", start_s=1.0, end_s=1.6),
    ]


def test_build_warns_when_upsampling(tmp_path: Path) -> None:
    # Fixture is 16 kHz; a 22050 target upsamples -> warn.
    config = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=22050,
        segment_min_s=1.0,
        segment_max_s=2.0,
        filters=FilterConfig(enabled=False),
    )
    report = build_dataset(SAMPLE, _words(), _meta(), config=config)
    assert any("upsampling" in w and "22050" in w for w in report.warnings)


def test_build_no_upsampling_warning_at_source_rate(tmp_path: Path) -> None:
    config = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.0,
        filters=FilterConfig(enabled=False),
    )
    report = build_dataset(SAMPLE, _words(), _meta(), config=config)
    assert not any("upsampling" in w for w in report.warnings)


def test_build_normalize_produces_clips(tmp_path: Path) -> None:
    config = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.0,
        normalize=True,
        filters=FilterConfig(enabled=False),
    )
    report = build_dataset(SAMPLE, _words(), _meta(), config=config)
    assert report.clip_count >= 1
    for clip in report.clips:
        assert _stream(tmp_path / clip.audio_path) == (16000, 1)


def _samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int64)


def test_slice_fade_declicks_edges(tmp_path: Path) -> None:
    plain = tmp_path / "plain.wav"
    faded = tmp_path / "faded.wav"
    slice_clip(SAMPLE, 0.5, 2.5, plain, sample_rate=16000, fade_s=0.0)
    slice_clip(SAMPLE, 0.5, 2.5, faded, sample_rate=16000, fade_s=0.05)
    assert probe_duration(faded) == pytest.approx(2.0, abs=0.03)  # fade preserves length
    ps, fs = _samples(plain), _samples(faded)
    # the cut edges are ramped to ~silence (no hard step -> no click)...
    assert abs(int(fs[0])) <= 3 and abs(int(fs[-1])) <= 3
    # ...while a hard cut leaves a non-trivial step at one edge, and fade only removes energy
    assert max(abs(int(ps[0])), abs(int(ps[-1]))) > 3
    assert int(np.abs(fs).sum()) < int(np.abs(ps).sum())


def test_build_edge_padding_extends_clips_without_changing_count(tmp_path: Path) -> None:
    # One ~1.0s word-extent segment (0.6-1.6s) well inside the source (4.71s).
    words = [
        Word(text="Hello", start_s=0.6, end_s=1.1),
        Word(text="there", start_s=1.1, end_s=1.6),
    ]
    unpadded = build_dataset(
        SAMPLE,
        words,
        _meta(),
        config=DatasetConfig(
            out_dir=tmp_path / "a",
            sample_rate=16000,
            segment_min_s=0.5,
            segment_max_s=3.0,
            fade_s=0.0,
            edge_pad_s=0.0,
            filters=FilterConfig(enabled=False),
        ),
    )
    padded = build_dataset(
        SAMPLE,
        words,
        _meta(),
        config=DatasetConfig(
            out_dir=tmp_path / "b",
            sample_rate=16000,
            segment_min_s=0.5,
            segment_max_s=3.0,
            fade_s=0.0,
            edge_pad_s=0.3,
            filters=FilterConfig(enabled=False),
        ),
    )
    # padding adds audio but never changes which clips survive (filters see the word extent)
    assert unpadded.clip_count == padded.clip_count == 1
    assert padded.clips[0].duration_s > unpadded.clips[0].duration_s + 0.4  # ~0.3s each side


def test_build_edge_padding_filters_on_unpadded_extent(tmp_path: Path) -> None:
    # Word extent 2.8s (1.0-3.8) inside the 4.71s source. With max_duration_s == 3.0 and
    # edge_pad_s == 0.3, the unpadded extent (2.8s) passes the duration filter but the
    # padded window (0.7-4.1 = 3.4s) would FAIL it — so this clip surviving proves the
    # filter sees the word extent, not the padded window. (Wire the padded span into the
    # candidate and this drops to 0.)
    words = [
        Word(text="alpha", start_s=1.0, end_s=2.4),
        Word(text="bravo", start_s=2.4, end_s=3.8),
    ]
    report = build_dataset(
        SAMPLE,
        words,
        _meta(),
        config=DatasetConfig(
            out_dir=tmp_path,
            sample_rate=16000,
            segment_min_s=0.5,
            segment_max_s=3.0,
            fade_s=0.0,
            edge_pad_s=0.3,
            filters=FilterConfig(
                enabled=True,
                min_duration_s=0.5,
                max_duration_s=3.0,
                require_target_script=False,
                min_avg_confidence=0.0,
                min_chars_per_s=0.0,
                max_chars_per_s=1000.0,
                max_compression_ratio=100.0,
            ),
        ),
    )
    assert report.clip_count == 1  # survives: duration filter saw the 2.8s extent
    assert report.clips[0].duration_s == pytest.approx(3.4, abs=0.05)  # padded audio written


def test_slice_fade_skipped_for_very_short_clip(tmp_path: Path) -> None:
    # A clip shorter than two fades cannot be faded without overlapping ramps, so the
    # guard skips the fade entirely — output must equal the unfaded slice byte-for-byte.
    short_faded = tmp_path / "sf.wav"
    short_plain = tmp_path / "sp.wav"
    slice_clip(SAMPLE, 1.0, 1.06, short_faded, sample_rate=16000, fade_s=0.05)
    slice_clip(SAMPLE, 1.0, 1.06, short_plain, sample_rate=16000, fade_s=0.0)
    assert short_faded.read_bytes() == short_plain.read_bytes()


def test_dataset_config_default_boundary_knobs() -> None:
    cfg = DatasetConfig(out_dir=Path("/tmp/x"))
    assert cfg.edge_pad_s == 0.1
    assert cfg.fade_s == 0.01
