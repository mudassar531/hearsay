"""Tests for the ffmpeg audio layer: loudness normalization, probes, filter check."""

import subprocess
from pathlib import Path

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
