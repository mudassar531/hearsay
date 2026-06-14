"""Tests for the `hearsay dataset` CLI subcommand — routing, flags, and summary.

Offline: the build_dataset_from_* functions (lazily imported inside the command)
are monkeypatched on hearsay.dataset.build, so no transcription/network happens.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import hearsay.dataset.build as build_mod
from hearsay.cli import app
from hearsay.dataset.models import BuildReport, CombinedReport
from hearsay.errors import AudioExportError

runner = CliRunner()
SAMPLE = Path(__file__).parent / "fixtures" / "sample.wav"


def _report(out: Path) -> BuildReport:
    return BuildReport(
        out_dir=str(out),
        source="s",
        clip_count=3,
        total_duration_s=12.0,
        oversized_count=0,
        sample_rate=22050,
        language="en",
        formats=["ljspeech", "jsonl"],
        dropped_count=1,
        drops_by_reason={"duration": 1},
        warnings=[],
    )


def _combined(out: Path) -> CombinedReport:
    return CombinedReport(
        out_dir=str(out),
        source="s",
        title="T",
        clip_count=5,
        total_duration_s=20.0,
        oversized_count=0,
        dropped_count=0,
        sample_rate=22050,
        language="en",
        formats=["ljspeech", "jsonl"],
        sources=[],
        succeeded=2,
        failed=0,
        warnings=[],
    )


def test_dataset_file_routes_and_passes_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake(path, *, config, **kw):
        captured["path"], captured["config"], captured["kw"] = path, config, kw
        return _report(config.out_dir)

    monkeypatch.setattr(build_mod, "build_dataset_from_file", fake)
    result = runner.invoke(
        app,
        [
            "dataset",
            str(SAMPLE),
            "--out",
            str(tmp_path),
            "--sample-rate",
            "16000",
            "--segment-min",
            "2",
            "--segment-max",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = captured["config"]
    assert cfg.sample_rate == 16000 and cfg.segment_min_s == 2.0 and cfg.segment_max_s == 8.0
    assert captured["kw"] == {"model_size": "auto", "language": None, "vad_filter": True}
    assert "3 clips" in result.output and "dropped 1" in result.output


def test_dataset_youtube_video_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_mod,
        "build_dataset_from_youtube",
        lambda url, *, config, **kw: _report(config.out_dir),
    )
    result = runner.invoke(
        app, ["dataset", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "3 clips" in result.output


def test_dataset_playlist_routes_with_limit_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake(url, *, config, limit=None, on_item=None, resume=True, **kw):
        captured["limit"], captured["resume"] = limit, resume
        return _combined(config.out_dir)

    monkeypatch.setattr(build_mod, "build_dataset_from_playlist", fake)
    result = runner.invoke(
        app,
        [
            "dataset",
            "https://www.youtube.com/playlist?list=PLabc123",
            "--out",
            str(tmp_path),
            "--limit",
            "3",
            "--no-resume",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["limit"] == 3 and captured["resume"] is False
    assert "5 clips" in result.output and "sources" in result.output


def test_dataset_feed_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_mod, "build_dataset_from_feed", lambda url, *, config, **kw: _combined(config.out_dir)
    )
    result = runner.invoke(app, ["dataset", "https://example.com/feed.xml", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_dataset_diarize_flags_set_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake(path, *, config, **kw):
        captured["config"] = config
        return _report(config.out_dir)

    monkeypatch.setattr(build_mod, "build_dataset_from_file", fake)
    runner.invoke(app, ["dataset", str(SAMPLE), "--out", str(tmp_path), "--per-speaker"])
    assert captured["config"].diarize.enabled is True
    assert captured["config"].diarize.mode == "per_speaker"


def test_dataset_format_and_no_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake(path, *, config, **kw):
        captured["config"] = config
        return _report(config.out_dir)

    monkeypatch.setattr(build_mod, "build_dataset_from_file", fake)
    runner.invoke(
        app, ["dataset", str(SAMPLE), "--out", str(tmp_path), "--format", "hf", "--no-filter"]
    )
    assert captured["config"].formats == ["hf"]
    assert captured["config"].filters.enabled is False


def test_dataset_default_formats_are_ljspeech_and_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake(path, *, config, **kw):
        captured["config"] = config
        return _report(config.out_dir)

    monkeypatch.setattr(build_mod, "build_dataset_from_file", fake)
    runner.invoke(app, ["dataset", str(SAMPLE), "--out", str(tmp_path)])
    assert captured["config"].formats == ["ljspeech", "jsonl"]


def test_dataset_warnings_surfaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(path, *, config, **kw):
        report = _report(config.out_dir)
        report.warnings = ["diarization not installed: MIXED speakers ..."]
        return report

    monkeypatch.setattr(build_mod, "build_dataset_from_file", fake)
    result = runner.invoke(app, ["dataset", str(SAMPLE), "--out", str(tmp_path)])
    assert "MIXED speakers" in result.output


def test_dataset_error_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(path, *, config, **kw):
        raise AudioExportError("ffmpeg missing", hint="install ffmpeg")

    monkeypatch.setattr(build_mod, "build_dataset_from_file", fake)
    result = runner.invoke(app, ["dataset", str(SAMPLE), "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ffmpeg missing" in result.output
