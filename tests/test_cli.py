"""Tests for the CLI: version/help smoke tests plus the ingest command."""

import json
import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hearsay.cli as cli
from hearsay.captions import CaptionResult, normalize_snippets
from hearsay.cli import app
from hearsay.errors import NoCaptionsError
from hearsay.models import Document
from hearsay.pipeline import build_document
from hearsay.youtube import parse_metadata

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


# --- Smoke tests ----------------------------------------------------------


def test_version_flag_prints_real_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert re.fullmatch(r"hearsay \d+\.\d+\.\d+\S*\n?", result.output)
    assert "0.0.0.dev0" not in result.output


def test_help_shows_pitch() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM-ready" in result.output


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "Usage:" in result.output


def test_console_script_points_at_cli_app() -> None:
    (script,) = entry_points(group="console_scripts", name="hearsay")
    assert script.value == "hearsay.cli:app"
    assert script.load() is app


# --- Ingest command (offline via a patched pipeline) ----------------------


def _fixture_document(video_id: str) -> Document:
    meta = parse_metadata(json.loads((FIXTURES / f"{video_id}.meta.json").read_text()), "url")
    data = json.loads((FIXTURES / f"{video_id}.transcript.json").read_text())
    captions = CaptionResult(
        segments=normalize_snippets(data["snippets"]),
        language_code=data["language_code"],
        is_generated=data["is_generated"],
    )
    return build_document(meta, captions, ingested_at="2026-06-13T10:00:00Z")


def test_ingest_writes_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    result = runner.invoke(app, ["https://www.youtube.com/watch?v=rStL7niR7gs"])
    assert result.exit_code == 0, result.output
    out = tmp_path / "rStL7niR7gs.md"
    assert out.exists()
    text = out.read_text()
    assert text.startswith("---\n")
    assert "You Would Be a Terrible Leader" in text


def test_ingest_respects_output_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    dest = tmp_path / "custom.md"
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "--output", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.exists()


def test_ingest_rejects_non_youtube_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["https://example.com/not-a-video"])
    assert result.exit_code == 1
    assert "YouTube video URLs" in result.output
    assert not list(tmp_path.glob("*.md"))


def test_ingest_surfaces_no_captions_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> Document:
        raise NoCaptionsError(
            "This video has no captions in any language: abc",
            hint="Local Whisper transcription lands in Phase 2.",
        )

    monkeypatch.setattr(cli, "ingest_youtube", boom)
    result = runner.invoke(app, ["https://www.youtube.com/watch?v=abcdefghijk"])
    assert result.exit_code == 1
    assert "no captions" in result.output
    assert "Phase 2" in result.output
