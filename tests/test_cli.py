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
from hearsay.models import Document, Paragraph, Section, SourceMetadata
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


def test_ingest_bad_output_path_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing parent directory must produce a hint, not a traceback.
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    dest = tmp_path / "missing_dir" / "out.md"
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "-o", str(dest)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "directory" in result.output.lower()


def test_ingest_output_is_directory_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "directory" in result.output.lower()


# --- Whisper paths (offline; ingestion functions are patched) -------------


def _whisper_doc(title: str = "clip") -> Document:
    meta = SourceMetadata(
        title=title, source="x", channel="Local file", duration_s=10.0, video_id=title
    )
    return Document(
        meta=meta,
        method="whisper-tiny",
        language="en",
        ingested_at="2026-06-13T10:00:00Z",
        sections=[
            Section(
                title="[00:00:00 – 00:00:10]",
                start_s=0,
                paragraphs=[Paragraph(text="hello world from whisper", start_s=0, end_s=10)],
            )
        ],
    )


def test_local_file_is_transcribed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "talk.wav"
    clip.write_bytes(b"x")
    captured: dict[str, object] = {}

    def fake_ingest_file(path, **kwargs):
        captured["path"] = path
        captured["model_size"] = kwargs.get("model_size")
        return _whisper_doc("talk")

    monkeypatch.setattr(cli, "ingest_file", fake_ingest_file)
    out = tmp_path / "talk.md"
    result = runner.invoke(app, [str(clip), "--model", "tiny", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert captured["model_size"] == "tiny"
    assert out.read_text().startswith("---\n")
    assert "whisper-tiny" in out.read_text()


def test_transcribe_flag_forces_whisper_on_youtube(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"captions": False, "transcribe": False}

    def fake_captions(*a, **k):
        called["captions"] = True
        return _fixture_document("rStL7niR7gs")

    def fake_transcribe(*a, **k):
        called["transcribe"] = True
        return _whisper_doc("forced")

    monkeypatch.setattr(cli, "ingest_youtube", fake_captions)
    monkeypatch.setattr(cli, "ingest_youtube_transcribe", fake_transcribe)
    result = runner.invoke(
        app, ["https://youtu.be/rStL7niR7gs", "--transcribe", "-o", str(tmp_path / "o.md")]
    )
    assert result.exit_code == 0, result.output
    assert called["transcribe"] is True
    assert called["captions"] is False  # --transcribe skips captions entirely


def test_auto_fallback_to_whisper_when_no_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_captions(*a, **k):
        raise NoCaptionsError("This video has no captions in any language: x", hint="...")

    fell_back = {"yes": False}

    def fake_transcribe(*a, **k):
        fell_back["yes"] = True
        return _whisper_doc("fallback")

    monkeypatch.setattr(cli, "ingest_youtube", no_captions)
    monkeypatch.setattr(cli, "ingest_youtube_transcribe", fake_transcribe)
    out = tmp_path / "o.md"
    result = runner.invoke(app, ["https://www.youtube.com/watch?v=abcdefghijk", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert fell_back["yes"] is True
    assert "No captions found" in result.output
    assert out.exists()


def test_unsupported_local_file_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [str(notes)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "unsupported" in result.output.lower()


def test_invalid_model_choice_rejected() -> None:
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "--model", "huge"])
    assert result.exit_code == 2  # Typer rejects the bad enum choice
    assert "huge" in result.output


def test_existing_file_wins_over_youtube_substring_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real file under a "youtu.be" folder must be transcribed, not sent to yt-dlp.
    weird = tmp_path / "youtu.be" / "abcdefghijk.wav"
    weird.parent.mkdir()
    weird.write_bytes(b"x")
    routed = {"file": False, "youtube": False}

    def fake_file(path, **kwargs):
        routed["file"] = True
        return _whisper_doc(path.stem)

    def fake_youtube(*a, **k):
        routed["youtube"] = True
        return _fixture_document("rStL7niR7gs")

    monkeypatch.setattr(cli, "ingest_file", fake_file)
    monkeypatch.setattr(cli, "ingest_youtube", fake_youtube)
    result = runner.invoke(app, [str(weird), "-o", str(tmp_path / "o.md")])
    assert result.exit_code == 0, result.output
    assert routed["file"] is True
    assert routed["youtube"] is False
