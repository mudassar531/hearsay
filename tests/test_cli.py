"""Smoke tests for the CLI skeleton."""

from typer.testing import CliRunner

from hearsay import __version__
from hearsay.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"hearsay {__version__}" in result.output


def test_help_shows_pitch() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM-ready markdown" in result.output
