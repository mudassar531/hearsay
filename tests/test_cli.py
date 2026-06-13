"""Smoke tests for the CLI skeleton."""

import re
from importlib.metadata import entry_points

from typer.testing import CliRunner

from hearsay.cli import app

runner = CliRunner()


def test_version_flag_prints_real_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert re.fullmatch(r"hearsay \d+\.\d+\.\d+\S*\n?", result.output)
    # The fallback for a missing install must never reach users.
    assert "0.0.0.dev0" not in result.output


def test_help_shows_pitch() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM-ready" in result.output


def test_bare_invocation_shows_help_as_usage_error() -> None:
    # Bare `hearsay` is a usage error (click convention, exit 2) but must
    # still show help rather than a bare traceback or silence.
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_console_script_points_at_cli_app() -> None:
    (script,) = entry_points(group="console_scripts", name="hearsay")
    assert script.value == "hearsay.cli:app"
    assert script.load() is app
