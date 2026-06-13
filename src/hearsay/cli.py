"""Command-line interface for hearsay."""

from typing import Annotated

import typer

from hearsay import __version__

PITCH = (
    "crawl4ai for video & audio — turn any YouTube video, podcast episode, "
    "or local recording into clean, timestamped, LLM-ready markdown."
)

app = typer.Typer(add_completion=False)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"hearsay {__version__}")
        raise typer.Exit()


@app.command(help=PITCH, no_args_is_help=True)
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Entry point; ingestion commands land in Phase 1."""
