"""Command-line interface for hearsay.

A single Typer command for now: ``hearsay <SOURCE> [--output] [--lang]``.
A single command keeps option order natural (``hearsay <url> --output x``);
the ``mcp`` subcommand planned for Phase 4 will convert this into a group.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from hearsay import __version__
from hearsay.errors import HearsayError, InvalidSourceError
from hearsay.pipeline import ingest_youtube
from hearsay.render import render_markdown
from hearsay.youtube import extract_video_id

PITCH = (
    "crawl4ai for video & audio — turn any YouTube video, podcast episode, "
    "or local recording into clean, timestamped, LLM-ready markdown."
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"hearsay {__version__}")
        raise typer.Exit()


@app.command(help=PITCH)
def main(
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE",
            help="A YouTube video URL to ingest into markdown.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Write markdown here. Defaults to ./<video-id>.md",
            show_default=False,
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--lang", help="Preferred caption language code."),
    ] = "en",
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
    """Ingest SOURCE into timestamped, LLM-ready markdown."""
    try:
        video_id = extract_video_id(source)
        if video_id is None:
            raise InvalidSourceError(
                f"hearsay does not know how to ingest this yet: {source}",
                hint=(
                    "Right now hearsay accepts YouTube video URLs "
                    "(https://www.youtube.com/watch?v=...). Local files and "
                    "podcast feeds arrive in later phases."
                ),
            )
        with console.status(f"[bold]Fetching captions for {video_id}…", spinner="dots"):
            document = ingest_youtube(source, language=language)
        destination = output if output is not None else Path(f"./{video_id}.md")
        destination.write_text(render_markdown(document), encoding="utf-8")
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow] No file was written.")
        raise typer.Exit(code=130) from None
    except HearsayError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc

    paragraphs = sum(len(section.paragraphs) for section in document.sections)
    console.print(
        Panel.fit(
            f"[bold green]✓[/bold green] {document.meta.title}\n"
            f"[dim]{len(document.sections)} sections · {paragraphs} paragraphs · "
            f"method: {document.method}[/dim]\n"
            f"→ [bold]{destination}[/bold]",
            title="hearsay",
            border_style="green",
        )
    )


def _print_error(exc: HearsayError) -> None:
    """Render a hearsay error as message + actionable hint (no traceback)."""
    body = f"[red]{exc.message}[/red]"
    if exc.hint:
        body += f"\n\n[bold]Try:[/bold] {exc.hint}"
    err_console.print(Panel.fit(body, title="hearsay error", border_style="red"))
