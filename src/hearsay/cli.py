"""Command-line interface for hearsay.

A single Typer command: ``hearsay <SOURCE> [--output] [--lang] [--transcribe]
[--model]``. SOURCE is a YouTube URL or a local audio/video file. A single
command keeps option order natural; the ``mcp`` subcommand planned for Phase 4
will convert this into a group.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

from hearsay import __version__
from hearsay.errors import HearsayError, InvalidSourceError, OutputWriteError
from hearsay.models import Document
from hearsay.pipeline import (
    NoCaptionsError,
    ingest_file,
    ingest_youtube,
    ingest_youtube_transcribe,
)
from hearsay.render import render_markdown
from hearsay.transcribe import DEFAULT_MODEL
from hearsay.youtube import extract_video_id

PITCH = (
    "crawl4ai for video & audio — turn any YouTube video, podcast episode, "
    "or local recording into clean, timestamped, LLM-ready markdown."
)


class ModelSize(StrEnum):
    """Whisper model sizes, smallest (fastest) to largest (most accurate)."""

    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large_v3 = "large-v3"


# Module-level singleton so it is not constructed in an argument default (B008).
_DEFAULT_MODEL = ModelSize(DEFAULT_MODEL)

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
            help="A YouTube video URL or a local audio/video file.",
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
        typer.Option("--lang", help="Preferred caption / transcription language code."),
    ] = "en",
    transcribe: Annotated[
        bool,
        typer.Option(
            "--transcribe",
            help="Force local whisper transcription even when captions exist.",
        ),
    ] = False,
    model: Annotated[
        ModelSize,
        typer.Option("--model", help="Whisper model size for transcription."),
    ] = _DEFAULT_MODEL,
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
        document, default_name = _produce_document(
            source, language=language, transcribe=transcribe, model=model.value
        )
        destination = output if output is not None else Path(f"./{default_name}.md")
        _write_output(destination, render_markdown(document))
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow] No file was written.")
        raise typer.Exit(code=130) from None
    except HearsayError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc

    _print_success(document, destination)


def _produce_document(
    source: str, *, language: str, transcribe: bool, model: str
) -> tuple[Document, str]:
    """Route SOURCE to the right ingestion path and return (document, name)."""
    video_id = extract_video_id(source)
    if video_id is not None:
        return _ingest_youtube_source(source, video_id, language, transcribe, model), video_id

    if source.startswith(("http://", "https://")):
        raise InvalidSourceError(
            f"hearsay does not know how to ingest this URL yet: {source}",
            hint=(
                "Right now hearsay accepts YouTube video URLs and local files. "
                "Podcast feeds and playlists arrive in Phase 3."
            ),
        )

    # Treat anything else as a local file path; ingest_file validates it.
    path = Path(source).expanduser()
    lang = None if language == "en" else language
    with _transcription_progress(f"Transcribing {path.name} with whisper '{model}'") as cb:
        document = ingest_file(path, model_size=model, language=lang, on_progress=cb)
    return document, path.stem


def _ingest_youtube_source(
    url: str, video_id: str, language: str, transcribe: bool, model: str
) -> Document:
    """Captions path, or whisper — forced by --transcribe or auto on no captions."""
    lang = None if language == "en" else language
    if transcribe:
        return _transcribe_youtube(url, model, lang, forced=True)
    try:
        with console.status(f"[bold]Fetching captions for {video_id}…", spinner="dots"):
            return ingest_youtube(url, language=language)
    except NoCaptionsError:
        console.print("[yellow]No captions found.[/yellow] Falling back to local transcription.")
        return _transcribe_youtube(url, model, lang, forced=False)


def _transcribe_youtube(url: str, model: str, lang: str | None, *, forced: bool) -> Document:
    """Download a YouTube video's audio and transcribe it locally."""
    why = "Transcribing" if forced else "Downloading audio, then transcribing"
    console.print(f"[dim]{why} locally with whisper '{model}'. This can take a few minutes.[/dim]")
    with _transcription_progress("Transcribing audio") as cb:
        return ingest_youtube_transcribe(url, model_size=model, language=lang, on_progress=cb)


@contextmanager
def _transcription_progress(label: str) -> Iterator:
    """A rich progress bar driven by an (processed_s, total_s) callback.

    Pulses while the model loads / audio downloads (total unknown), then fills
    as whisper reports processed seconds.
    """
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task: TaskID = progress.add_task(label, total=None)

        def on_progress(processed_s: float, total_s: float) -> None:
            if total_s > 0:
                progress.update(task, total=total_s, completed=processed_s)

        yield on_progress


def _write_output(destination: Path, markdown: str) -> None:
    """Write markdown to disk, turning filesystem errors into friendly ones."""
    try:
        destination.write_text(markdown, encoding="utf-8")
    except IsADirectoryError as exc:
        raise OutputWriteError(
            f"The output path is a directory, not a file: {destination}",
            hint="Pass a file path to -o/--output, e.g. -o transcript.md",
        ) from exc
    except OSError as exc:
        raise OutputWriteError(
            f"Could not write to {destination}: {exc.strerror or exc}",
            hint=(
                "Check that the parent directory exists and is writable, "
                "then try again (hearsay does not create missing folders)."
            ),
        ) from exc


def _print_success(document: Document, destination: Path) -> None:
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
