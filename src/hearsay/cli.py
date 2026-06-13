"""Command-line interface for hearsay.

A single Typer command. SOURCE may be a YouTube video URL, a YouTube playlist
URL, a podcast RSS feed, or a local audio/video file. Single sources write one
markdown file; batch sources (playlists/feeds) list their items, or ingest a
selection into an output directory. A single command keeps option order natural;
the ``mcp`` subcommand planned for Phase 4 will convert this into a group.
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
from rich.table import Table

from hearsay import __version__
from hearsay.batch import BatchItem, run_batch, select, slugify
from hearsay.errors import HearsayError, OutputWriteError
from hearsay.feeds import Episode, fetch_feed
from hearsay.models import Document, Transcript
from hearsay.pipeline import (
    NoCaptionsError,
    ingest_episode,
    ingest_file,
    ingest_youtube,
    ingest_youtube_transcribe,
)
from hearsay.render import render_markdown
from hearsay.timefmt import format_timestamp
from hearsay.transcribe import DEFAULT_MODEL
from hearsay.youtube import (
    PlaylistEntry,
    extract_playlist_id,
    extract_video_id,
    fetch_playlist,
)

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


_DEFAULT_MODEL = ModelSize(DEFAULT_MODEL)
_DEFAULT_OUTPUT_DIR = Path("./hearsay-out")

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
            help="YouTube video/playlist URL, podcast RSS feed, or local file.",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Output file for a single source. Default ./<id>.md",
            show_default=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Output directory for batch (playlist/feed) ingestion."),
    ] = _DEFAULT_OUTPUT_DIR,
    language: Annotated[
        str | None,
        typer.Option(
            "--lang",
            help="Language code. Captions default to English; transcription auto-detects.",
            show_default=False,
        ),
    ] = None,
    transcribe: Annotated[
        bool,
        typer.Option(
            "--transcribe", help="Force local whisper transcription even when captions exist."
        ),
    ] = False,
    model: Annotated[
        ModelSize,
        typer.Option("--model", help="Whisper model size for transcription."),
    ] = _DEFAULT_MODEL,
    write_json: Annotated[
        bool,
        typer.Option("--json", help="Also write a .json sidecar matching the Transcript schema."),
    ] = False,
    latest: Annotated[
        bool,
        typer.Option("--latest", help="Batch: ingest only the most recent item."),
    ] = False,
    episode: Annotated[
        int | None,
        typer.Option(
            "--episode", help="Batch: ingest only item number N (1-indexed).", show_default=False
        ),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option("--all", help="Batch: ingest every item (cap with --limit)."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", help="Batch: cap the number ingested with --all.", show_default=False
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Ingest SOURCE into timestamped, LLM-ready markdown."""
    try:
        path = Path(source).expanduser()
        if path.is_file():
            _run_single(_ingest_file(path, language, model.value), path.stem, output, write_json)
        elif extract_playlist_id(source) is not None:
            _run_playlist(source, locals())
        elif extract_video_id(source) is not None:
            document = _ingest_youtube_source(source, language, transcribe, model.value)
            _run_single(document, extract_video_id(source) or "video", output, write_json)
        elif source.startswith(("http://", "https://")):
            _run_feed(source, locals())
        else:
            _run_single(_ingest_file(path, language, model.value), path.stem, output, write_json)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except HearsayError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc


# --- Single-source ingestion ---------------------------------------------


def _run_single(
    document: Document, default_name: str, output: Path | None, write_json: bool
) -> None:
    destination = output if output is not None else Path(f"./{default_name}.md")
    written = _write_document(document, destination, write_json=write_json)
    _print_success(document, written)


def _ingest_file(path: Path, language: str | None, model: str) -> Document:
    with _progress(f"Transcribing {path.name} with whisper '{model}'") as cb:
        return ingest_file(path, model_size=model, language=language, on_progress=cb)


def _ingest_youtube_source(
    url: str, language: str | None, transcribe: bool, model: str
) -> Document:
    """Captions path, or whisper — forced by --transcribe or auto on no captions."""
    if transcribe:
        return _transcribe_youtube(url, model, language, forced=True)
    video_id = extract_video_id(url) or url
    try:
        with console.status(f"[bold]Fetching captions for {video_id}…", spinner="dots"):
            return ingest_youtube(url, language=language or "en")
    except NoCaptionsError:
        console.print("[yellow]No captions found.[/yellow] Falling back to local transcription.")
        return _transcribe_youtube(url, model, language, forced=False)


def _transcribe_youtube(url: str, model: str, language: str | None, *, forced: bool) -> Document:
    why = "Transcribing" if forced else "Downloading audio, then transcribing"
    console.print(f"[dim]{why} locally with whisper '{model}'. This can take a few minutes.[/dim]")
    with _progress("Transcribing audio") as cb:
        return ingest_youtube_transcribe(url, model_size=model, language=language, on_progress=cb)


# --- Batch ingestion (playlists & feeds) ----------------------------------


def _run_playlist(url: str, opts: dict) -> None:
    with console.status("[bold]Listing playlist…", spinner="dots"):
        title, entries = fetch_playlist(url)
    items = [
        BatchItem(
            title=e.title,
            slug=e.video_id,
            ingest=_youtube_entry_ingester(
                e, opts["language"], opts["transcribe"], opts["model"].value
            ),
        )
        for e in entries
    ]
    _run_batch_or_list(items, title, "video", opts, has_duration=False, entries=entries)


def _run_feed(url: str, opts: dict) -> None:
    with console.status("[bold]Fetching feed…", spinner="dots"):
        feed = fetch_feed(url)
    items = [
        BatchItem(
            title=ep.title,
            slug=slugify(ep.title, fallback=f"episode-{i}"),
            ingest=_episode_ingester(ep, feed.title, opts["language"], opts["model"].value),
        )
        for i, ep in enumerate(feed.episodes, start=1)
    ]
    _run_batch_or_list(
        items, feed.title, "episode", opts, has_duration=True, episodes=feed.episodes
    )


def _youtube_entry_ingester(
    entry: PlaylistEntry, language: str | None, transcribe: bool, model: str
):
    def _ingest() -> Document:
        if transcribe:
            return ingest_youtube_transcribe(entry.url, model_size=model, language=language)
        try:
            return ingest_youtube(entry.url, language=language or "en")
        except NoCaptionsError:
            return ingest_youtube_transcribe(entry.url, model_size=model, language=language)

    return _ingest


def _episode_ingester(episode: Episode, show: str, language: str | None, model: str):
    def _ingest() -> Document:
        return ingest_episode(episode, show, model_size=model, language=language)

    return _ingest


def _run_batch_or_list(
    items: list[BatchItem],
    source_title: str,
    noun: str,
    opts: dict,
    *,
    has_duration: bool,
    entries: list[PlaylistEntry] | None = None,
    episodes: list[Episode] | None = None,
) -> None:
    selected = select(
        items,
        latest=opts["latest"],
        episode=opts["episode"],
        all_=opts["all_"],
        limit=opts["limit"],
    )
    if not selected:
        _print_listing(items, source_title, noun, episodes=episodes)
        return

    output_dir: Path = opts["output_dir"]
    write_json: bool = opts["write_json"]
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(document: Document, slug: str) -> Path:
        return _write_document(document, output_dir / f"{slug}.md", write_json=write_json)

    def announce(index: int, total: int, item: BatchItem) -> None:
        console.print(f"[bold blue]\\[{index}/{total}][/bold blue] {item.title}")

    results = run_batch(selected, write, on_item=announce)
    _print_summary(results, output_dir)


# --- Output writing -------------------------------------------------------


def _write_document(document: Document, destination: Path, *, write_json: bool) -> Path:
    """Write the markdown (and optional JSON sidecar); return the markdown path."""
    _write_text(destination, render_markdown(document))
    if write_json:
        sidecar = destination.with_suffix(".json")
        transcript = Transcript.from_document(document)
        _write_text(sidecar, transcript.model_dump_json(indent=2) + "\n")
    return destination


def _write_text(destination: Path, text: str) -> None:
    try:
        destination.write_text(text, encoding="utf-8")
    except IsADirectoryError as exc:
        raise OutputWriteError(
            f"The output path is a directory, not a file: {destination}",
            hint="Pass a file path to -o/--output, e.g. -o transcript.md",
        ) from exc
    except OSError as exc:
        raise OutputWriteError(
            f"Could not write to {destination}: {exc.strerror or exc}",
            hint="Check that the parent directory exists and is writable, then try again.",
        ) from exc


# --- Presentation ---------------------------------------------------------


@contextmanager
def _progress(label: str) -> Iterator:
    """A rich progress bar driven by an (processed_s, total_s) callback."""
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


def _print_listing(
    items: list[BatchItem],
    source_title: str,
    noun: str,
    *,
    episodes: list[Episode] | None,
) -> None:
    table = Table(title=source_title, title_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column(noun.capitalize())
    if episodes is not None:
        table.add_column("Duration", justify="right", style="dim")
    for index, item in enumerate(items, start=1):
        if episodes is not None:
            dur = (
                format_timestamp(episodes[index - 1].duration_s)
                if episodes[index - 1].duration_s
                else "—"
            )
            table.add_row(str(index), item.title, dur)
        else:
            table.add_row(str(index), item.title)
    console.print(table)
    console.print(
        f"[dim]{len(items)} {noun}(s). Ingest with "
        "--latest, --episode N, or --all [--limit N].[/dim]"
    )


def _print_summary(results: list, output_dir: Path) -> None:
    table = Table(title="hearsay batch", title_style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Item")
    table.add_column("Output / error", overflow="fold")
    ok_count = 0
    for result in results:
        if result.ok:
            ok_count += 1
            table.add_row("[green]✓[/green]", result.title, str(result.output))
        else:
            table.add_row("[red]✗[/red]", result.title, f"[red]{result.error}[/red]")
    console.print(table)
    failed = len(results) - ok_count
    summary = f"[bold]{ok_count} succeeded[/bold]"
    if failed:
        summary += f", [red]{failed} failed[/red]"
    console.print(f"{summary} · → [bold]{output_dir}/[/bold]")


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
    body = f"[red]{exc.message}[/red]"
    if exc.hint:
        body += f"\n\n[bold]Try:[/bold] {exc.hint}"
    err_console.print(Panel.fit(body, title="hearsay error", border_style="red"))
