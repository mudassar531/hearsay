"""Command-line interface for hearsay.

A Typer command group with a default command (``ingest``), so
``hearsay <SOURCE> [options]`` works without typing ``ingest`` while
``hearsay mcp`` runs the MCP server. SOURCE may be a YouTube video/playlist URL,
a podcast RSS feed, or a local audio/video file. Single sources write one
markdown file; batch sources (playlists/feeds) list their items, or ingest a
selection into an output directory.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from typer.core import TyperGroup

from hearsay import __version__
from hearsay.batch import BatchItem, ensure_unique_slugs, run_batch, select, slugify
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


class DefaultCommandGroup(TyperGroup):
    """A command group whose default command is ``ingest``.

    Lets ``hearsay <SOURCE> [options]`` work without typing ``ingest`` while
    still supporting the ``mcp`` subcommand. If the first argument is neither a
    known command nor an option, ``ingest`` is prepended.
    """

    default_command = "ingest"
    # Options handled by the group itself (not by the default command).
    _group_options = frozenset({"--version", "--help", "-h"})

    def parse_args(self, ctx, args):
        # Prepend `ingest` unless the first token is a known subcommand or a
        # group-level option, so both `hearsay <SOURCE> [options]` and
        # `hearsay [options] <SOURCE>` reach ingestion, while `hearsay mcp`,
        # `hearsay --version`, and `hearsay --help` still work.
        if args and args[0] not in self.commands and args[0] not in self._group_options:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultCommandGroup,
    add_completion=False,
    no_args_is_help=True,
    help=PITCH,
)
console = Console()
err_console = Console(stderr=True)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"hearsay {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """crawl4ai for video & audio — clean, timestamped, LLM-ready transcripts."""


@app.command("mcp", help="Run the MCP stdio server (give your AI agent ears).")
def mcp_command() -> None:
    """Start the hearsay MCP server on stdio."""
    from hearsay.mcp_server import run_server

    try:
        run_server()
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except HearsayError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc


@app.command("ingest", help=PITCH)
def ingest(
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
) -> None:
    """Ingest SOURCE into timestamped, LLM-ready markdown."""
    opts = _Options(
        output=output,
        output_dir=output_dir,
        language=language,
        transcribe=transcribe,
        model=model.value,
        write_json=write_json,
        latest=latest,
        episode=episode,
        all_=all_,
        limit=limit,
    )
    try:
        path = Path(source).expanduser()
        video_id = extract_video_id(source)
        if path.is_file():
            _run_single(_ingest_file(path, opts), path.stem, opts)
        elif extract_playlist_id(source) is not None:
            _run_playlist(source, opts)
        elif video_id is not None:
            _run_single(_ingest_youtube_source(source, opts), video_id, opts)
        elif source.startswith(("http://", "https://")):
            _run_feed(source, opts)
        else:
            _run_single(_ingest_file(path, opts), path.stem, opts)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except HearsayError as exc:
        _print_error(exc)
        raise typer.Exit(code=1) from exc


@dataclass(frozen=True)
class _Options:
    """Resolved CLI options, passed explicitly so a typo can't become a KeyError."""

    output: Path | None
    output_dir: Path
    language: str | None
    transcribe: bool
    model: str
    write_json: bool
    latest: bool
    episode: int | None
    all_: bool
    limit: int | None


# --- Single-source ingestion ---------------------------------------------


def _run_single(document: Document, default_name: str, opts: _Options) -> None:
    destination = opts.output if opts.output is not None else Path(f"./{default_name}.md")
    written = _write_document(document, destination, write_json=opts.write_json)
    _print_success(document, written)


def _ingest_file(path: Path, opts: _Options) -> Document:
    with _progress(f"Transcribing {path.name} with whisper '{opts.model}'") as cb:
        return ingest_file(path, model_size=opts.model, language=opts.language, on_progress=cb)


def _ingest_youtube_source(url: str, opts: _Options) -> Document:
    """Captions path, or whisper — forced by --transcribe or auto on no captions."""
    if opts.transcribe:
        return _transcribe_youtube(url, opts, forced=True)
    video_id = extract_video_id(url) or url
    try:
        with console.status(f"[bold]Fetching captions for {video_id}…", spinner="dots"):
            return ingest_youtube(url, language=opts.language or "en")
    except NoCaptionsError:
        console.print("[yellow]No captions found.[/yellow] Falling back to local transcription.")
        return _transcribe_youtube(url, opts, forced=False)


def _transcribe_youtube(url: str, opts: _Options, *, forced: bool) -> Document:
    why = "Transcribing" if forced else "Downloading audio, then transcribing"
    console.print(
        f"[dim]{why} locally with whisper '{opts.model}'. This can take a few minutes.[/dim]"
    )
    with _progress("Transcribing audio") as cb:
        return ingest_youtube_transcribe(
            url, model_size=opts.model, language=opts.language, on_progress=cb
        )


# --- Batch ingestion (playlists & feeds) ----------------------------------


def _run_playlist(url: str, opts: _Options) -> None:
    with console.status("[bold]Listing playlist…", spinner="dots"):
        title, entries = fetch_playlist(url)
    items = [
        BatchItem(title=e.title, slug=e.video_id, ingest=_youtube_entry_ingester(e, opts))
        for e in entries
    ]
    _run_batch_or_list(items, title, "video", opts, episodes=None)


def _run_feed(url: str, opts: _Options) -> None:
    with console.status("[bold]Fetching feed…", spinner="dots"):
        feed = fetch_feed(url)
    items = [
        BatchItem(
            title=ep.title,
            slug=slugify(ep.title, fallback=f"episode-{i}"),
            ingest=_episode_ingester(ep, feed.title, opts),
        )
        for i, ep in enumerate(feed.episodes, start=1)
    ]
    _run_batch_or_list(items, feed.title, "episode", opts, episodes=feed.episodes)


def _youtube_entry_ingester(entry: PlaylistEntry, opts: _Options) -> Callable[[], Document]:
    def _ingest() -> Document:
        if opts.transcribe:
            return ingest_youtube_transcribe(
                entry.url, model_size=opts.model, language=opts.language
            )
        try:
            return ingest_youtube(entry.url, language=opts.language or "en")
        except NoCaptionsError:
            return ingest_youtube_transcribe(
                entry.url, model_size=opts.model, language=opts.language
            )

    return _ingest


def _episode_ingester(episode: Episode, show: str, opts: _Options) -> Callable[[], Document]:
    def _ingest() -> Document:
        return ingest_episode(episode, show, model_size=opts.model, language=opts.language)

    return _ingest


def _run_batch_or_list(
    items: list[BatchItem],
    source_title: str,
    noun: str,
    opts: _Options,
    *,
    episodes: list[Episode] | None,
) -> None:
    ensure_unique_slugs(items)  # distinct items must not overwrite each other's output
    selected = select(
        items, latest=opts.latest, episode=opts.episode, all_=opts.all_, limit=opts.limit
    )
    if not selected:
        _print_listing(items, source_title, noun, episodes=episodes)
        return

    _make_output_dir(opts.output_dir)

    def write(document: Document, slug: str) -> Path:
        return _write_document(document, opts.output_dir / f"{slug}.md", write_json=opts.write_json)

    def announce(index: int, total: int, item: BatchItem) -> None:
        console.print(f"[bold blue]\\[{index}/{total}][/bold blue] {item.title}")

    results = run_batch(selected, write, on_item=announce)
    _print_summary(results, opts.output_dir)


def _make_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(
            f"Could not create output directory {output_dir}: {exc.strerror or exc}",
            hint="Pick a writable --output-dir whose path is not an existing file.",
        ) from exc


# --- Output writing -------------------------------------------------------


def _write_document(document: Document, destination: Path, *, write_json: bool) -> Path:
    """Write the markdown (and optional JSON sidecar); return the markdown path."""
    _write_text(destination, render_markdown(document))
    if write_json:
        sidecar = destination.with_suffix(".json")
        if sidecar == destination:  # destination already ends in .json — don't clobber it
            sidecar = destination.with_name(destination.name + ".json")
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
