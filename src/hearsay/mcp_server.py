"""MCP stdio server: give an AI agent ears.

Exposes two tools — ``ingest_url`` and ``ingest_file`` — that return clean,
timestamped markdown. The ``mcp`` SDK is an optional extra, so it is imported
lazily inside ``run_server``; ``hearsay mcp`` prints an install hint if missing.

Configuration via environment (MCP tool signatures are fixed, so knobs live in
the agent host's server config):
  HEARSAY_MODEL   transcription model: auto (default; Parakeet on Apple Silicon,
                  else whisper-small), a parakeet alias, or a whisper size
  HEARSAY_LANG    default language code (default: English captions / auto-detect)
  HEARSAY_VAD     voice-activity filter: 1 (default, speech); 0/false/no for music
  HEARSAY_PARAKEET_MODEL  override the Parakeet MLX repo id (advanced)
"""

import os
from functools import partial
from pathlib import Path

from hearsay import __version__
from hearsay.errors import HearsayError, NoCaptionsError
from hearsay.pipeline import ingest_file as _ingest_file
from hearsay.pipeline import ingest_youtube, ingest_youtube_transcribe
from hearsay.render import render_markdown
from hearsay.transcribe import DEFAULT_MODEL
from hearsay.youtube import extract_video_id


def _model() -> str:
    return os.environ.get("HEARSAY_MODEL", DEFAULT_MODEL)


def _default_lang() -> str | None:
    return os.environ.get("HEARSAY_LANG") or None


def _vad() -> bool:
    # On by default (speech); set HEARSAY_VAD=0/false/no for music/songs.
    return os.environ.get("HEARSAY_VAD", "1").strip().lower() not in {"0", "false", "no"}


def ingest_url_markdown(url: str, transcribe: bool = False, lang: str | None = None) -> str:
    """Ingest a single media URL and return the markdown (used by the MCP tool)."""
    language = lang if lang is not None else _default_lang()
    # Captions are a YouTube API; every other site yt-dlp supports transcribes directly,
    # as the CLI and web UI do. Sending a Dailymotion URL down the captions path raised
    # "video unavailable" — not the no-captions error the fallback below listens for.
    if transcribe or extract_video_id(url) is None:
        document = ingest_youtube_transcribe(
            url, model_size=_model(), language=language, vad_filter=_vad()
        )
    else:
        try:
            document = ingest_youtube(url, language=language or "en")
        except NoCaptionsError:
            document = ingest_youtube_transcribe(
                url, model_size=_model(), language=language, vad_filter=_vad()
            )
    return render_markdown(document)


def ingest_file_markdown(path: str) -> str:
    """Transcribe a local file and return the markdown (used by the MCP tool)."""
    document = _ingest_file(
        Path(path).expanduser(), model_size=_model(), language=_default_lang(), vad_filter=_vad()
    )
    return render_markdown(document)


def build_server():
    """Construct the FastMCP server with the two ingestion tools.

    Imports the optional ``mcp`` SDK lazily; raises HearsayError with an install
    hint if the extra is not installed.
    """
    try:
        import anyio.to_thread
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise HearsayError(
            "MCP support is not installed.",
            hint="Install the optional extra: pip install 'hearsay[mcp]' (or uv add hearsay[mcp]).",
        ) from exc

    server = FastMCP("hearsay")
    # FastMCP doesn't take a version and leaves the low-level server reporting the MCP
    # SDK's own version, so every client's serverInfo showed the SDK release instead of
    # hearsay's — useless for telling which hearsay an agent is actually talking to.
    server._mcp_server.version = __version__

    # Both tools are long, blocking, CPU-bound work (a model decode plus yt-dlp/ffmpeg
    # subprocesses) — minutes for a full episode. FastMCP calls a *sync* tool directly
    # on the event loop, which would stall the whole stdio session for the duration:
    # no other tool call, no keepalive, no cancellation until it returns. Running them
    # in a worker thread keeps the server answering while a transcription proceeds.
    @server.tool()
    async def ingest_url(url: str, transcribe: bool = False, lang: str | None = None) -> str:
        """Ingest a video or audio URL into clean, timestamped, LLM-ready markdown.

        YouTube videos use their captions when they have them (fast); every other site
        yt-dlp supports — Dailymotion, SoundCloud, Twitch, ~1800 more — is transcribed
        locally.

        Args:
            url: A YouTube video URL, or any media page yt-dlp supports.
            transcribe: Force local Whisper transcription instead of captions.
            lang: Preferred language code (captions default to English).
        """
        try:
            return await anyio.to_thread.run_sync(
                partial(ingest_url_markdown, url, transcribe=transcribe, lang=lang)
            )
        except HearsayError as exc:
            raise ValueError(_format_error(exc)) from exc

    @server.tool()
    async def ingest_file(path: str) -> str:
        """Transcribe a local audio/video file into clean, timestamped markdown.

        Args:
            path: Path to a local audio or video file.
        """
        try:
            return await anyio.to_thread.run_sync(partial(ingest_file_markdown, path))
        except HearsayError as exc:
            raise ValueError(_format_error(exc)) from exc

    return server


def _format_error(exc: HearsayError) -> str:
    return f"{exc.message} ({exc.hint})" if exc.hint else exc.message


def run_server() -> None:
    """Start the hearsay MCP server on stdio (blocks until the client exits)."""
    build_server().run(transport="stdio")
