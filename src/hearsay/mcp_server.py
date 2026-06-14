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
from pathlib import Path

from hearsay.errors import HearsayError, NoCaptionsError
from hearsay.pipeline import ingest_file as _ingest_file
from hearsay.pipeline import ingest_youtube, ingest_youtube_transcribe
from hearsay.render import render_markdown
from hearsay.transcribe import DEFAULT_MODEL


def _model() -> str:
    return os.environ.get("HEARSAY_MODEL", DEFAULT_MODEL)


def _default_lang() -> str | None:
    return os.environ.get("HEARSAY_LANG") or None


def _vad() -> bool:
    # On by default (speech); set HEARSAY_VAD=0/false/no for music/songs.
    return os.environ.get("HEARSAY_VAD", "1").strip().lower() not in {"0", "false", "no"}


def ingest_url_markdown(url: str, transcribe: bool = False, lang: str | None = None) -> str:
    """Ingest a single YouTube URL and return the markdown (used by the MCP tool)."""
    language = lang if lang is not None else _default_lang()
    if transcribe:
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
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise HearsayError(
            "MCP support is not installed.",
            hint="Install the optional extra: pip install 'hearsay[mcp]' (or uv add hearsay[mcp]).",
        ) from exc

    server = FastMCP("hearsay")

    @server.tool()
    def ingest_url(url: str, transcribe: bool = False, lang: str | None = None) -> str:
        """Ingest a YouTube video URL into clean, timestamped, LLM-ready markdown.

        Args:
            url: A YouTube video URL.
            transcribe: Force local Whisper transcription instead of captions.
            lang: Preferred language code (captions default to English).
        """
        try:
            return ingest_url_markdown(url, transcribe=transcribe, lang=lang)
        except HearsayError as exc:
            raise ValueError(_format_error(exc)) from exc

    @server.tool()
    def ingest_file(path: str) -> str:
        """Transcribe a local audio/video file into clean, timestamped markdown.

        Args:
            path: Path to a local audio or video file.
        """
        try:
            return ingest_file_markdown(path)
        except HearsayError as exc:
            raise ValueError(_format_error(exc)) from exc

    return server


def _format_error(exc: HearsayError) -> str:
    return f"{exc.message} ({exc.hint})" if exc.hint else exc.message


def run_server() -> None:
    """Start the hearsay MCP server on stdio (blocks until the client exits)."""
    build_server().run(transport="stdio")
