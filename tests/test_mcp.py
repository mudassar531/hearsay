"""MCP server tests: tool wiring (offline) and a real stdio round-trip.

The round-trip spawns `python -m hearsay mcp` as a subprocess and talks the MCP
protocol over stdio. It runs the real `tiny` model with HF_HUB_OFFLINE=1 so it
stays offline when the model is cached, and skips otherwise. (The subprocess is
not subject to conftest's in-process socket guard, so HF_HUB_OFFLINE is what
keeps it from reaching the network.)
"""

import asyncio
import datetime as dt
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import hearsay.mcp_server as mcp_server
from hearsay.errors import NoCaptionsError
from hearsay.mcp_server import build_server, ingest_file_markdown, ingest_url_markdown
from hearsay.models import Document, Paragraph, Section, SourceMetadata

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"


def _doc(title: str = "clip", method: str = "captions") -> Document:
    meta = SourceMetadata(title=title, source="x", channel="c", duration_s=5.0, video_id=title)
    return Document(
        meta=meta,
        method=method,
        language="en",
        ingested_at="2026-06-13T10:00:00Z",
        sections=[
            Section(
                title="Intro",
                start_s=0,
                paragraphs=[Paragraph(text="hello there world", start_s=0, end_s=5)],
            )
        ],
    )


def _tiny_cached() -> bool:
    try:
        from faster_whisper import WhisperModel

        WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
        return True
    except Exception:
        return False


def test_server_exposes_both_tools() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"ingest_url", "ingest_file"} <= names
    ingest_url = next(t for t in tools if t.name == "ingest_url")
    # The ingest_url tool accepts url + optional transcribe/lang per the spec.
    props = ingest_url.inputSchema.get("properties", {})
    assert "url" in props
    assert "transcribe" in props
    assert "lang" in props


def test_ingest_url_markdown_uses_captions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tool logic without a real model: captions path returns rendered markdown.
    monkeypatch.setattr(mcp_server, "ingest_youtube", lambda url, language: _doc())
    markdown = ingest_url_markdown("https://youtu.be/x")
    assert markdown.startswith("---")
    assert "# clip" in markdown


def test_ingest_url_markdown_falls_back_to_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_captions(url, language):
        raise NoCaptionsError("no captions", hint="...")

    fell_back = {"yes": False}

    def fake_transcribe(url, **kwargs):
        fell_back["yes"] = True
        return _doc(method="whisper-small")

    monkeypatch.setattr(mcp_server, "ingest_youtube", no_captions)
    monkeypatch.setattr(mcp_server, "ingest_youtube_transcribe", fake_transcribe)
    markdown = ingest_url_markdown("https://youtu.be/x")
    assert fell_back["yes"] is True
    assert "whisper-small" in markdown


def test_ingest_url_markdown_transcribe_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_transcribe(url, **kwargs):
        captured["model"] = kwargs.get("model_size")
        return _doc(method="whisper-tiny")

    monkeypatch.setenv("HEARSAY_MODEL", "tiny")
    monkeypatch.setattr(mcp_server, "ingest_youtube_transcribe", fake_transcribe)
    markdown = ingest_url_markdown("https://youtu.be/x", transcribe=True)
    assert captured["model"] == "tiny"  # HEARSAY_MODEL env is honored
    assert markdown.startswith("---")


def test_ingest_file_markdown_renders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mcp_server, "_ingest_file", lambda path, **kwargs: _doc("rec"))
    markdown = ingest_file_markdown(str(tmp_path / "rec.wav"))
    assert markdown.startswith("---")
    assert "# rec" in markdown


@pytest.mark.skipif(not _tiny_cached(), reason="tiny whisper model not cached")
def test_mcp_stdio_roundtrip() -> None:
    async def _run() -> str:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "hearsay", "mcp"],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "HEARSAY_MODEL": "tiny"},
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            assert {"ingest_url", "ingest_file"} <= {t.name for t in listed.tools}
            result = await session.call_tool(
                "ingest_file",
                {"path": str(SAMPLE)},
                read_timeout_seconds=dt.timedelta(seconds=120),
            )
            assert result.isError is False
            return "".join(getattr(c, "text", "") for c in result.content)

    markdown = asyncio.run(_run())
    assert markdown.startswith("---")
    assert "# sample" in markdown
    assert "fox" in markdown.lower()


def test_server_reports_hearsay_version() -> None:
    # FastMCP takes no version, so serverInfo defaulted to the MCP SDK's release —
    # which tells a client nothing about which hearsay it is talking to.
    from hearsay import __version__

    assert build_server()._mcp_server.version == __version__


def test_tools_do_not_block_the_event_loop() -> None:
    """Transcription must run off the event loop.

    FastMCP invokes a *sync* tool directly in the async handler, so a multi-minute
    decode would stall the whole stdio session — no other call, no keepalive, no
    cancellation — until it returned.
    """
    import time

    server = build_server()
    tool = server._tool_manager.get_tool("ingest_file")
    assert tool.is_async, "tool must be a coroutine so it can offload its work"

    async def exercise() -> int:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await tool.run({"path": str(SAMPLE)})
        finally:
            beat.cancel()
        return ticks

    def slow(path: str) -> str:
        time.sleep(0.5)
        return "ok"

    original = mcp_server.ingest_file_markdown
    mcp_server.ingest_file_markdown = slow
    try:
        ticks = asyncio.run(exercise())
    finally:
        mcp_server.ingest_file_markdown = original
    assert ticks > 5, f"event loop was blocked during the tool call (only {ticks} ticks)"
