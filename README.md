# hearsay

> **crawl4ai for video & audio.** One command turns any YouTube video, podcast
> episode, or local recording into clean, timestamped, chunked, LLM-ready
> markdown — for RAG pipelines and AI agents.

[![CI](https://github.com/mudassar531/hearsay/actions/workflows/ci.yml/badge.svg)](https://github.com/mudassar531/hearsay/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![hearsay in action](demo/demo.gif)

## Why

Getting a transcript into your RAG pipeline usually means gluing together
`yt-dlp`, Whisper, and a pile of timestamp-wrangling scripts — and you still end
up with one line per caption fragment or an undifferentiated wall of text.
hearsay does the whole thing in one command and gives you back markdown a human
*and* a model can read: readable paragraphs, real timestamps, chapter headings,
and an optional JSON sidecar with a stable schema.

## Install

```bash
uv tool install hearsay          # recommended
# or
pipx install hearsay
# transcription + MCP server support:
uv tool install "hearsay[mcp]"
```

> **Pre-release:** hearsay isn't on PyPI yet. Until the first release, install
> from a checkout:
>
> ```bash
> git clone https://github.com/mudassar531/hearsay
> cd hearsay
> uv tool install .               # puts `hearsay` on your PATH
> # or, for development:  uv sync && uv run hearsay --help
> ```

**System requirement:** [ffmpeg](#requirements) on your PATH.

## 30-second quickstart

```bash
# YouTube → markdown via captions (fast — no download)
hearsay "https://www.youtube.com/watch?v=VIDEO_ID"

# Local audio/video → markdown via local Whisper (runs on CPU)
hearsay talk.mp3

# Force Whisper on a YouTube URL, pick a model, also emit JSON
hearsay "https://youtu.be/VIDEO_ID" --transcribe --model small --json

# A podcast feed or YouTube playlist: list, then ingest a selection
hearsay "https://example.com/feed.xml"
hearsay "https://example.com/feed.xml" --all --limit 3 --output-dir ./out
```

No captions on a video? hearsay falls back to local Whisper automatically.

## What you get

```markdown
---
title: "You Would Be a Terrible Leader"
source: "https://www.youtube.com/watch?v=rStL7niR7gs"
channel: "CGP Grey"
duration: "00:18:13"
ingested: "2026-06-13T10:00:00Z"
method: "captions"
language: "en"
---

# You Would Be a Terrible Leader

## [00:00:00 – 00:05:21]

**[00:00:00]** Do you want to rule? Do you see the problems in your country and
know how to fix them? If only you had the power to do so. Well. You've come to
the right place. But, before we begin this lesson in political power, ask
yourself, why don't rulers see as clearly as you...
```

Pass `--json` for a sidecar matching the [`Transcript` schema](docs/schema.json):
metadata plus `chunks[]`, each with `start_s`, `end_s`, `section`, and `text` —
ready to embed.

## How it compares

| | **hearsay** | DIY `yt-dlp` + Whisper | markitdown / docling |
| --- | --- | --- | --- |
| Input | video & **audio** | video & audio (you wire it) | documents (pdf/docx/pptx) |
| One command | ✅ | ❌ multi-step plumbing | ✅ (for docs) |
| Captions-first (no download) | ✅ | ✗ usually re-transcribes | n/a |
| Timestamps + paragraph grouping | ✅ readable | ✗ raw segments | n/a |
| Chapters → sections | ✅ | ✗ manual | n/a |
| Podcasts · playlists · batch | ✅ | ✗ manual | ✗ |
| JSON sidecar for RAG | ✅ stable schema | ✗ manual | varies |
| MCP server for agents | ✅ | ✗ | varies |

hearsay does **media**; document tools like
[markitdown](https://github.com/microsoft/markitdown) and
[docling](https://github.com/docling-project/docling) do **documents**. Use both.

## Give your agent ears

hearsay ships an [MCP](https://modelcontextprotocol.io) server so AI agents can
ingest media themselves. It exposes two tools — `ingest_url(url, transcribe?, lang?)`
and `ingest_file(path)` — that each return clean, timestamped markdown.

```bash
uv tool install "hearsay[mcp]"
hearsay mcp                      # stdio MCP server (Ctrl-C to stop)
```

**Claude Code:**

```bash
claude mcp add hearsay -- hearsay mcp
```

or add to `.mcp.json` (project) / `~/.claude.json` (user):

```json
{
  "mcpServers": {
    "hearsay": {
      "type": "stdio",
      "command": "hearsay",
      "args": ["mcp"]
    }
  }
}
```

**Claude Desktop** — add to `claude_desktop_config.json` (Settings → Developer →
Edit Config; macOS: `~/Library/Application Support/Claude/`, Windows:
`%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "hearsay": {
      "type": "stdio",
      "command": "hearsay",
      "args": ["mcp"],
      "env": {
        "HEARSAY_MODEL": "small"
      }
    }
  }
}
```

If `hearsay` is not on the host's PATH, use the absolute path (`which hearsay`),
or `"command": "python"`, `"args": ["-m", "hearsay", "mcp"]`.

Server configuration (env vars, since MCP tool signatures are fixed):

| Variable | Default | Effect |
| --- | --- | --- |
| `HEARSAY_MODEL` | `small` | Whisper model size (`tiny`…`large-v3`) |
| `HEARSAY_LANG` | _(unset)_ | Default language: English captions, else Whisper auto-detect |

## CLI reference

```text
hearsay <SOURCE> [options]      SOURCE = YouTube video/playlist URL, podcast RSS, or local file

  -o, --output PATH    Output file for a single source (default ./<id>.md)
  --output-dir PATH    Output directory for batch (playlist/feed) ingestion (default ./hearsay-out)
  --lang CODE          Language: captions default to English; transcription auto-detects
  --transcribe         Force local Whisper even when captions exist
  --model SIZE         Whisper model: tiny | base | small | medium | large-v3 (default small)
  --json               Also write a .json sidecar (Transcript schema)
  --latest             Batch: ingest only the most recent item
  --episode N          Batch: ingest only item N (1-indexed)
  --all [--limit N]    Batch: ingest all items (optionally capped)
  --version            Show version

hearsay mcp            Run the MCP stdio server
```

## Requirements

- **Python 3.11+**
- **ffmpeg** on your PATH. hearsay decodes most audio/video directly
  (faster-whisper bundles its own decoder), but ffmpeg is the safe baseline and
  is used for some yt-dlp format merges.

| OS | Install ffmpeg |
| --- | --- |
| macOS (Homebrew) | `brew install ffmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| Windows (winget) | `winget install Gyan.FFmpeg` |
| Windows (Chocolatey) | `choco install ffmpeg` |

The first transcription downloads the chosen Whisper model once (tens of MB to
~1.5 GB), then caches it for offline use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](docs/good-first-issues.md). hearsay does one thing well —
media → great markdown — and aims to keep doing exactly that.

## License

[MIT](LICENSE)
