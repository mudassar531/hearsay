# hearsay

> **crawl4ai for video & audio.** One command turns any YouTube video, podcast
> episode, or local recording into clean, timestamped, chunked, LLM-ready
> markdown — for RAG pipelines and AI agents.

[![PyPI](https://img.shields.io/pypi/v/hearsay)](https://pypi.org/project/hearsay/)
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
# MCP server support:
uv tool install "hearsay[mcp]"
# Apple Silicon: add the fast Parakeet engine (~3x faster than CPU Whisper):
uv tool install "hearsay[parakeet]"
```

**System requirement:** [ffmpeg](#requirements) on your PATH.

### Transcription engines

hearsay transcribes with the fastest engine your machine has. The default
`--model auto` picks:

- **Parakeet** (NVIDIA Parakeet-TDT on Apple's MLX) on Apple Silicon when the
  `parakeet` extra is installed — about **3× faster** than `whisper-small` on
  CPU (~24× vs ~7× realtime on an M1 Pro), with comparable accuracy. `parakeet`
  is multilingual (25 European languages); `parakeet-en` is English-only.
- **Whisper** (faster-whisper, CPU int8) everywhere else, or when you pass an
  explicit size (`tiny`…`large-v3`).

If the Parakeet extra isn't installed, `auto` falls back to `whisper-small`
automatically, so hearsay works the same everywhere — it's just faster on a Mac.

<details>
<summary>From source (for development)</summary>

```bash
git clone https://github.com/mudassar531/hearsay
cd hearsay
uv sync && uv run hearsay --help    # or: uv tool install .
```

</details>

## 30-second quickstart

```bash
# YouTube → markdown via captions (fast — no download)
hearsay "https://www.youtube.com/watch?v=VIDEO_ID"

# Local audio/video → markdown (fast Parakeet on Apple Silicon, else CPU Whisper)
hearsay talk.mp3

# Force local transcription on a YouTube URL, pick an engine, also emit JSON
hearsay "https://youtu.be/VIDEO_ID" --transcribe --model parakeet --json

# Music/song? Add --no-vad so the lyrics aren't filtered out as "non-speech"
hearsay "https://youtu.be/SONG_ID" --no-vad

# A podcast feed or YouTube playlist: list, then ingest a selection
hearsay "https://example.com/feed.xml"
hearsay "https://example.com/feed.xml" --all --limit 3 --output-dir ./out
```

No captions on a video? hearsay falls back to local transcription automatically.

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

## Web UI

Prefer a browser? `hearsay web` starts a tiny local web UI — paste a YouTube URL
or drop in an audio/video file, pick the model, and get a live markdown preview
with copy and download. It's a single self-contained page with **no extra
dependencies** (built on the standard library).

```bash
hearsay web                      # → http://localhost:8756
hearsay web --port 9000          # custom port
```

(Single video URLs and file uploads; for playlists and podcast feeds use the CLI.)

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
        "HEARSAY_MODEL": "auto"
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
| `HEARSAY_MODEL` | `auto` | `auto`, `parakeet`, `parakeet-en`, or a Whisper size (`tiny`…`large-v3`) |
| `HEARSAY_LANG` | _(unset)_ | Default language: English captions, else transcription auto-detect |
| `HEARSAY_VAD` | `1` | Voice-activity filter (Whisper); set `0` for music/songs |
| `HEARSAY_PARAKEET_MODEL` | _(unset)_ | Override the Parakeet MLX repo id (advanced) |

> **Speech vs. music:** hearsay is tuned for spoken audio (podcasts, talks,
> interviews, meetings), where transcription is accurate. For music, pass
> `--no-vad` so the vocals aren't discarded — but expect a rough, approximate
> lyric transcript, since Whisper is a speech model, not a lyrics transcriber.

## CLI reference

```text
hearsay <SOURCE> [options]      SOURCE = YouTube video/playlist URL, podcast RSS, or local file

  -o, --output PATH    Output file for a single source (default ./<id>.md)
  --output-dir PATH    Output directory for batch (playlist/feed) ingestion (default ./hearsay-out)
  --lang CODE          Language: captions default to English; transcription auto-detects
  --transcribe         Force local transcription even when captions exist
  --model MODEL        auto (default) | parakeet | parakeet-en | tiny | base | small | medium | large-v3
  --no-vad             Disable voice-activity filtering (Whisper; use for music/songs)
  --json               Also write a .json sidecar (Transcript schema)
  --latest             Batch: ingest only the most recent item
  --episode N          Batch: ingest only item N (1-indexed)
  --all [--limit N]    Batch: ingest all items (optionally capped)

hearsay web            Run the local web UI (--host, --port)
hearsay mcp            Run the MCP stdio server
hearsay --version      Show the version
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

The first transcription downloads the chosen model once (Whisper: tens of MB to
~1.5 GB; Parakeet v3: ~2.5 GB), then caches it for offline use.

> **Apple Silicon speed:** the `parakeet` extra (`uv tool install
> "hearsay[parakeet]"`) runs NVIDIA Parakeet on MLX, transcribing ~3× faster
> than CPU Whisper (~24× realtime on an M1 Pro). It requires macOS on arm64; on
> other platforms hearsay uses CPU Whisper automatically.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](docs/good-first-issues.md). hearsay does one thing well —
media → great markdown — and aims to keep doing exactly that.

## License

[MIT](LICENSE)
