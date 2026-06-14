# hearsay

> **crawl4ai for video & audio.** One command turns any YouTube video, podcast
> episode, or local recording into clean, timestamped, chunked, LLM-ready
> markdown — for RAG pipelines, notes, and AI agents.

[![PyPI](https://img.shields.io/pypi/v/hearsay)](https://pypi.org/project/hearsay/)
[![CI](https://github.com/mudassar531/hearsay/actions/workflows/ci.yml/badge.svg)](https://github.com/mudassar531/hearsay/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<table>
  <tr>
    <td width="50%" valign="top" align="center">
      <b>Command line</b> — for pipelines &amp; automation<br><br>
      <img src="demo/demo.gif" alt="hearsay on the command line" width="100%">
    </td>
    <td width="50%" valign="top" align="center">
      <b>Web UI</b> — <code>hearsay web</code>, in your browser<br><br>
      <img src="demo/webui.gif" alt="hearsay web UI" width="100%">
    </td>
  </tr>
</table>

Same engine, two front ends — use whichever fits. Paste a link, get back
markdown a human *and* a model can read: readable
paragraphs, real timestamps, chapter headings, and an optional JSON sidecar with
a stable schema. Captions when they exist (fast, no download); local Whisper or
Apple-Silicon **Parakeet** transcription when they don't.

```bash
uv tool install hearsay
hearsay "https://www.youtube.com/watch?v=VIDEO_ID"   # → ./VIDEO_ID.md
```

- [Two ways to use it](#two-ways-to-use-it)
- [Why](#why) · [Install](#install) · [Transcription engines](#transcription-engines)
- [Quickstart](#quickstart) · [Web UI](#web-ui) · [What you get](#what-you-get)
- [Build training datasets](#build-training-datasets) · [How it compares](#how-it-compares)
- [Give your agent ears (MCP)](#give-your-agent-ears) · [CLI reference](#cli-reference)
- [Requirements](#requirements) · [Contributing](#contributing)

## Two ways to use it

**Command line** — for pipelines, batches, and automation:

```bash
hearsay "https://youtu.be/VIDEO_ID" --json        # markdown + JSON sidecar
hearsay "https://example.com/feed.xml" --all      # batch a whole podcast feed
```

**Web UI** — for a browser: run `hearsay web`, paste a URL or drop in a file, and
watch the clean markdown appear with a live preview, copy, and download. It's a
single self-contained page with **no extra dependencies** (Python standard
library only). See it in action up top, and [details below](#web-ui).

## Why

Getting a transcript into your RAG pipeline usually means gluing together
`yt-dlp`, Whisper, and a pile of timestamp-wrangling scripts — and you still end
up with one line per caption fragment or an undifferentiated wall of text.
hearsay does the whole thing in one command:

- **Captions-first.** Uses YouTube captions when available — fast, no media download.
- **Falls back to transcription** automatically (local Whisper, or Parakeet on Apple Silicon) when there are none.
- **Readable output.** Caption fragments are regrouped into real paragraphs (40–120 words) split on pauses and sentence boundaries — never one line per fragment, never a wall of text.
- **Structured.** Chapters become `##` sections (or ~5-minute time windows), every paragraph keeps a `[hh:mm:ss]` timestamp, and `--json` emits a sidecar with a stable schema.
- **Scales.** Single videos, whole YouTube playlists, and podcast RSS feeds — batch into a folder, continuing past per-item failures.

## Install

```bash
uv tool install hearsay          # recommended
# or
pipx install hearsay
```

Optional extras:

```bash
uv tool install "hearsay[mcp]"        # MCP server, for AI agents
uv tool install "hearsay[parakeet]"   # fast Apple-Silicon engine (macOS arm64)
uv tool install "hearsay[diarize]"    # speaker diarization for single-voice TTS datasets
```

**System requirement:** [ffmpeg](#requirements) on your PATH.

<details>
<summary>From source (for development)</summary>

```bash
git clone https://github.com/mudassar531/hearsay
cd hearsay
uv sync && uv run hearsay --help    # or: uv tool install .
```

</details>

## Transcription engines

When a video has no captions (or you pass `--transcribe`), hearsay transcribes
locally with the fastest engine your machine has. The default `--model auto`
picks:

| Engine | When | Speed | Notes |
| --- | --- | --- | --- |
| **Parakeet** (NVIDIA Parakeet-TDT on Apple MLX) | Apple Silicon + `parakeet` extra | ~24× realtime (M1 Pro) | `parakeet` is multilingual (25 European languages); `parakeet-en` is English-only |
| **Whisper** (faster-whisper, CPU int8) | everywhere else, or an explicit size | ~7× realtime | sizes `tiny`…`large-v3`; `large-v3` is the multilingual ceiling |

On Apple Silicon, Parakeet is about **3× faster** than `whisper-small` at
comparable accuracy. If the `parakeet` extra isn't installed, `auto` falls back
to `whisper-small` automatically — so hearsay behaves the same everywhere, just
faster on a Mac. Models download once (Whisper: tens of MB to ~1.5 GB; Parakeet
v3: ~2.5 GB) and are cached for offline use.

## Quickstart

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

## Web UI

Prefer a browser? `hearsay web` starts a tiny local web UI — paste a YouTube URL
or attach an audio/video file, pick the model, and get a live markdown preview
with copy, download, and a history of past transcripts. It's a single
self-contained page built on the Python standard library, with **no extra
dependencies**, and it binds to `127.0.0.1` so nothing leaves your machine.

```bash
hearsay web                      # → http://localhost:8756
hearsay web --port 9000          # custom port
hearsay web --host 0.0.0.0       # expose on your LAN (unauthenticated — careful)
```

1. Run `hearsay web` and open the printed URL.
2. Paste a YouTube link (or click **Attach** to upload a file).
3. Optionally tick **Force transcription**, toggle **VAD**, or pick a model.
4. Hit send — the transcript renders with **Copy** / **Download** buttons.

Tick **Dataset** to build a [training dataset](#build-training-datasets) instead
(set clip length and sample rate); it downloads as a `.zip`. Single video URLs and
file uploads go through the UI; for playlists and podcast feeds, use the CLI (the UI
shows a friendly hint).

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

The `method` field records exactly how the text was produced — `captions`,
`captions-auto`, `whisper-small`, `parakeet-tdt-0.6b-v3`, etc. — so a downstream
consumer can tell a human transcript from a machine one.

Pass `--json` for a sidecar matching the [`Transcript` schema](docs/schema.json):
metadata plus `chunks[]`, each with `start_s`, `end_s`, `section`, and `text` —
ready to embed.

## Build training datasets

The markdown above is for **reading** (RAG, humans). `hearsay dataset` is a second,
separate mode that turns the same media into **ML training datasets** for **TTS and
STT** — audio sliced into short clips paired with exact, verbatim transcripts, in
standard layouts a training pipeline reads directly.

```bash
# A short video → a dataset folder (LJSpeech metadata.csv + NeMo manifest.jsonl + wavs/)
hearsay dataset "https://youtu.be/VIDEO_ID" --out ./voice-data

# A whole playlist / channel / podcast feed → one merged dataset
hearsay dataset "https://example.com/feed.xml" --out ./speech-data

# 16 kHz mono for ASR, custom clip length, also emit a HuggingFace audiofolder index
hearsay dataset talk.mp3 --sample-rate 16000 --segment-min 2 --segment-max 12 --format hf
```

You get a folder like:

```text
voice-data/
  wavs/VIDEO_ID_0001.wav …     # mono 16-bit PCM, cut on sentence/pause boundaries, never mid-word
  metadata.csv                  # LJSpeech: id|text|text  (Coqui / Piper read this directly)
  manifest.jsonl                # NeMo/ESPnet: {"audio_filepath","duration","text","offset"}
  dataset_card.md               # provenance, counts, language + a rights/consent note
  dropped.jsonl                 # every filtered-out clip, with the reason
```

- **Word-accurate cuts.** Clips are sliced on word-level timestamps (faster-whisper
  `word_timestamps`, or Parakeet on Apple Silicon) — never mid-word.
- **Quality filtering** (on by default) drops junk: too-short/long, internal silence,
  wrong-script/odd speaking-rate text, repetition, and low ASR confidence. Each drop is
  logged; `--no-filter` keeps everything.
- **Single-voice TTS from multi-speaker audio** needs diarization:
  `uv tool install "hearsay[diarize]"`, accept the
  [pyannote model](https://hf.co/pyannote/speaker-diarization-community-1) conditions and
  set `HF_TOKEN`, then `--dominant-speaker` (keep the host) or `--per-speaker` (one index
  per speaker). Without it, datasets are **mixed-speaker** (fine for STT) and hearsay says so.

> **Accuracy & rights.** Word boundaries from Whisper/Parakeet are good but not
> phonetically exact — clips are padded and snapped to pauses, and you should spot-check.
> **You are responsible** for the rights to any media you process and for voice consent
> (cloning a real person's voice may require it); extracting audio from YouTube may breach
> its Terms. hearsay is local and ships no datasets. *Informational, not legal advice* —
> see each generated `dataset_card.md`.

## How it compares

| | **hearsay** | DIY `yt-dlp` + Whisper | markitdown / docling |
| --- | --- | --- | --- |
| Input | video & **audio** | video & audio (you wire it) | documents (pdf/docx/pptx) |
| One command | ✅ | ❌ multi-step plumbing | ✅ (for docs) |
| Captions-first (no download) | ✅ | ✗ usually re-transcribes | n/a |
| Timestamps + paragraph grouping | ✅ readable | ✗ raw segments | n/a |
| Chapters → sections | ✅ | ✗ manual | n/a |
| Podcasts · playlists · batch | ✅ | ✗ manual | ✗ |
| Fast Apple-Silicon engine | ✅ Parakeet (MLX) | ✗ DIY | n/a |
| JSON sidecar for RAG | ✅ stable schema | ✗ manual | varies |
| **TTS/STT dataset export** | ✅ LJSpeech + JSONL, filtered, diarizable | ✗ DIY plumbing | ✗ |
| Browser UI + MCP server | ✅ | ✗ | varies |

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
> lyric transcript, since these are speech models, not lyrics transcribers.

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

hearsay dataset <SOURCE> [options]   Build a TTS/STT training dataset
  --out PATH           Dataset output directory (default ./hearsay-dataset)
  --format FMT         ljspeech | jsonl | hf (repeatable; default ljspeech + jsonl)
  --sample-rate HZ     Output WAV rate (default 22050; 16000 for ASR)
  --segment-min/max S  Clip length bounds in seconds (default 1–15)
  --model / --lang / --vad / --no-vad    Transcription (as above)
  --normalize          EBU R128 loudness-normalize each clip
  --no-filter          Keep every clip (skip the quality filters)
  --diarize            Label speakers (needs hearsay[diarize] + HF_TOKEN)
  --per-speaker        Diarize and emit a per-speaker index
  --dominant-speaker   Diarize and keep only the most-spoken speaker
  --limit N            Batch: cap items from a playlist/feed

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
