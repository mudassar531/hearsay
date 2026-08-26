# hearsay

> **crawl4ai for video & audio.** One command turns any YouTube video, podcast
> episode, or local recording into clean, timestamped, LLM-ready **markdown** —
> or a **TTS/STT training dataset** of sliced audio clips paired with verbatim
> transcripts. Captions-first, runs locally, no plumbing.

[![PyPI](https://img.shields.io/pypi/v/hearsay)](https://pypi.org/project/hearsay/)
[![CI](https://github.com/mudassar531/hearsay/actions/workflows/ci.yml/badge.svg)](https://github.com/mudassar531/hearsay/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**One input — a link or a file — and two kinds of output.** Read it (RAG, notes,
agents) *or* train on it (text-to-speech, speech recognition):

<table>
  <tr>
    <td width="50%" valign="top" align="center">
      📄 <b>Clean markdown</b> — for RAG, notes &amp; agents<br><br>
      <img src="demo/demo.gif" alt="hearsay turning a video into markdown" width="100%">
    </td>
    <td width="50%" valign="top" align="center">
      🎙️ <b>TTS/STT dataset</b> — for training<br><br>
      <img src="demo/dataset.gif" alt="hearsay turning a recording into a training dataset" width="100%">
    </td>
  </tr>
</table>

```bash
uv tool install hearsay

hearsay "https://youtu.be/VIDEO_ID"                       # → markdown
hearsay dataset "https://youtu.be/VIDEO_ID" --out ./data  # → TTS/STT dataset
```

Captions when they exist (fast, no download); local **Whisper** or Apple-Silicon
**Parakeet** transcription when they don't. Single videos, whole playlists, and
podcast feeds. Nothing leaves your machine.

- [How it works](#how-it-works) · [Install](#install)
- [🎙️ Build TTS/STT datasets](#-build-ttsstt-training-datasets) · [📄 Clean markdown](#-clean-timestamped-markdown)
- [Web UI](#web-ui) · [Transcription engines](#transcription-engines) · [MCP server](#give-your-agent-ears)
- [How it compares](#how-it-compares) · [CLI reference](#cli-reference) · [Requirements](#requirements)

## How it works

One pipeline, two outputs. hearsay gets a **word-timestamped transcript** the
cheapest way it can — existing captions if the source has them, otherwise local
transcription — then either reflows it into readable markdown or slices the audio
into training clips:

```mermaid
flowchart LR
    S["YouTube · podcast feed<br/>playlist · local file"] --> T{"captions?"}
    T -- yes --> C["fetch captions<br/>(no download)"]
    T -- no --> W["transcribe locally<br/>Whisper / Parakeet"]
    C --> X(["timestamped transcript<br/>+ word timings"])
    W --> X
    X --> M["📄 markdown<br/>paragraphs · timestamps<br/>chapters · JSON sidecar"]
    X --> D["🎙️ TTS/STT dataset<br/>audio clips + transcripts<br/>LJSpeech · NeMo · HF"]
```

- **Captions-first.** Uses the source's captions when available — fast, no media download.
- **Falls back to transcription** automatically (CPU Whisper, or Parakeet on Apple Silicon).
- **Local & private.** Everything runs on your machine; hearsay hosts nothing and ships no data.
- **Scales.** One video, a whole YouTube playlist, or a podcast RSS feed — batched into one output.

## Install

```bash
uv tool install hearsay          # recommended
# or
pipx install hearsay
```

Optional extras:

```bash
uv tool install "hearsay[parakeet]"   # fast Apple-Silicon transcription (macOS arm64)
uv tool install "hearsay[diarize]"    # speaker diarization → single-voice TTS datasets
uv tool install "hearsay[mcp]"        # MCP server, for AI agents
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

## 🎙️ Build TTS/STT training datasets

`hearsay dataset` turns spoken media into a **machine-learning training dataset**:
the audio is sliced into short clips on word-level timestamps — **never mid-word** —
each paired with its exact, verbatim transcript and timing, in the standard layouts
training pipelines read directly.

```bash
# A short video → a dataset folder (LJSpeech metadata.csv + NeMo manifest.jsonl + wavs/)
hearsay dataset "https://youtu.be/VIDEO_ID" --out ./voice-data

# Any site yt-dlp supports — Dailymotion, SoundCloud, Twitch, ~1800 more
hearsay dataset "https://www.dailymotion.com/video/VIDEO_ID" --out ./voice-data

# A whole playlist / channel / podcast feed → one merged dataset
hearsay dataset "https://example.com/feed.xml" --out ./speech-data

# 16 kHz mono for ASR, custom clip length
hearsay dataset talk.mp3 --sample-rate 16000 --segment-min 2 --segment-max 12

# A HuggingFace audiofolder index (on its own — see the note below)
hearsay dataset talk.mp3 --format hf --format jsonl
```

You get a portable folder, ready to point a trainer at:

```text
voice-data/
  wavs/VIDEO_ID_0001.wav …     # mono 16-bit PCM, cut on sentence/pause boundaries, never mid-word
  metadata.csv                  # LJSpeech: id|text|text   (Coqui / Piper read this directly)
  manifest.jsonl                # NeMo / ESPnet: {"audio_filepath","duration","text","offset"}
  metadata.jsonl                # (with --format hf) HuggingFace audiofolder index
  dataset_card.md               # provenance, counts, language + a rights/consent note
  dropped.jsonl                 # every filtered-out clip, with the reason
```

- **Any source yt-dlp reaches.** Metadata and audio both come from yt-dlp, so a
  Dailymotion, SoundCloud or Twitch link works exactly like a YouTube one. A playlist
  or feed merges into a single dataset.
- **Word-accurate, click-free cuts.** Clips are sliced on word-level timestamps
  (faster-whisper `word_timestamps`, or Parakeet on Apple Silicon), padded a little
  on each edge (`--pad`) and given a short fade so boundaries never click or clip a
  phoneme.
- **Pick a model that can align.** `auto` (Parakeet on Apple Silicon, else
  `whisper-small`) is right for training data. `--model tiny`/`base` are fine for
  *reading* a transcript but their word-alignment pass can omit an audible word,
  pairing a clip with text that is missing it — hearsay warns when you use one.
- **Language is detected, not assumed.** The script and speaking-rate filters follow
  the language transcription detected, so non-English and non-Latin sources build
  normally; pass `--lang` only to force one.
- **Low-resource languages: bring a model that speaks them.** Stock Whisper is not
  merely weak on some languages, it is unusable — its own FLEURS table scores Uzbek at
  **90% word error**, because it trained on *18 minutes* of Uzbek. `auto` opens
  `large-v3` for those ~50 languages rather than `small`, which helps, but the real fix
  is a fine-tune: `--model` accepts any CTranslate2 Whisper model, a Hugging Face id or
  a local path. Community Uzbek models reach single-digit error. hearsay warns when you
  are pointing stock Whisper at a language it cannot read.

  ```bash
  # Stock Whisper, Uzbek news:  "Iran xafsizli kushlariga madxiya, sadıklar koshıqı"
  # An Uzbek fine-tune:         "eron xavfsizlik kuchlariga madxiya, sodiqlar qo'shig'i"
  hearsay dataset "https://youtu.be/VIDEO_ID" --lang uz \
      --model AlexAnoshka/fast-whisper-uz-rubaistt-v2-medium-ct2
  ```

  Any Whisper fine-tune works once converted to CTranslate2
  (`ct2-transformers-converter --model <hf-id> --output_dir ./model-ct2`). Word
  timestamps survive conversion, so dataset mode slices normally.
- **`--format hf` wants its own folder.** HuggingFace `audiofolder` refuses a tree
  containing both a `metadata.csv` and a `metadata.jsonl`, so pair `hf` with `jsonl`
  rather than with the default `ljspeech` (hearsay warns if you mix them).
- **Quality filtering** (on by default) drops junk — too short/long, internal silence,
  wrong-script or odd speaking-rate text, repetition, low ASR confidence — and logs
  every drop with its reason to `dropped.jsonl`. `--no-filter` keeps everything.
- **Single-voice TTS from multi-speaker audio.** Install `hearsay[diarize]`, accept the
  [pyannote model](https://hf.co/pyannote/speaker-diarization-community-1) conditions and
  set `HF_TOKEN`, then `--dominant-speaker` (keep the main voice) or `--per-speaker` (one
  index per speaker). Without it, datasets are **mixed-speaker** (fine for STT) — and the
  card says so.
- **`--normalize`** loudness-normalizes each clip (two-pass EBU R128); builds over
  playlists/feeds are **resumable**.

> **Accuracy & rights.** Word boundaries from Whisper/Parakeet are good but not
> phonetically exact — clips are padded and snapped to pauses, and you should spot-check.
> **You are responsible** for the rights to any media you process and for voice consent
> (cloning a real person's voice may require it); extracting audio from YouTube may breach
> its Terms. hearsay is local and ships no datasets. *Informational, not legal advice* —
> see each generated `dataset_card.md`.

Want to see the output shape without running anything? There's a tiny committed
example under [`examples/dataset/`](examples/dataset/).

## 📄 Clean, timestamped markdown

Getting a transcript into a RAG pipeline usually means gluing together `yt-dlp`,
Whisper, and a pile of timestamp-wrangling scripts — and you still end up with one
line per caption fragment or an undifferentiated wall of text. hearsay does the whole
thing in one command, and the markdown is readable by a human *and* a model:

```bash
# YouTube → markdown via captions (fast — no download)
hearsay "https://www.youtube.com/watch?v=VIDEO_ID"

# Local audio/video → markdown (fast Parakeet on Apple Silicon, else CPU Whisper)
hearsay talk.mp3

# Force local transcription, pick an engine, also emit a JSON sidecar
hearsay "https://youtu.be/VIDEO_ID" --transcribe --model parakeet --json

# A podcast feed or YouTube playlist (list, or batch with --all)
hearsay "https://example.com/feed.xml" --all --limit 3 --output-dir ./out
```

The output: real paragraphs (not one line per caption), a `[hh:mm:ss]` timestamp on
each, and chapters as `##` sections (or ~5-minute windows when there are none):

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
`captions-auto`, `whisper-small`, `parakeet-tdt-0.6b-v3` — so a consumer can tell a
human transcript from a machine one. Pass `--json` for a sidecar matching the
[`Transcript` schema](docs/schema.json): metadata plus `chunks[]`, each with
`start_s`, `end_s`, `section`, and `text` — ready to embed. Music or a song? Add
`--no-vad` so the vocals aren't filtered out as "non-speech."

## Web UI

Prefer a browser? `hearsay web` starts a tiny local web UI — paste a URL or drop in a
file, pick a model, and watch clean markdown render live with copy, download, and a
history. Tick **Dataset** to build a [training dataset](#-build-ttsstt-training-datasets)
instead and download it as a `.zip`. It's a single self-contained page on the Python
standard library — **no extra dependencies** — bound to `127.0.0.1`, so nothing leaves
your machine.

```bash
hearsay web                      # → http://localhost:8756
hearsay web --port 9000          # custom port
hearsay web --host 0.0.0.0       # expose on your LAN (unauthenticated — careful)
```

<p align="center">
  <img src="demo/webui.gif" alt="hearsay web UI" width="80%">
</p>

Videos, playlists, podcast feeds and file uploads all go through the UI. Batch sources
are capped at the first 5 items in the browser (the whole build streams back in one
response) — use the CLI for a full playlist or a hundred-episode feed.

## Transcription engines

When a source has no captions (or you pass `--transcribe`), hearsay transcribes
locally with the fastest engine your machine has. `--model auto` (the default) picks:

| Engine | When | Speed | Notes |
| --- | --- | --- | --- |
| **Parakeet** (NVIDIA Parakeet-TDT on Apple MLX) | Apple Silicon + `parakeet` extra | ~24× realtime (M1 Pro) | `parakeet` is multilingual (25 European languages); `parakeet-en` is English-only |
| **Whisper** (faster-whisper, CPU int8) | everywhere else, or an explicit size | ~7× realtime | sizes `tiny`…`large-v3`; `large-v3` is the multilingual ceiling |

On Apple Silicon, Parakeet is about **3× faster** than `whisper-small` at comparable
accuracy. If the `parakeet` extra isn't installed, `auto` falls back to `whisper-small`
automatically — so hearsay behaves the same everywhere, just faster on a Mac. Models
download once (Whisper: tens of MB to ~1.5 GB; Parakeet v3: ~2.5 GB) and cache for
offline use.

> **Speech vs. music:** hearsay is tuned for spoken audio (podcasts, talks, interviews,
> meetings), where transcription is accurate. For music, pass `--no-vad` so the vocals
> aren't discarded — but expect a rough lyric transcript, since these are speech models.

## Give your agent ears

hearsay ships an [MCP](https://modelcontextprotocol.io) server so AI agents can ingest
media themselves. It exposes two tools — `ingest_url(url, transcribe?, lang?)` and
`ingest_file(path)` — that each return clean, timestamped markdown.

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

**Claude Desktop** — add to `claude_desktop_config.json` (Settings → Developer → Edit
Config; macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`).
If `hearsay` isn't on the host's PATH, use the absolute path (`which hearsay`), or
`"command": "python", "args": ["-m", "hearsay", "mcp"]`.

Server configuration (env vars, since MCP tool signatures are fixed):

| Variable | Default | Effect |
| --- | --- | --- |
| `HEARSAY_MODEL` | `auto` | `auto`, `parakeet`, `parakeet-en`, or a Whisper size (`tiny`…`large-v3`) |
| `HEARSAY_LANG` | _(unset)_ | Default language: English captions, else transcription auto-detect |
| `HEARSAY_VAD` | `1` | Voice-activity filter (Whisper); set `0` for music/songs |
| `HEARSAY_PARAKEET_MODEL` | _(unset)_ | Override the Parakeet MLX repo id (advanced) |

## How it compares

| | **hearsay** | DIY `yt-dlp` + Whisper | markitdown / docling |
| --- | --- | --- | --- |
| Input | video & **audio** | video & audio (you wire it) | documents (pdf/docx/pptx) |
| One command | ✅ | ❌ multi-step plumbing | ✅ (for docs) |
| **TTS/STT dataset export** | ✅ LJSpeech + NeMo + HF, filtered, diarizable | ✗ DIY plumbing | ✗ |
| Captions-first (no download) | ✅ | ✗ usually re-transcribes | n/a |
| Timestamps + paragraph grouping | ✅ readable | ✗ raw segments | n/a |
| Chapters → sections | ✅ | ✗ manual | n/a |
| Podcasts · playlists · batch | ✅ | ✗ manual | ✗ |
| Fast Apple-Silicon engine | ✅ Parakeet (MLX) | ✗ DIY | n/a |
| JSON sidecar for RAG | ✅ stable schema | ✗ manual | varies |
| Browser UI + MCP server | ✅ | ✗ | varies |

hearsay does **media**; document tools like
[markitdown](https://github.com/microsoft/markitdown) and
[docling](https://github.com/docling-project/docling) do **documents**. Use both.

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
  --pad S              Edge padding added to each side of a clip (default 0.1)
  --normalize          EBU R128 loudness-normalize each clip
  --no-filter          Keep every clip (skip the quality filters)
  --diarize            Label speakers (needs hearsay[diarize] + HF_TOKEN)
  --per-speaker        Diarize and emit a per-speaker index
  --dominant-speaker   Diarize and keep only the most-spoken speaker
  --model / --lang / --vad / --no-vad    Transcription (as above)
  --limit N            Batch: cap items from a playlist/feed

hearsay web            Run the local web UI (--host, --port)
hearsay mcp            Run the MCP stdio server
hearsay --version      Show the version
```

## Requirements

- **Python 3.11+**
- **ffmpeg** on your PATH. hearsay decodes most audio/video directly (faster-whisper
  bundles its own decoder), but ffmpeg is the safe baseline, slices dataset clips, and
  handles some yt-dlp format merges.

| OS | Install ffmpeg |
| --- | --- |
| macOS (Homebrew) | `brew install ffmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| Windows (winget) | `winget install Gyan.FFmpeg` |
| Windows (Chocolatey) | `choco install ffmpeg` |

The first transcription downloads the chosen model once, then caches it for offline use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](docs/good-first-issues.md). Changes are documented in
[CHANGELOG.md](CHANGELOG.md). hearsay does one thing well — turn media into clean
markdown and training-ready datasets — and aims to keep doing exactly that.

## License

[MIT](LICENSE)
