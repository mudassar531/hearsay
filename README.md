# hearsay

> crawl4ai for video & audio — one command turns any YouTube video, podcast episode, or local recording into clean, timestamped, LLM-ready markdown.

🚧 **Under construction.** The full README (install, quickstart, demo) lands in Phase 5.

## Requirements

- **Python 3.11+**
- **ffmpeg** — a documented system requirement. hearsay decodes most audio/video
  directly (faster-whisper bundles its own decoder), but ffmpeg is needed for some
  yt-dlp format merges and is the safe baseline for local-file transcription.

Install ffmpeg:

| OS | Command |
| --- | --- |
| macOS (Homebrew) | `brew install ffmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| Windows (winget) | `winget install Gyan.FFmpeg` |
| Windows (Chocolatey) | `choco install ffmpeg` |

Verify it is on your PATH with `ffmpeg -version`.

## Usage so far

```bash
# YouTube → markdown via captions (fast)
hearsay "https://www.youtube.com/watch?v=VIDEO_ID"

# Local audio/video → markdown via local Whisper transcription
hearsay path/to/recording.mp3

# Force Whisper on a YouTube URL, choosing a model size
hearsay "https://youtu.be/VIDEO_ID" --transcribe --model small
```

The first transcription downloads the chosen Whisper model once (tens of MB to
~1.5 GB depending on size), then caches it for offline use.

Podcasts and playlists:

```bash
# List a podcast feed's episodes, then ingest a selection into a folder
hearsay "https://example.com/feed.xml"
hearsay "https://example.com/feed.xml" --all --limit 3 --json --output-dir ./out

# Ingest a YouTube playlist (same flags); --json writes a sidecar per item
hearsay "https://www.youtube.com/playlist?list=PLAYLIST_ID" --latest
```

## Give your agent ears

hearsay ships an [MCP](https://modelcontextprotocol.io) server so AI agents can
ingest media themselves. It exposes two tools — `ingest_url(url, transcribe?, lang?)`
and `ingest_file(path)` — that each return clean, timestamped markdown.

Install the optional extra and confirm it runs:

```bash
uv tool install "hearsay[mcp]"   # or: pipx install "hearsay[mcp]"
hearsay mcp                      # starts a stdio MCP server (Ctrl-C to stop)
```

**Claude Code** — register the server (project or user scope):

```bash
claude mcp add hearsay -- hearsay mcp
```

or add it to `.mcp.json` / `~/.claude.json`:

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
Edit Config; on macOS it lives at `~/Library/Application Support/Claude/`, on
Windows at `%APPDATA%\Claude\`):

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

If `hearsay` is not on the host's PATH, use the absolute path to the executable
(find it with `which hearsay`), or `"command": "python"`, `"args": ["-m", "hearsay", "mcp"]`.

Server configuration (env vars, since MCP tool signatures are fixed):

| Variable | Default | Effect |
| --- | --- | --- |
| `HEARSAY_MODEL` | `small` | Whisper model size for transcription (`tiny`…`large-v3`) |
| `HEARSAY_LANG` | _(unset)_ | Default language: English captions, else Whisper auto-detect |
