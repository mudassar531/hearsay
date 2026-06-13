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
