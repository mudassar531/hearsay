# Show HN launch draft

Plain, honest, demo-first. No marketing superlatives — lead with what it does
and a link to the gif, and let the output speak for itself.

---

## Show HN title

> Show HN: hearsay – turn any video or podcast into clean, timestamped markdown

(Alternate: `Show HN: hearsay – one command from a YouTube URL to LLM-ready markdown`)

---

## Show HN post (~150 words)

I kept writing the same throwaway script: `yt-dlp` to pull a transcript, Whisper
when there were no captions, then a mess of code to turn caption fragments into
readable paragraphs with timestamps so I could put them in a RAG pipeline.
hearsay is that script, done properly.

One command turns a YouTube video, podcast episode, or local recording into
clean, timestamped markdown:

    hearsay "https://www.youtube.com/watch?v=..."

It uses captions when they exist (fast, no download) and falls back to local
Whisper (faster-whisper, runs on CPU) when they don't. Chapters become sections;
text is grouped into real paragraphs, not one line per caption. `--json` gives
you a sidecar with a stable schema for embedding. It does podcasts and playlists
in batch, and ships an MCP server so an agent can ingest media itself.

It does media — pair it with markitdown/docling for documents. MIT licensed,
Python 3.11+. Feedback welcome, especially on the paragraph grouping.

GitHub: https://github.com/mudassar531/hearsay

---

## Tweet / X thread (4 tweets)

**1/**
hearsay: one command turns a YouTube video, podcast, or local recording into
clean, timestamped, LLM-ready markdown.

    hearsay "https://youtube.com/watch?v=..."

Captions when they exist, local Whisper when they don't. Open source, MIT.
[gif]

**2/**
The point is the output. Not one line per caption fragment, not a wall of text —
readable paragraphs with timestamps, and chapters turned into sections. The same
grouping runs whether the text came from captions or Whisper.

**3/**
Built for pipelines: `--json` writes a sidecar with a stable schema (chunks with
start/end/section/text) you can embed directly. It also does podcast feeds and
YouTube playlists in batch, writing a folder of markdown + JSON.

**4/**
And it ships an MCP server, so you can give an AI agent ears:

    claude mcp add hearsay -- hearsay mcp

Two tools: ingest_url and ingest_file, both return markdown.
Try it: https://github.com/mudassar531/hearsay
