# Changelog

All notable changes to hearsay are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0 — 2026-06-15

### What's new: dataset export mode

hearsay can now turn the same media into **machine-learning training datasets**
for **TTS and STT**, alongside the existing reading-oriented markdown/JSON output.
The new `hearsay dataset <SOURCE>` command (and a "Dataset" mode in the web UI)
slices audio into short clips on word-level timestamps — **never mid-word** — and
pairs each with its verbatim transcript and timing, in the layouts training
pipelines read directly. The markdown/JSON engine is unchanged.

### Added

- **`hearsay dataset <SOURCE>` command** — same source routing as `ingest`
  (local file, single video, playlist/channel, podcast feed). A playlist or feed
  merges into one combined dataset.
- **Standard dataset layouts** — LJSpeech (`metadata.csv`), NeMo/ESPnet
  (`manifest.jsonl`), and HuggingFace `audiofolder` (`metadata.jsonl`), selectable
  with repeatable `--format`. Every build also writes a `dataset_card.md`
  (provenance, counts, language, and a rights/consent note) and a `dropped.jsonl`.
- **Word-accurate slicing** — clips are cut on faster-whisper `word_timestamps`
  (or Parakeet on Apple Silicon), snapped to sentence/pause boundaries and bounded
  by `--segment-min`/`--segment-max`; a run too long to break is flagged, not split.
- **Quality filtering (on by default)** — drops too-short/long clips, internal
  silence, wrong-script or odd speaking-rate text, repetition, and low-confidence
  ASR; each drop is logged with its reason. `--no-filter` keeps everything.
  Opt-in `detect_clipping` reads the WAV (stdlib `wave` plus numpy, which
  faster-whisper already pulls in — no new dependency).
- **Optional speaker diarization** via the `hearsay[diarize]` extra (pyannote,
  HF-gated) — `--diarize` to tag speakers, `--dominant-speaker` for single-voice
  TTS from multi-speaker audio, or `--per-speaker` for a per-speaker index.
  Without it, datasets are mixed-speaker and the card says so.
- **`--normalize`** — two-pass, length-preserving EBU R128 loudness normalization
  (`loudnorm`).
- **Clean clip boundaries** — clips get a little edge padding (`--pad`, default
  100 ms each side, capturing onset/offset phonemes the ASR word-timestamps clip)
  plus a short in/out fade that removes the click/pop from cutting on a non-zero
  sample. The quality filters still see the unpadded word extent, so padding never
  changes which clips survive.
- **Resumable combined builds** — a `_state.json` fingerprint lets an interrupted
  playlist/feed build resume, reconciling orphaned WAVs against the manifest.
- **Web UI "Dataset" mode** — build a single source and download it as a zip
  (large/batch jobs are steered to the CLI).
- An example mini-dataset under [`examples/dataset/`](examples/dataset/).

### Changed

- `transcribe` can now emit word-level timestamps (`word_timestamps`), exposed as
  `TranscriptionResult.words`.
- Packaging metadata bumped to 0.3.0; keywords gained tts/stt/dataset/training-data.

### Notes

- Dataset export adds **no new required dependency** (audio is sliced via the
  ffmpeg that hearsay already requires; zips use the stdlib). Diarization is the
  only extra, and it is opt-in.
- Word boundaries from Whisper/Parakeet are good but not phonetically exact, and
  **you are responsible** for the rights to any media you process and for voice
  consent — see each generated `dataset_card.md`.

## 0.2.0 — 2026-06-14

Maintenance/modernization release of the 0.1.0 markdown engine — packaging refresh
and internal preparation (including word-level timestamp plumbing in `transcribe`).
No user-facing feature changes over 0.1.0; the dataset export mode landed afterward
and ships in 0.3.0.

## 0.1.0 — 2026-06-13

Initial release: YouTube / podcast / local-audio → clean, timestamped **markdown**
(captions-first, Whisper/Parakeet fallback), paragraph grouping, chapters →
sections, a stable JSON sidecar, a browser UI, and an MCP server.
