# Changelog

All notable changes to hearsay are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- **Words split by the tokenizer are rejoined without a stray space.** ASR tokenizers
  split inside words — Whisper emits Uzbek `qo'shig'i.` as `["qo'shig", "'i."]` — and
  mark a genuine word start with a leading space. hearsay stripped that space away and
  then rejoined with a blind one, writing `qo'shig 'i.`: two broken tokens in the
  transcript shipped alongside the audio. One fresh Uzbek video carried **98** such
  words. Every Uzbek okina and every French elision hit this. English is unaffected
  (Whisper keeps contractions whole), which is why it went unnoticed.

## 0.5.0 — 2026-08-26

### Added

- **Uzbek, and ~50 other low-resource languages, actually work.** `auto` used to open
  `whisper-small` for anything Parakeet could not read. On Uzbek that returns romanised
  approximations, and `medium` is worse still — it collapses into Khmer and Georgian
  glyphs. Only `large-v3` returns readable Uzbek, so `auto` now opens it for the
  languages where the smaller checkpoints do not merely lose accuracy but return the
  wrong thing. A ~3 GB download, once, and only for those languages.
- **`--model` accepts any CTranslate2 Whisper model** — a Hugging Face id or a local
  path — not just the built-in sizes. This is what actually makes a low-resource
  language usable: stock Whisper trained on *18 minutes* of Uzbek and scores ~90% WER on
  its own FLEURS benchmark, while a community fine-tune reaches single digits. Measured
  on the same Uzbek news clip: stock `large-v3` gives *"Iran xafsizli kushlariga
  madxiya, sadıklar koshıqı"*; an Uzbek fine-tune gives *"eron xavfsizlik kuchlariga
  madxiya, sodiqlar qo'shig'i"*. Word timestamps survive conversion, so dataset mode
  slices normally.
- **A warning when stock Whisper is pointed at a language it cannot read**, naming the
  fix. Clips would otherwise ship paired with text that is largely wrong, and no
  downstream filter can detect that.
- **`prompt=` on `transcribe_audio`** for callers who want to steer spelling or
  vocabulary. hearsay never sets one itself: seeding the decoder does pin the output
  script (unprompted Uzbek measured 38% Latin / 61% Cyrillic in a single clip, versus
  100% either way when seeded), but on audio the model cannot read, Whisper returns the
  seed *as the transcript* — large-v3 on a 20s Uzbek clip echoed it back verbatim and
  transcribed nothing. For a training set that is a clip paired with words nobody said.
- The web UI language picker gained Kazakh, Mongolian and Serbian.

## 0.4.0 — 2026-08-26

### Security

- **The web UI refuses to fetch private addresses.** It hands caller-supplied URLs to
  yt-dlp, which fetches them server-side; now that any URL is accepted, loopback,
  link-local, and RFC1918 targets are rejected so the page cannot be used to reach
  hosts inside the user's own network.

- **Fixed an SSRF in YouTube URL validation.** `extract_video_id` matched its
  patterns anywhere in the string, so any URL merely *containing*
  `youtu.be/<11 chars>` — e.g.
  `http://169.254.169.254/latest/meta-data/#youtu.be/dQw4w9WgXcQ` — passed
  validation and was handed to yt-dlp to fetch. The id is now located by parsing
  the URL and matching the real hostname. This is the gate the web UI relies on to
  decide whether a caller-supplied URL is fetched server-side.
- **The web UI now validates the `Host` header.** Without it, a public web page
  could re-point its own hostname at `127.0.0.1` (DNS rebinding) and talk to the
  local server as same-origin, reading every transcript it returned. Loopback names
  and the bound address are accepted; anything else gets a 403.

### Added

- **Any language, not just the ones Parakeet knows.** `auto` picks Parakeet on Apple
  Silicon, but Parakeet covers exactly 25 European languages and does not refuse the
  rest — it transliterates them into confident nonsense. An Urdu naat came back as
  fluent-looking Latin gibberish with no error anywhere. hearsay now identifies the
  language first (one window, smallest Whisper checkpoint) and routes anything Parakeet
  cannot read to Whisper. The probe only chooses the engine; Whisper re-detects during
  the real decode, because forcing a tiny model's guess turned a Cyrillic Uzbek bulletin
  into Arabic-script nonsense.
- **A language picker in the web UI.** It was a free-text box, so `urdu` (instead of
  `ur`) silently produced a wall of 100 codes under the hint "check the file is a valid
  audio/video file". Language is now a list, and the CLI answers a name with the code it
  meant: *Unknown language code: 'urdu'. Did you mean 'ur'?*
- **Output is a visible mode, not a checkbox.** The training dataset — the thing most
  people come here for — was an unticked checkbox next to VAD, easy to miss entirely.
  It is now a Markdown / Training dataset toggle that says what the .zip contains.

### Added

- **Any site yt-dlp supports, not just YouTube.** Metadata and audio always came from
  yt-dlp, which takes the URL verbatim — only the CLI/web routers were YouTube-shaped.
  An http(s) source is now tried as a podcast feed and, when it isn't one, handed to
  yt-dlp, so Dailymotion, SoundCloud, Twitch and ~1800 other sites build datasets and
  markdown. `build_dataset_from_youtube` is renamed `build_dataset_from_media_url`.
- **Playlists and feeds build datasets in the web UI**, merged into one training set
  (capped at the first 5 items, since the whole build streams back in one response).
- **An elapsed clock while a build runs.** The browser got no output until a build
  finished, so a slow source was indistinguishable from a hung server.
- **A favicon**, ending the 404 every page load logged.

### Fixed

- **Diarization works on MP3s again** (`--diarize`, `--per-speaker`,
  `--dominant-speaker`). pyannote reads fixed windows sized from the reported
  duration, and a compressed container decodes to a frame-quantized sample count
  that can fall short of the last window — so it raised `resulted in N samples
  instead of the expected M` on every podcast enclosure, the exact input the
  single-voice-TTS workflow is for. Audio is now decoded to a PCM WAV first.
- **`--out` no longer deletes the user's own audio.** Reconciliation swept every
  unreferenced `.wav` under `<out>/wavs/`, so pointing `--out` at a folder that
  already held recordings destroyed them. Only hearsay's own
  `<source_id>_NNNN.wav` clips are eligible for cleanup.
- **Non-English sources build normally.** The script/speaking-rate filters were
  pinned to English whenever `--lang` was omitted, so an Urdu, Arabic, Russian,
  Hindi or Chinese recording had every clip dropped as `non_target_script` — an
  empty dataset under a green success tick. `target_language` now defaults to the
  language transcription detected.
- **Widening the clip window works.** `--segment-min`/`--segment-max` reached the
  segmenter but not the duration filter, which kept its 1–15s defaults; e.g.
  `--segment-max 30` produced "0 clips" while reporting success. The filter now
  follows the segment window unless bounds are set explicitly.
- **`--segment-min` above `--segment-max` is rejected** instead of silently
  building a near-empty dataset and exiting 0.
- **MCP tools no longer block the event loop.** FastMCP invokes a sync tool
  directly in its async handler, so a multi-minute transcription stalled the whole
  stdio session — no other call, no keepalive, no cancellation. Both tools now run
  their work in a worker thread.
- **The MCP server reports hearsay's version** in `serverInfo` rather than the MCP
  SDK's.
- **YouTube audio downloads work again.** The locked `yt-dlp` (2026.6.9) could no
  longer fetch audio at all — YouTube now requires a GVS PO token, and every player
  client either 403'd or had its formats stripped. Dataset mode was therefore broken
  for every YouTube source. The floor is now 2026.8.19; this dependency needs to keep
  moving, because YouTube breaks it on its own schedule.
- **Out-of-range dataset options print a hearsay error** naming the flag, instead
  of a raw pydantic `ValidationError` traceback with a pydantic.dev URL.

### Added

- **A warning when the chosen model cannot align reliably.** `whisper-tiny` and
  `whisper-base` can omit an audible word during the `word_timestamps` pass,
  shipping a clip paired with a transcript missing it — silent audio/text
  misalignment that no downstream filter can detect.
- **A warning when `--format hf` is combined with `ljspeech`.** HuggingFace
  `audiofolder` refuses a tree holding both a `metadata.csv` and a
  `metadata.jsonl` (`Found metadata files with different extensions`), so the
  HuggingFace index was unloadable whenever it was requested alongside the default
  LJSpeech index.

### Changed

- `FilterConfig.target_language` now defaults to `None` ("follow the source's
  detected language") rather than `"en"`. Callers that want the previous behaviour
  should pass `target_language="en"` explicitly.
- The diarization tests no longer depend on `pyannote.audio` being absent from the
  environment, so `uv sync --all-extras` keeps the suite green.

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
