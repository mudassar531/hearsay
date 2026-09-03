# Changelog

All notable changes to hearsay are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.8.0 — 2026-09-03

A dataset that cannot prove itself is a guess. 0.8.0 adds `hearsay verify`: the
ten-language sweep behind the 0.7.0 fixes, run by a command on the produced files, so
every dataset ships with its own evidence. Around it, a review of the whole codebase
fixed six silent-output bugs — several in exactly the languages hearsay is measured on.

### Added

- **`hearsay verify DIR`** re-transcribes a random sample of clips and diffs each
  against its own row *and* against another clip's row; the gap is the pairing signal
  (a manifest shifted by one row scores ~0, a correct one +0.3 or more) and the self
  score alone is the accuracy signal, so a perfect pairing through a model that cannot
  read the language is told apart from a broken pairing. It also measures the share of
  clips in the language's real script, the share cut through speech rather than on
  silence (first/last 25 ms energy against the body), and eight structural invariants.
  Writes `verification.md` + `verification.json`; the verdict is trainable / marginal /
  not trainable with every reason spelled out, and the exit code follows it (0/1/2).
  Reads any LJSpeech, NeMo or HuggingFace `audiofolder` tree. **`hearsay dataset
  --verify`** runs it straight after a build, with the same model.
- **YouTube channels** are batch sources: `/@handle`, `/channel/UC…`, `/c/Name`,
  `/user/Name`, with or without a `videos`/`streams`/`shorts` tab.
- **`--device auto|cpu|cuda`** (and `HEARSAY_DEVICE` for the MCP server): Whisper runs on
  CUDA in float16 when ctranslate2 sees a GPU. It was pinned to the CPU.
- **`--cookies-from-browser chrome|firefox|safari|edge`** on every command, and
  `HEARSAY_YTDLP_ARGS` for extra yt-dlp flags in the web UI and MCP server. YouTube's
  "Sign in to confirm you're not a bot" now names this flag instead of "update yt-dlp".
- The MCP `ingest_url` tool accepts any site yt-dlp supports, as the CLI and web UI did.
- The dataset card records `transcription_method`, and a combined (playlist/feed) card
  names the language its sources were detected as instead of `und` whenever they agree.

### Fixed

- **Sentence marks beyond ASCII.** Both the paragraph grouper and the clip segmenter
  tested for `.!?` and `,;:` only, so an Urdu `۔`, a Hindi `।`, an Arabic `؟` or a
  Chinese `。` never counted as a boundary and those languages were cut on pauses and
  duration alone. One shared table (`hearsay.punctuation`) covers Latin, Arabic-script,
  Indic, CJK, Ethiopic, Myanmar, Khmer, Tibetan and Armenian marks.
- **Resume state could attach the wrong episode's clips.** Source ids were an ASCII-only
  slug, so every Urdu, Bengali or Chinese episode title became `clip`, numbered by feed
  order; a newly published episode shifted the numbers and a resumed build re-used
  another episode's cached clips. Ids now carry a short digest of the episode's guid
  (or of the title itself when the slug lost letters), stable across runs and distinct
  across episodes. **A feed dataset started under 0.7 rebuilds once under 0.8.** YouTube
  video ids are unchanged. The markdown batch slug also kept dropping the vowel signs
  of abugida titles (`বাংলা` → `বল`); it keeps them.
- **Channel URLs** fell through to the single-video path, where yt-dlp tried to dump
  every video, hit the 60 s timeout and blamed the network.
- **The language probe only ran on Apple Silicon.** It was gated on the parakeet extra,
  but it also decides whether a low-resource language gets `large-v3` — so on Linux and
  Windows a Hindi or Bengali file with no `--lang` went through `whisper-small` while
  the README promised otherwise. It now runs wherever `auto` runs, and decodes only its
  30 s window instead of the whole file (700 MB of float32 for a three-hour podcast).
- **Edge padding bled into the neighbouring word.** Padding was applied per clip with no
  knowledge of the next one, so at a cut with a gap shorter than twice the pad adjacent
  WAVs overlapped and each carried a phoneme its transcript did not contain. Each side
  now gets at most half the silence to its neighbour.
- **The MCP server could only ingest YouTube.** A Dailymotion URL went down the captions
  path and surfaced "video unavailable" instead of transcribing.
- The web UI sent the dataset zip base64-encoded inside JSON (an hour of audio became a
  213 MB string the browser had to decode); it is now a file download with the summary
  in a header. A malformed JSON body is a 400, not a 500.
- `--lang` is validated before captions are fetched (`--lang urdu` used to quietly hand
  back whatever track existed), and a fallback to another caption language is announced.
- Whisper reports `und` rather than `en` when it could not name the language.

### Changed

- Models are opened once per process (a fifty-video playlist loaded the same checkpoint
  fifty times; the web UI and MCP server once per request), and one lock serialises
  transcription so two browser tabs cannot load two copies of `large-v3` into 16 GB.
- Dataset mode turns off `condition_on_previous_text` and sets a hallucination-silence
  threshold, so a music bed no longer becomes "Thank you for watching" repeated down the
  file. The markdown path keeps Whisper's defaults.
- CI runs on Python 3.13 as well as 3.11 and 3.12.

## 0.7.0 — 2026-08-27

Ten languages — Mandarin, Hindi, Spanish, Bengali, Japanese, Russian, Turkish,
Vietnamese, Swahili and Korean — were built into datasets from real YouTube audio
and checked end to end: structure, audio/text pairing, script authenticity, clip
boundaries, and loading the result in HuggingFace `datasets`. Five defects came
out of it, all of which produced *silently wrong* data rather than an error.

### Fixed

- **`auto` handed Spanish and French to an engine that returns English.** Parakeet's
  25-language list came from its model card, not from measurement. On real audio a
  Spanish podcast came back as fluent English — 0 Spanish function words against 89
  English ones, and 0.02 character similarity to the same audio through `large-v3`.
  A second Spanish source and a French one failed identically; German, Italian and
  Russian were correct. `parakeet-mlx` takes **no language argument**, so `--lang es`
  never reached the decoder — it only relabelled the output, stamping English text
  with `language: "es"` in the dataset card. Both languages are Latin-script, so the
  wrong-script filter could not see it either. `auto` now uses Parakeet only for
  languages measured on real audio, and an unidentified language goes to Whisper
  rather than to an engine that can be neither steered nor checked.
- **Bengali had no target script, so nothing checked what came back.** A missing
  entry in the script map does not fall back — it disables the wrong-script *and*
  char-rate filters entirely. A real Bengali news bulletin built with no flags
  shipped clips whose text was Telugu script and English (`independent news desk`),
  0% Bengali, under `language: "bn"`, with zero `non_target_script` drops. The gap
  was two levels deep: there was no Bengali block in the script table either, so
  Bengali text classified as "other". Adds the missing scripts and maps the
  languages that have exactly one. Uzbek stays unmapped on purpose — two scripts.
- **Six languages Whisper's own table calls unusable were not warned about.** The
  list cited FLEURS but was assembled by hand; parsing all 82 languages out of the
  paper found Georgian 105.0, **Bengali 104.1**, Gujarati 102.7, Punjabi 102.4,
  Malayalam 100.7 and Telugu 99.0 missing — all at or above the ~90% word error the
  list is defined by, i.e. worse than transcribing nothing. `auto` also now opens
  `large-v3` for Hindi, which was losing nearly half its accuracy to `whisper-small`
  (38.4 vs 21.5), and for Bengali.
- **A tokenizer-split number read as a sentence end.** Whisper emits `19.8` as
  `["19.", "8"]`, and `"19."` matched the full-stop test. On a Mandarin news
  bulletin the *only* three tokens in 977 that scored as sentence ends were `1.`,
  `19.` and `5.` — every one half of a split decimal — and the sentence bonus then
  pulled the clip boundary into the middle of the number: one clip ended `…同比多19.`
  and the next began `8万亩`. `joins_left` already carried the tokenizer's answer;
  nothing consulted it when choosing a cut. It does now.
- **A playlist or feed card claimed English it never detected.** Both combined paths
  defaulted `language` to `"en"`, which goes straight into the HuggingFace YAML front
  matter, so a Mandarin playlist built without `--lang` asserted English. Uses
  `"und"` — the value hearsay already uses for a language it cannot name.

### Changed

- The README's per-language guidance is now measured rather than asserted: the
  fine-tune table covers all ten languages with Whisper's own FLEURS rates, Parakeet
  is no longer described as broadly multilingual, and Bengali joins Uzbek and Pashto
  as unusable without a fine-tune.

## 0.6.0 — 2026-08-27

### Added

- **A language fine-tune works in the web UI**, not just on the CLI. The Model box takes
  a Hugging Face id or a local path alongside the built-in sizes. For Uzbek and Pashto
  that is the difference between usable data and confident nonsense, so the browser
  previously could not build a usable dataset in those languages at all.
- **Per-language guidance in the README**, from Whisper's own FLEURS table: Arabic
  ~16% WER and Urdu ~22.6% are ordinary, Uzbek ~90% and Pashto ~93.7% are not.

### Fixed

- **Urdu, Pashto and Sindhi had no target script**, so the `non_target_script` filter was
  silently disabled for them — for exactly the languages most at risk. Measured on a real
  Urdu podcast: Whisper detects the audio as Hindi (p=0.91) and emits Devanagari, so a
  whole dataset can come back in the wrong script while the card still reads `language:
  "ur"`. Uzbek is deliberately left unmapped, being written in two scripts.
- **The low-resource warning was fired at languages that are fine.** Urdu and Arabic were
  told "stock Whisper barely saw this language, ~90% word error, go find a fine-tune",
  which is alarming and false. Choosing a model *size* and declaring a language
  *unusable* are now separate sets, and the warning names the real failure: for Pashto
  stock Whisper does not fail loudly, it transliterates into Arabic/Dari, so the text
  reads fluently and is the wrong language.
- **A wrong `--model` id blamed the network.** With any Hugging Face id accepted, a typo
  is the likeliest mistake; the hint now distinguishes an unknown id, a gated repo and a
  real network failure, and no longer buries the cause under a request id.

## 0.5.1 — 2026-08-26

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
