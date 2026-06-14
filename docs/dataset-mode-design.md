# Dataset Export Mode (TTS/STT) — Design Memo

> **Status:** proposed (Step Zero). Awaiting approval before Phase D1.
> **Scope:** a second, additive output mode that turns media into machine-learning
> **training datasets** for TTS and STT — alongside (never replacing) the existing
> readable-markdown mode.
> **This memo is informational, not legal advice.** See [§9 Legal & ethical](#9-legal--ethical-use).

Every external fact below was researched against current (2025–2026) primary
sources and then adversarially fact-checked; corrections from that pass are folded
in and flagged **[verified]** / **[corrected]** where they changed a decision.

---

## 0. The goal, in one paragraph

A user points hearsay at a YouTube video, a whole playlist/channel, a podcast
feed, or local files, and gets back a clean, standard, ready-to-train dataset:
the audio **sliced into short segments** (≈1–15 s, cut on sentence/pause
boundaries, **never mid-word**), each paired with its **exact verbatim transcript**
and **precise timestamps**, written in a **standard layout** (an audio-clip folder
plus manifest files that real TTS/STT pipelines read directly), with **quality
filtering** that drops junk (music, silence, overlapping speech, wrong language,
too-short/long). This is a new `hearsay dataset` subcommand and a Web-UI path; the
markdown/JSON engine is untouched.

---

## 1. What mainstream open TTS/STT datasets actually look like

We surveyed the formats real training pipelines consume. Concrete, verified
structures:

| Dataset / convention | Index file | Audio | Notes |
| --- | --- | --- | --- |
| **LJSpeech-1.1** | `metadata.csv`, **pipe-delimited, 3 cols, no header**: `id\|transcription\|normalized_transcription` | `wavs/<id>.wav`, **16-bit PCM, mono, 22050 Hz** | The de-facto single-speaker **TTS** layout. Public domain. **[verified]** |
| **Mozilla Common Voice** | per-split TSVs (`validated/train/dev/test/other/invalidated.tsv`), cols `client_id, path, sentence, …` | `clips/*.mp3`, 48 kHz | CC0. Moved off HF to Mozilla Data Collective (Oct 2025). Version-dependent columns → a *moving target*, so **not** a default. **[verified]** |
| **NeMo / ESPnet / Kaldi JSONL manifest** | `manifest.jsonl`, one JSON object per line: `{"audio_filepath", "duration", "text"}` | shared `wavs/` | The de-facto **STT/ASR** manifest. `duration` is **seconds (float)**; optional `offset` (seconds). `text` required for *training*, optional for inference. **[verified]** |
| **HuggingFace `audiofolder`** | `metadata.csv` / `metadata.jsonl` with a **mandatory `file_name` column** + arbitrary feature cols (e.g. `transcription`) | files referenced relative to the metadata file; splits by sub-directory | One-step `load_dataset("audiofolder", …)`. `file_name` (not `path`/`filename`) is the column people get wrong. |
| **Coqui TTS** | reads LJSpeech `metadata.csv` via its `ljspeech` formatter | `wavs/` | **Reads `cols[2]`** (the 3rd/normalized column) as the text. **[corrected]** |
| **Piper TTS** | `--dataset-format ljspeech`: `id\|text` (single-speaker) or `id\|speaker\|text` (multi) | mono 22050 WAV | A literal `id\|transcription\|normalized` line risks Piper reading the **middle** field as a *speaker id*. **[corrected]** |

### Decision — the two default formats

hearsay exports **two indices over one shared `wavs/` tree**:

1. **TTS — LJSpeech-style** `metadata.csv` (pipe-delimited, no header) + `wavs/`.
   To satisfy **both** Coqui (reads col 3) and Piper (reads the last col) without
   tripping Piper's multi-speaker parse, we emit **`id|text|text`** — the verbatim
   transcript duplicated into the transcription and normalized columns. **[corrected:
   the safe portable form is `id|text|text`, not `id|transcription|normalized`.]**
   Audio: mono 16-bit PCM WAV at **22050 Hz** (LJSpeech canonical, Coqui/Piper default).

2. **STT — NeMo-style** `manifest.jsonl`, one object per line:
   `{"audio_filepath": "wavs/<id>.wav", "duration": <sec float>, "text": "<verbatim>"}`,
   with `offset: 0.0` included for forward-compat with segmented clips, and
   `audio_filepath` kept **relative to the manifest** for portability.

3. **Optional third target** (`--format` includes `hf`): a HuggingFace
   `audiofolder` `metadata.csv` (`file_name,transcription`) — a trivial
   column-rename of (1) that unlocks the whole `datasets` ecosystem.

`--format ljspeech|jsonl|hf|all` (default `all` of ljspeech+jsonl). `--sample-rate`
defaults to **22050** (TTS-canonical); document **16000** mono as the typical
ASR/Whisper rate and let users set it. A single shared tree can't be optimal for
both rates at once, so sample rate is an explicit flag, not a guess.

> **Why not Common Voice as a default:** its column set is version-dependent and it
> just migrated distribution channels (Oct 2025) — emulating it precisely is a
> moving target with little upside over LJSpeech+JSONL.

---

## 2. The alignment problem (the crux)

Caption timestamps are **cue-level and loose** — `youtube-transcript-api` returns
only `text/start/duration` per multi-second cue, with **no word timing**. Slicing
on those would routinely cut mid-word. **Captions are text, not timing-of-record.**
**[verified]**

So dataset mode needs **word-level timestamps**, sourced like this:

### Default path (zero new dependencies) — `faster-whisper word_timestamps=True`

faster-whisper (already core, `>=1.2.1`, MIT) returns per-word
`Word(word, start, end, probability)` when you pass `word_timestamps=True`,
computed via DTW over Whisper's cross-attention. **[verified]** No torch, no extra
deps. This is the **default** word source for dataset mode.

Caveat: these boundaries are **coarse** — ~100–400 ms jitter across model sizes,
with occasional **inverted spans (`start > end`)**, especially segment-initial and
non-English. **[verified]** So the slicer is **defensive by design** (see §3):
repair inverted spans, snap cuts to inter-word silence, and pad clip edges.

### Apple-Silicon path — reuse Parakeet token timings (zero new deps)

Where Parakeet (`parakeet-mlx`) is active, it already emits token-level timings.
`AlignedToken(id, text, start, duration, confidence, end)`; **word boundaries are
marked by a literal leading space** (`token.text.startswith(' ')`, **not** the
SentencePiece `▁`). **[corrected]** We merge subword tokens into words (new word when a token's
text starts with a space; `word.start` = first sub-token start, `word.end` = last
sub-token end).

### Engine-agnostic `Word` adapter

Field names differ (whisper `word`/`probability` vs Parakeet `text`/`confidence`,
default `1.0`), so D1 introduces one normalized `Word` model and per-engine
adapters. Anything assuming a single shape silently breaks the other engine.

### Better-but-optional — forced alignment (`hearsay[align]`)

For phonetically tight boundaries, an opt-in extra re-aligns transcript text to the
waveform (CTC forced alignment). Measured boundary error (arXiv:2406.19363,
Table 3): **MFA ~22 ms < WhisperX ~34 ms < MMS ~41–69 ms < raw Whisper DTW ~500 ms**.
**[verified]** Candidates: **WhisperX** (BSD-2, reuses our faster-whisper backend),
`torchaudio` **MMS_FA**, or `ctc-forced-aligner`. All pull torch — strictly opt-in.

> **License guardrail for redistributable datasets.** Default word sources are
> commercial-clean: faster-whisper/Whisper weights are **MIT**, Parakeet-TDT v3 is
> **CC-BY-4.0** (commercial OK). But common forced-alignment **default models are
> non-commercial**: `ctc-forced-aligner`'s default (`mms-300m-1130-forced-aligner`)
> and `torchaudio` MMS_FA weights are **CC-BY-NC-4.0**, and **even WhisperX's bundled
> non-English (fr/de/es/it VOXPOPULI) alignment models are CC-BY-NC-4.0** — only its
> English model (WAV2VEC2_ASR_BASE_960H) is MIT. **[corrected/verified]** The `align`
> extra will surface this and prefer commercial-clean models.

**Excluded:** MFA (Kaldi/conda-only, breaks pip/uv install) and `aeneas`
(AGPL-3.0, sentence-level) — mentioned only as external power-user workflows.

### Decision

Default = faster-whisper `word_timestamps=True` (Whisper) / merged Parakeet tokens
(Apple Silicon), no new deps. Dataset mode **always transcribes** for word timing
(it does not use the caption fast-path); a future `--align` enhancement (in
`hearsay[align]`) can sharpen boundaries or force-align official caption text.

---

## 3. Audio handling (cutting & normalization)

**Cut by re-encoding, never `-c copy`.** Stream-copy snaps cuts to the nearest
keyframe (not sample-accurate); for WAV PCM, re-encoding is **lossless** anyway, so
there's no quality cost. **[verified]**

**Seeking — input-seek is the right call. [corrected]** Since FFmpeg 2.1, *input*
seeking (`-ss` before `-i`) is **also frame-accurate when transcoding** and is
**much faster** on long sources (no decode-and-discard from 0). Since our segments
come from potentially hours-long podcasts/lectures and we always re-encode, we use
input-seek with a **duration** (timeline resets to 0 at the seek point, so use `-t`,
not absolute `-to`):

```bash
ffmpeg -hide_banner -loglevel error -nostdin -y \
  -ss <START> -i <SOURCE> -t <END-START> \
  -ar <22050|16000> -ac 1 -c:a pcm_s16le <wavs/ID.wav>
```

(`-ac 1` mono, `pcm_s16le` 16-bit PCM.) We probe/learn durations exactly from the
known `[start,end]` and `ffprobe`.

**Target profiles:** TTS = 22050 Hz / mono / 16-bit; STT = 16000 Hz / mono / 16-bit.
**[verified]** Warn when the source sample rate is *below* the target (upsampling
adds no real high-frequency content and can mislead TTS training).

**Silence trimming:** prefer trimming to the **first word's start / last word's end
+ a small pad** (default ~100 ms) using the word timestamps — controllable and
reproducible — over the heuristic `silenceremove` filter (offered as a fallback).

**Loudness normalization: optional, default off.** When `--normalize` is set, run
two-pass `loudnorm` (EBU R128; pass 1 `print_format=json` to measure
`input_i/lra/tp/thresh`, pass 2 with `measured_*` + `linear=true`), targeting
**I=-23 LUFS, TP=-1.5 dBTP, LRA=7** (offer **I=-16** streaming-style).
ffmpeg defaults are I=-24/TP=-2/LRA=7; EBU R128 max true peak is -1 dBTP, so -1.5 is
a conservative ceiling. **[verified]** Default-off rationale **[corrected]**: the
common targets don't need it — Whisper does only *feature-level* (mel) normalization,
and Coqui defaults to `do_sound_norm=False` — and some recipes do their own; two-pass
also ~doubles per-clip ffmpeg work and `loudnorm` upsamples to 192 kHz (so we always
pin `-ar`).

**Clip naming:** primary `<source_id>_<index4>.wav` (e.g. `dQw4w9WgXcQ_0007.wav`) —
a clean, sortable `metadata.csv` join key matching Coqui's expectations; optional
`<source_id>-<start_ms>-<end_ms>.wav` for self-describing, segmentation-stable names.
`source_id` = YouTube id, else a slug + short content hash (collision-free across
re-runs for local/RSS).

**Tooling:** call **ffmpeg/ffprobe via `subprocess`** (explicit arg list, no
`shell=True`) — matching the existing `yt-dlp` pattern in `youtube.py`; ffmpeg is
already a documented PATH requirement. **No pydub/librosa.** On startup of a dataset
build we verify the ffmpeg build has `loudnorm`/`silenceremove` and fail with an
actionable hint if not.

---

## 4. Quality filters & default thresholds

Two tiers. **Tier 1 is default-on and needs no heavy deps** — it uses only audio
metadata and the transcript hearsay already produces:

| Filter | Default | Rationale |
| --- | --- | --- |
| **Duration** | drop `< 1.0 s` or `> 15.0 s` | Standard TTS clip range; very short = too little context, very long strains alignment. |
| **Edge silence** | trim leading/trailing (~30 dB below peak), pad ~100 ms; drop if trimming leaves no speech | Reproducible via word timestamps. |
| **Internal silence** | drop if an internal gap `> ~2.0 s` | Often a join of two utterances; faster-whisper's own VAD uses `min_silence=2000 ms`. |
| **ASR confidence** (free from Whisper) | drop if `avg_logprob < -1.0` **or** `no_speech_prob > 0.6` **or** `compression_ratio > 2.4` | These are Whisper's own decoding-default thresholds — good provenance. **[verified]** Aggregate to clip level by worst-segment (pinned in D3). |
| **Language match** | drop if `detected_language != target` **and** `language_probability >= 0.5` | i.e. "top language is confidently non-target." **[corrected wording]** Gate on min duration (LID samples first 30 s, unreliable < 3–5 s). |
| **Text sanity** | drop empty/whitespace; drop if chars/sec outside **~5–25**; drop non-target-script chars | chars/sec is language-dependent — **per-language, disabled for CJK**. |

**Tier 2 — optional, off by default (gated behind installed deps):**

- **Silero VAD** to drop music/non-speech. **Key finding:** Silero ships an ONNX
  model and **`onnxruntime` is already a core transitive dep** (via faster-whisper),
  so a **torch-free** Silero path is feasible; `webrtcvad-wheels` (MIT, no deps) is a
  lighter heuristic fallback. **[verified]**
- **SNR** estimate (drop very low SNR), **clipping** detection (true-peak ≥ 0 dBFS),
  **loudness** normalization (a normalize step, not a drop).

**Every dropped clip is logged** as a structured record and summarized by reason:

```json
{"clip": "dQw4w9WgXcQ_0042.wav", "filter": "avg_logprob", "value": -1.34, "threshold": -1.0, "action": "dropped"}
```

The build prints `kept N · dropped M` with a per-reason breakdown; all thresholds
are config-overridable. We do **not** adopt perfectionist corpora bars as defaults
(e.g. Hi-Fi TTS used **zero-WER** inclusion and **SNR ≥ 32 dB** **[corrected: 32, not
40]**) — they'd empty out real-world audio. (If we ever wrap NeMo SDP's
`DropHighCER/WER`, note its thresholds are on a **0–100 scale**, so 10–20% = `10`–`20`,
**not** `0.10`–`0.20`. **[corrected]**)

---

## 5. The speaker problem (diarization)

Whisper/Parakeet produce **one unlabeled transcript**. Concatenating multi-speaker
audio into a "single speaker" TTS set **splices every voice into one target** —
fatal for voice cloning. So: **clean single-voice TTS from multi-speaker audio
genuinely needs diarization.** Mixed-speaker output is fine for **STT**, not for
single-voice **TTS**. We state this honestly in output and docs.

**Decision — optional `hearsay[diarize]` extra that degrades cleanly:**

- **Without it (default):** emit a mixed-speaker dataset and warn —
  *"diarization not installed: this dataset contains MIXED speakers; fine for STT,
  not for single-voice TTS. Install `hearsay[diarize]` and set `HF_TOKEN` to enable
  per-speaker export."*
- **With it:** `--per-speaker` (one sub-dataset/manifest per `SPEAKER_xx`) and
  `--dominant-speaker` (keep only the speaker with the most total speech — the
  podcast-host case).

**Engine:** **pyannote.audio** (MIT code, torch ≥ 2.8) — default model
`speaker-diarization-3.1` (MIT model) with opt-in `community-1` (CC-BY-4.0,
pyannote 4.x, better offline + timestamp reconciliation); or WhisperX (BSD-2),
which assigns each word to the speaker of **maximum temporal overlap** (not
midpoint — midpoint is only the no-overlap fallback). **[corrected]**

**HF-token friction (important).** pyannote models are **gated on HuggingFace**:
the user must (1) accept conditions on **both** `pyannote/segmentation-3.0` **and**
`pyannote/speaker-diarization-3.1` (or `community-1`), and (2) provide an HF read
token. **[verified]** hearsay reads `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` (already
present in our gitignored `local.env`) or `--hf-token`, and on auth failure prints
the exact remediation with **both** URLs. **Avoid NeMo Sortformer as a default**
(CC-BY-NC-4.0, max 4 speakers). The extra adds ~2–3 GB (torch); GPU recommended.

This is the home for the diarization idea formerly parked in `IDEAS.md`.

---

## 6. New dependencies

**Core install gains nothing.** Verified already-present transitive deps we can
build on with zero additions: **numpy** (via `ctranslate2`), **PyAV/`av`** (via
faster-whisper, ffmpeg-bundled wheels — can decode/probe), and **`onnxruntime`**
(via faster-whisper — enables torch-free Silero VAD). **[verified against `uv.lock`]**
ffmpeg/ffprobe are an existing system requirement.

| Extra | Packages | Weight / license | Purpose |
| --- | --- | --- | --- |
| (core) | — | — | slicing via ffmpeg subprocess; duration via ffprobe; arrays via numpy; WAV via stdlib `wave` |
| **`hearsay[dataset]`** *(optional, light, no torch)* | `soundfile>=0.14` (BSD-3, bundles libsndfile; +`cffi`,`typing-extensions`), `webrtcvad-wheels>=2.0.14` (MIT, no deps) | a few MB | ergonomic WAV/SNR/clipping; heuristic VAD |
| **`hearsay[align]`** *(optional, heavy, torch)* | `whisperx` (BSD-2) or `torchaudio` | torch wheel ~532 MB (Linux, CUDA-bundled) | sharper forced-alignment word boundaries |
| **`hearsay[diarize]`** *(optional, heavy, torch, gated models)* | `pyannote.audio>=4.0` (MIT code) | ~2–3 GB | per-speaker / dominant-speaker TTS |

We follow the existing `mcp`/`parakeet` optional-extra pattern. **Excluded:**
`pydub` (unmaintained since 2021, redundant with ffmpeg/PyAV), MFA, `aeneas`. Each
extra documents a CPU-only torch install path (`download.pytorch.org/whl/cpu`).

> Tentative leaning: the default dataset path may need **no extra at all** (ffmpeg +
> ffprobe + stdlib `wave` + numpy). `soundfile` only buys ergonomics for Tier-2
> acoustic filters. We'll confirm during D2/D3 whether `hearsay[dataset]` is even
> needed or whether the light filters ride entirely on already-present deps. Each
> dependency that lands is justified in `DECISIONS.md`.

---

## 7. CLI / front-end surface

**Decision: a new `hearsay dataset <SOURCE>` subcommand**, not a `--dataset` flag on
`ingest`. Dataset mode has a large, distinct option surface and a fundamentally
different output (a folder tree + manifests, not one markdown file); bolting ~10
dataset-only flags onto `ingest` would muddy that command and its `--help`. A
subcommand keeps `ingest` pristine, separates the two mental models cleanly, and
reuses all shared plumbing (metadata, transcription, batch, feeds, youtube). It
joins the existing `mcp`/`web` subcommands under the current `DefaultCommandGroup`.
(Justification logged in `DECISIONS.md`.)

```text
hearsay dataset <SOURCE> [options]   SOURCE = YouTube video/playlist URL, podcast RSS, or local file/glob

  --out DIR                 dataset output directory (default ./hearsay-dataset)
  --format ljspeech|jsonl|hf|all     (default: ljspeech+jsonl)
  --sample-rate 22050|16000|N        (default 22050; 16000 typical for ASR)
  --segment-min SECONDS     (default 1.0)
  --segment-max SECONDS     (default 15.0)
  --model auto|parakeet|tiny..large-v3   (reuses existing engine selection)
  --lang CODE               target language (also drives the language-match filter)
  --normalize               opt-in two-pass EBU R128 loudness normalization (default off)
  --no-filter / --filter-*  toggle / tune individual quality filters
  --diarize                 require hearsay[diarize]; label speakers
  --per-speaker             one sub-dataset per speaker (implies --diarize)
  --dominant-speaker        keep only the most-spoken speaker (implies --diarize)
  --vad                     enable optional Silero/webrtc VAD non-speech drop
```

**Web UI:** a "Dataset mode" path — choose dataset output, set segment min/max,
sample rate, format, optional diarize — run, then **download the dataset as a zip**.
Show a hint that big playlists/channels are better via the CLI (mirrors the existing
playlist/feed hint). No new server dependency (same stdlib `http.server` approach;
zip via stdlib `zipfile`).

---

## 8. Output layout (what a build produces)

```text
hearsay-dataset/
  wavs/
    dQw4w9WgXcQ_0001.wav        # mono 16-bit PCM @ sample-rate
    dQw4w9WgXcQ_0002.wav
    ...
  metadata.csv                 # LJSpeech: id|text|text   (pipe, no header)
  manifest.jsonl               # NeMo: {"audio_filepath","duration","text","offset"}
  dataset_card.md              # provenance, counts, license & consent note
  dropped.jsonl                # one record per filtered-out clip, with reason
  build_report.json            # totals: kept/dropped, hours, per-source breakdown
```

With `--per-speaker`, `wavs/` and the indices are namespaced per `SPEAKER_xx`.
With `--format hf`, an additional `metadata.csv` (`file_name,transcription`) form is
emitted in an `audiofolder`-compatible layout.

---

## 9. Legal & ethical use

**hearsay is a local, user-side tool.** It processes media on your machine and
writes dataset files locally; it does **not** host, redistribute, or download
content on your behalf, and it ships no datasets. This framing runs through the
design doc, the generated `dataset_card.md`, and the README.

**Drop-in note (README + `dataset_card.md`):**

> *Legal and ethical use.* hearsay is a local, user-side tool: it processes media on
> your own machine and writes dataset files locally. It does not host, distribute,
> or download content on your behalf, and it ships no datasets. **You are solely
> responsible** for ensuring you have the rights to any audio you process and for
> complying with the source platform's terms — extracting audio from services such
> as YouTube may violate their Terms of Service. Audio and transcripts are typically
> copyrighted, and building a personal/research dataset is different from
> redistributing one (redistribution generally needs the rights holder's
> permission). If a dataset contains a real, identifiable person's voice — especially
> if you intend to clone or synthesize it — **you are responsible for obtaining that
> person's informed consent.** Voice and voiceprints can be regulated as biometric or
> personal data and by voice/likeness laws (e.g. **GDPR**, U.S. state biometric
> statutes such as **Illinois BIPA**, the **Tennessee ELVIS Act**, and the **EU AI
> Act's** synthetic-media transparency rules, which apply from **Aug 2, 2026**). This
> note is **informational, not legal advice**; consult a qualified attorney.

**`dataset_card.md` provenance fields** (auto-populated where machine-knowable;
defaults to `unknown — user must verify` rather than guessing): `source_url`,
`source_platform`, `channel_or_creator`, `title`, `source_license`, `license_link`,
`retrieval_date`, `language`, `clip_count`, `total_duration`, `sample_rate`,
`audio_format`, `speaker_count`, `consent_status` (default `unknown`), `consent_note`,
`intended_use`, `pii_note`, `generated_by: hearsay vX.Y.Z`.

Card conventions **[corrected]**: emit a HuggingFace-compatible YAML front-matter
block using **lowercase** license identifiers (`cc-by-4.0`, `cc0-1.0`, `mit`, …) and
HF's dedicated **`unknown`** identifier as the default; put hearsay's custom
provenance keys in the standard markdown sections (*Source Data*, *Personal and
Sensitive Information*, *Licensing Information*), **not** in YAML (unknown keys are
ignored). Record provenance **per source**. For a YouTube CC clip, record the
**actual** license version shown — YouTube's CC option moved to **CC BY 4.0** around
Aug 1, 2025 (older uploads remain CC BY 3.0), so we don't hard-code a version. The
mixed-license, per-sample-provenance exemplar to follow is **MLCommons People's
Speech** (Common Voice/LibriSpeech are single-license).

---

## 10. Scope boundary (explicitly OUT)

- **No model training** — hearsay produces *datasets*, not trained TTS/STT models.
- **No cloud upload / hosting / dataset redistribution** — local files only.
- **No new heavy core dependency** — torch-based features are opt-in extras.
- **No change to the markdown/JSON engine** — dataset mode is purely additive; all
  existing tests must keep passing.
- **No heavier GUI framework** — the Web-UI path reuses the existing stdlib server.

---

## 11. Module plan (additive)

A new `src/hearsay/dataset/` package; the **only** touch to existing code is adding
an opt-in `word_timestamps` capture to `transcribe.py` (default off → current
behavior, JSON schema, and markdown output all unchanged).

| Module | Phase | Responsibility |
| --- | --- | --- |
| `dataset/words.py` | D1 | engine-agnostic `Word` model + Whisper/Parakeet adapters |
| `dataset/segmentation.py` | D1 | **pure** core: words → `DatasetSegment`s (pause/sentence cuts, min/max clamp, never mid-word, repair inverted spans) |
| `dataset/audio.py` | D2 | ffmpeg slice + resample/mono + optional loudnorm; ffprobe |
| `dataset/formats.py` | D2 | LJSpeech `metadata.csv`, NeMo `manifest.jsonl`, HF `audiofolder`, `dataset_card.md` |
| `dataset/filters.py` | D3 | Tier-1/Tier-2 filters + structured drop logging |
| `dataset/diarize.py` | D5 | optional pyannote/whisperx wrapper (lazy import, clean degrade) |
| `dataset/build.py` | D2/D4 | orchestrate one source; D4 merges playlists/feeds into one dataset |
| `dataset/models.py` | D1 | `DatasetSegment`, `DatasetConfig`, `BuildReport` pydantic models |
| `transcribe.py` (edit) | D1 | add `word_timestamps: bool = False` → capture `seg.words`; `TranscriptionResult` gains optional `words` |

`transcribe.py` change is backward-compatible (new trailing optional field; the
captions/markdown paths never set it).

---

## 12. Proposed phases (gates as in the original build)

Each phase: run full suite + acceptance commands, paste real output into
`PROGRESS.md` under `### Phase D<n> Evidence`, one commit per task, then STOP for
approval. Tests stay **offline** (fixtures + cached tiny model, `local_files_only`).

- **D1 — Core segmentation engine.** Pure, heavily-tested `Word` adapter +
  `segmentation.py`: words → dataset segments cut on sentence/pause boundaries,
  clamped to `[min,max]`, never mid-word, with exact start/end + verbatim text.
  Word source = faster-whisper `word_timestamps=True` (default) / merged Parakeet
  tokens. **No audio cutting yet.** Boundary-case fixtures (long sentences, no
  punctuation, tiny gaps, inverted spans).
- **D2 — Audio export + manifest.** ffmpeg slicing (mono, sample-rate), LJSpeech
  `metadata.csv` **and** JSONL manifest **and** `dataset_card.md`. Acceptance: one
  short real video → a folder a TTS/STT pipeline can load; verify clip count,
  audio↔text alignment, durations.
- **D3 — Quality filtering.** Tier-1 filters + structured drop log + summary; tests
  on fixtures with deliberately bad segments.
- **D4 — Scale.** Playlists / channels / feeds → **one merged dataset** with a shared
  manifest, continue-past-failure, totals table (clips, hours), progress, resumable
  if feasible. Acceptance: a small real playlist → one coherent dataset.
- **D5 — Optional diarization.** `hearsay[diarize]`: per-speaker / dominant-speaker
  export; clean degrade when absent; explicit HF-token remediation.
- **D6 — Both front ends + docs.** CLI flags wired; Web-UI "Dataset mode" with zip
  download + CLI hint for big playlists; README "Build training datasets" section
  (honest accuracy/rights notes, comparison to markdown mode), comparison table +
  topics updated.
- **D7 — Launch-ready polish.** Fresh-run test following only the README; CI green;
  a tiny **license-clean example mini-dataset** committed so people see the shape; a
  short "what's new in v0.2" note; final summary.

---

## 13. Open questions / things to confirm during build

1. **Clip-level confidence aggregation** — worst-segment vs mean for the ASR-confidence
   filter (lean: worst-segment, since one bad span ruins a clip). Pin in D3.
2. **`hearsay[dataset]` necessity** — confirm whether the default path needs *any*
   extra or rides entirely on already-present ffmpeg/ffprobe/numpy/`wave`/`onnxruntime`.
3. **HF `audiofolder` `file_name`** — spot-check the exact column name against current
   HF docs before locking the `hf` format (research flagged it as the common error).
4. **Resumability (D4)** — skip already-built clips by manifest presence; decide the
   idempotency key (source_id + segment index).
5. **Sample-rate dual-target** — whether to optionally emit both 22050 and 16000 trees
   or document a one-line ffmpeg resample; default stays single-rate via `--sample-rate`.
