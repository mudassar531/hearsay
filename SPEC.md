# PASTE EVERYTHING BELOW THIS LINE INTO CLAUDE CODE (run it inside a new, empty folder)

---

You are the lead engineer building **earshot** — an open-source tool with the pitch: **"crawl4ai for video & audio."** One command turns any YouTube video, podcast episode, or local recording into clean, timestamped, chunked, LLM-ready markdown for RAG pipelines and AI agents.

Work through the phases below **in order**. Do not skip ahead. Do not expand scope. Your job is a small tool that does one thing beautifully.

## STEP ZERO (do this before anything else)

1. Save this entire prompt, verbatim, to `SPEC.md` in the repo root. It is the source of truth for all future sessions.
2. Create `PROGRESS.md` with every phase and task below as markdown checkboxes (`- [ ]`).
3. Create `DECISIONS.md` (empty, with a header). Any time reality forces a deviation from this spec, log it there with one line of reasoning. Never silently deviate.

## OPERATING RULES (non-negotiable)

- **Checkbox integrity:** Never tick a checkbox in `PROGRESS.md` without first running the verification command for that task and seeing it pass. Update `PROGRESS.md` immediately after each task — never batch updates.
- **Phase gates:** At the end of every phase: run the full test suite + that phase's acceptance commands, paste the real terminal output into `PROGRESS.md` under a `### Phase N Evidence` heading, commit everything, then **STOP**. Print: `PHASE N COMPLETE — review PROGRESS.md and reply "approved" to continue.` Do not start the next phase without approval.
- **Commits:** One commit per completed task, conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- **Blocker protocol:** If something external blocks you (network restrictions, YouTube rate limits, a model download failing) after 2 honest attempts: log it under `## Blockers` in `PROGRESS.md` with what you tried, mark the task `- [ ] ⚠ BLOCKED`, and continue with the next task. Never fake a result, never stub it silently.
- **Tests don't touch the network.** Record real API/transcript responses once into `tests/fixtures/` and test against those. CI must pass offline.
- **Quality bar:** Python type hints everywhere, docstrings on public functions, helpful error messages (every failure tells the user what to do next), handle Ctrl-C gracefully, no dead code, minimal dependencies — justify every new dep in `DECISIONS.md`.
- **Non-goals (refuse scope creep):** No GUI. No summarization features. No vector database. No storing or re-hosting content (this is a user-side ingestion tool). No video downloads when audio suffices. If a "nice idea" appears mid-build, append it to `IDEAS.md` and move on.

## TECH STACK (pinned — do not substitute without logging in DECISIONS.md)

- Python **3.11+**, managed with **uv** (env, deps, build). `src/` layout, package name `earshot`.
- CLI: **typer**. Terminal output: **rich** (progress bars, panels). Data models: **pydantic v2**.
- Captions-first path: **youtube-transcript-api**.
- Metadata/chapters/audio extraction: **yt-dlp** (use `--dump-json` for metadata without downloading; audio-only `m4a` when audio is needed). **ffmpeg** is a documented system requirement.
- Transcription fallback: **faster-whisper** (default model `small`, flag for `tiny|base|small|medium|large-v3`; must run on CPU).
- Podcast feeds: **feedparser**. Tooling: **pytest**, **ruff** (lint + format), **mypy** (lenient config). License: **MIT**.

## OUTPUT FORMAT (the product — get this exactly right)

Every ingestion produces a markdown file:

```markdown
---
title: "<video/episode title>"
source: "<original URL or file path>"
channel: "<channel/show name>"
duration: "01:02:33"
ingested: "2026-06-13T10:00:00Z"
method: "captions" | "whisper-small"
language: "en"
---

# <Title>

## <Chapter 1 title>   (only if chapters exist; otherwise time-based sections every ~5 min)

**[00:00:12]** Paragraph of transcript text, grouped into readable
paragraphs by pause length and sentence boundaries — never one line
per caption fragment, never a wall of text.

**[00:01:45]** Next paragraph...
```

Plus, when `--json` is passed, a sidecar `.json` matching a pydantic `Transcript` model: metadata + `chunks[]` each with `start_s`, `end_s`, `section`, `text`. Paragraph target: 40–120 words. This formatting quality is the entire reason the tool will get stars — treat it as the core feature, with dedicated tests.

---

## PHASE 0 — Scaffold (target: everything runs)

- [ ] Check name availability: attempt to fetch `https://pypi.org/pypi/earshot/json` (404 = free). If taken or network-blocked, log result in `DECISIONS.md`; if taken, propose 5 alternatives, pick the best, and use it consistently everywhere.
- [ ] `uv init` with `src/earshot/` layout; pin Python 3.11+.
- [ ] Typer CLI skeleton: `earshot --version`, `earshot --help` with the one-liner pitch.
- [ ] Configure ruff + mypy + pytest; add one placeholder test.
- [ ] MIT `LICENSE`, stub `README.md` (one-liner + "under construction"), `.gitignore`, `git init` + first commit.

**Acceptance:** `uv run earshot --version` prints version · `uv run pytest` passes · `uv run ruff check .` clean. **STOP.**

## PHASE 1 — The magic moment: YouTube → markdown (captions path)

- [ ] `earshot <youtube-url>`: fetch metadata via `yt-dlp --dump-json` (title, channel, duration, chapters) — no media download.
- [ ] Fetch captions via youtube-transcript-api (prefer manual subs over auto-generated; `--lang` flag, default `en` with fallback to first available).
- [ ] Implement the paragraph-grouping algorithm (pause threshold + sentence boundaries + 40–120 word targets) as a pure, heavily-tested function.
- [ ] Chapter-aware sectioning: chapters → `##` headings; no chapters → time-based sections every ~5 minutes.
- [ ] Markdown renderer producing exactly the OUTPUT FORMAT above; `-o/--output` flag, default `./<video-id>.md`.
- [ ] Friendly errors: no captions ("transcription lands in Phase 2"), private/unavailable/invalid URL.
- [ ] Fixtures: record 2 real transcript+metadata payloads (one video with chapters, one without) into `tests/fixtures/`; unit-test grouping, sectioning, rendering against them.
- [ ] Run end-to-end on 2 real videos; commit outputs to `examples/`.

**Acceptance:** `uv run earshot https://www.youtube.com/watch?v=<id>` produces a clean file in <10s · tests pass offline · `examples/` contains 2 real outputs. **STOP.**

## PHASE 2 — Ears: whisper fallback + local files

- [ ] `earshot <file.mp3|.mp4|.wav|.m4a>`: extract audio via ffmpeg if needed → faster-whisper → same markdown pipeline (whisper segments → same grouping function).
- [ ] `--transcribe` flag to force whisper on YouTube URLs (yt-dlp audio-only download → transcribe); auto-fallback to this path when no captions exist, with a clear "transcribing locally, this takes a few minutes" notice.
- [ ] `--model` flag (default `small`); rich progress bar during transcription; clean temp-file handling (always delete downloaded audio).
- [ ] Add a short (<30s) public-domain audio clip to `tests/fixtures/`; integration test transcribes it with `tiny`.
- [ ] Document ffmpeg as a requirement with install commands per OS in README.

**Acceptance:** `uv run earshot tests/fixtures/sample.wav` produces correct markdown · captionless YouTube video works end-to-end via fallback. **STOP.**

## PHASE 3 — Scale: podcasts, playlists, JSON

- [ ] Podcast RSS: `earshot <feed-url>` lists episodes (rich table); `--latest`, `--episode N`, `--all --limit N`.
- [ ] YouTube playlists: same flags; batch mode writes to `--output-dir` (default `./earshot-out/`), continues past per-item failures, prints a summary table.
- [ ] `--json` sidecar implementing the `Transcript` pydantic model; export the JSON schema to `docs/schema.json`.
- [ ] Tests: feed parsing fixture, batch-failure handling, JSON schema validation.

**Acceptance:** one command ingests 3 podcast episodes into a folder with markdown+json · tests pass. **STOP.**

## PHASE 4 — Distribution hack: MCP server mode

- [ ] Add the official `mcp` Python SDK; `earshot mcp` starts a stdio MCP server exposing two tools: `ingest_url(url, transcribe?, lang?)` and `ingest_file(path)` — both return the markdown string.
- [ ] Keep MCP deps in an optional extra (`pip install earshot[mcp]`) if they're heavy.
- [ ] Round-trip test: spawn the server, call `ingest_file` on the fixture clip, assert markdown comes back.
- [ ] README section: "Give your agent ears" — exact JSON config snippets for Claude Code and Claude Desktop.

**Acceptance:** MCP round-trip test passes · config snippet verified syntactically. **STOP.**

## PHASE 5 — Launch kit

- [ ] Real README: hero one-liner, install (`uv tool install` / `pipx`), 30-second quickstart, output example, comparison table (earshot vs DIY yt-dlp+whisper plumbing vs document tools like markitdown/docling — we do *media*), badges.
- [ ] `demo/record.sh` using **vhs** (or asciinema fallback) that records the money shot: paste URL → beautiful markdown appears. Output `demo.gif` referenced at the top of README.
- [ ] GitHub Actions CI: ruff + mypy + pytest on 3.11/3.12.
- [ ] `CONTRIBUTING.md` + 3 written-up "good first issue" drafts in `docs/good-first-issues.md`.
- [ ] `uv build` succeeds; dry-run publish checklist in `docs/release.md` (do **not** actually publish).
- [ ] `launch/show-hn.md`: a draft Show HN title + 150-word post, and a 4-tweet thread draft. Plain, honest, demo-first tone — no hype words.

**Acceptance:** fresh-clone test — `git clone` → install → ingest a URL works following only the README · CI green · demo gif renders. **STOP — project complete. Print a final summary of what shipped, what's blocked, and the top 3 next ideas from IDEAS.md.**

## PHASE 6 — Stretch (DO NOT START without explicit approval)

Speaker diarization via whisperX · `--frames` keyframe extraction · vector-store export helpers · web URL article fallback. Park all of these in `IDEAS.md`.

---

# v0.2 — Dataset Export Mode (TTS/STT)

*Appended 2026-06-14. The PHASE 0–6 spec above is the v0.1 source of truth and remains unchanged. This section adds a second, additive output mode. Full design + citations: `docs/dataset-mode-design.md`. Operating rules below mirror the original; deviations still go to `DECISIONS.md`.*

## THE GOAL

Today hearsay turns media into **readable** markdown (40–120 word paragraphs) for RAG and humans. v0.2 adds a separate output mode that turns media into **machine-learning training datasets** for **TTS and STT**: a user points hearsay at a YouTube video, a whole playlist/channel, a podcast feed, or local files and gets a clean, standard, ready-to-train dataset —

- the **audio sliced into short segments** (~1–15 s, cut on sentence/pause boundaries, **never mid-word**),
- each segment paired with its **exact verbatim transcript** and **precise timestamps**,
- saved in a **standard layout** (audio-clip folder + manifest files) that real TTS/STT training pipelines read directly,
- with **quality filtering** that drops junk (music, silence, overlapping speech, wrong language, too short/long).

This does **not** replace or break the markdown/JSON output — it is a new mode, surfaced in **both** the CLI and the web UI.

## CHOSEN FORMATS (the two defaults, over one shared `wavs/` tree)

- **TTS — LJSpeech-style** `metadata.csv` (pipe-delimited, no header) + `wavs/` of mono 16-bit PCM WAV @ 22050 Hz. Emit **`id|text|text`** (verbatim transcript duplicated) so it loads in **both** Coqui (reads col 3) and Piper (reads last col) without a multi-speaker misparse.
- **STT — NeMo-style** `manifest.jsonl`: one object per line `{"audio_filepath","duration"(sec float),"text","offset":0.0}`, `audio_filepath` relative to the manifest.
- Optional third target `--format hf`: HuggingFace `audiofolder` `metadata.csv` (`file_name,transcription`).
- `--sample-rate` default 22050 (TTS-canonical); 16000 mono documented for ASR.

## ALIGNMENT APPROACH

Caption timestamps are cue-level and loose (no word timing) — **not** usable for clip-accurate slicing. Default word source: **faster-whisper `word_timestamps=True`** (already core, MIT, no new deps); on Apple Silicon, merge Parakeet token timings (split on the literal leading-space marker). One engine-agnostic `Word` adapter normalizes field-name differences. Boundaries are coarse (~100–400 ms jitter, occasional inverted spans), so the slicer repairs inverted spans, snaps to inter-word silence, and pads edges (~100 ms). Better-but-optional: forced alignment via `hearsay[align]` (WhisperX/torchaudio MMS_FA) — heavy (torch), and with a model-license guardrail (many default alignment models are CC-BY-NC).

## NEW DEPENDENCIES (keep core light)

Core gains **nothing** — numpy, PyAV (`av`), and `onnxruntime` are already transitive via faster-whisper; ffmpeg/ffprobe are an existing system requirement. New optional extras only: `hearsay[dataset]` (light: `soundfile`, `webrtcvad-wheels`; may prove unnecessary), `hearsay[align]` (torch), `hearsay[diarize]` (`pyannote.audio`, torch, HF-gated models). Each justified in `DECISIONS.md`.

## DEFAULT BEHAVIOR ON THE SPEAKER PROBLEM

Without diarization (default): emit a **mixed-speaker** dataset (fine for STT) and **warn** it is unsuitable for single-voice TTS. With `hearsay[diarize]` + an HF token: `--per-speaker` or `--dominant-speaker` export. pyannote models are gated (accept conditions on both `segmentation-3.0` and the diarization pipeline + read token); hearsay reads `HF_TOKEN`/`--hf-token` and prints exact remediation on auth failure.

## SCOPE BOUNDARY (explicitly OUT)

No model **training** (we produce datasets, not models) · no cloud upload / hosting / dataset redistribution (local files only) · no new heavy **core** dependency · **no change to the markdown/JSON engine** · no heavier GUI framework (web UI reuses the stdlib server).

## LEGAL / ETHICS

hearsay is a local, user-side tool: it does not host, redistribute, or download on the user's behalf, and ships no datasets. The **user is responsible** for rights to the media and for voice consent (YouTube ToS, copyright, GDPR/biometric law, Illinois BIPA, Tennessee ELVIS Act, EU AI Act Art. 50 synthetic-media transparency from Aug 2 2026). An honest note goes in the README and every generated `dataset_card.md`. Informational, not legal advice.

## DELIVERY (a new `hearsay dataset <SOURCE>` subcommand)

A subcommand, not a `--dataset` flag on `ingest` (distinct large option surface + folder-tree output; keeps `ingest` pristine; reuses shared plumbing). Web UI gains a "Dataset mode" path with a zip download and a CLI hint for big playlists.

## PHASES (gates identical to v0.1: full suite + acceptance → paste evidence → commit → STOP)

- **Phase D1 — Core segmentation engine.** Pure, tested `Word` adapter + segmentation: words → dataset segments (sentence/pause cuts, min/max clamp, never mid-word, repair inverted spans, verbatim text + exact times). No audio cutting yet. Heavy boundary-case unit tests.
- **Phase D2 — Audio export + manifest.** ffmpeg slicing (mono, sample rate), LJSpeech `metadata.csv` **and** JSONL manifest **and** `dataset_card.md`. Acceptance: one short real video → a loadable dataset; verify clip count, audio↔text alignment, durations.
- **Phase D3 — Quality filtering.** Tier-1 filters (duration, silence, ASR confidence, language match, text sanity) + structured per-clip drop log + kept/dropped summary. Tests on deliberately bad fixtures.
- **Phase D4 — Scale.** Playlists / channels / feeds → one merged dataset with a shared manifest, continue-past-failure, totals (clips, hours), progress, resumable if feasible. Acceptance: a small real playlist → one coherent dataset.
- **Phase D5 — Optional diarization.** `hearsay[diarize]`: per-speaker / dominant-speaker export; clean degrade when absent; explicit HF-token remediation.
- **Phase D6 — Both front ends + docs.** CLI flags wired; web-UI "Dataset mode" with zip download + CLI hint; README "Build training datasets" section + comparison table/topics updated.
- **Phase D7 — Launch-ready polish.** Fresh README-only run; CI green; tiny license-clean example mini-dataset committed; "what's new in v0.2" note; final summary.
