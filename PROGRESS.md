# PROGRESS

Project: **hearsay** (see DECISIONS.md — `earshot` was taken on PyPI).
Rule: a box is ticked only after its verification command ran and passed. One commit per task.

## STEP ZERO

- [x] Save the prompt verbatim to `SPEC.md` — verified: file present, committed in 58127c1
- [x] Create `PROGRESS.md` with every phase and task as checkboxes — verified: this file, committed in 58127c1
- [x] Create `DECISIONS.md` with a header — verified: file present, committed in 58127c1

## PHASE 0 — Scaffold (target: everything runs)

- [x] Check name availability on PyPI; if taken, propose 5 alternatives, pick the best, use it everywhere — `earshot` taken (HTTP 200); picked `hearsay` (HTTP 404 = free); see DECISIONS.md
- [x] `uv init` with `src/hearsay/` layout; pin Python 3.11+ — verified: `uv run python -c "import hearsay"` OK on Python 3.11.15
- [x] Typer CLI skeleton: `hearsay --version`, `hearsay --help` with the one-liner pitch — verified: `uv run hearsay --version` → `hearsay 0.1.0`; `--help` shows pitch
- [x] Configure ruff + mypy + pytest; add one placeholder test — verified: `ruff check` clean, `mypy` clean, `pytest` 2 passed
- [x] MIT `LICENSE`, stub `README.md` (one-liner + "under construction"), `.gitignore`, `git init` + first commit — verified: all files present; .gitignore + git init were done in STEP ZERO (see DECISIONS.md)

**Acceptance:** `uv run hearsay --version` prints version · `uv run pytest` passes · `uv run ruff check .` clean.

### Phase 0 Evidence

Run on 2026-06-13 (macOS, uv 0.11.17, Python 3.11.15). An independent 3-agent gate
review re-ran acceptance, audited spec compliance, and code-reviewed the scaffold;
its accepted findings (scoped mypy leniency, hardened smoke tests) were fixed before
this evidence run.

```text
$ uv run hearsay --version
hearsay 0.1.0

$ uv run pytest
....                                                                     [100%]
4 passed in 0.05s

$ uv run ruff check .
All checks passed!

$ uv run mypy
Success: no issues found in 3 source files
```

## PHASE 1 — The magic moment: YouTube → markdown (captions path)

- [x] `hearsay <youtube-url>`: fetch metadata via `yt-dlp --dump-json` (title, channel, duration, chapters) — no media download — verified: 22 tests pass offline; live fetch returns correct title/channel/duration/chapters
- [x] Fetch captions via youtube-transcript-api (prefer manual subs over auto-generated; `--lang` flag wiring lands with the CLI task; default `en` with fallback to first available) — verified: 31 tests pass offline; live fetch selects manual `en` for rStL7niR7gs
- [x] Implement the paragraph-grouping algorithm (pause threshold + sentence boundaries + 40–120 word targets) as a pure, heavily-tested function — verified: 67 tests pass; lossless + 100% word-budget on both real fixtures; chosen via judge panel (see DECISIONS.md)
- [x] Chapter-aware sectioning: chapters → `##` headings; no chapters → time-based sections every ~5 minutes — verified: 8 sectioning tests pass against the real chapter fixture and synthetic time-window cases
- [x] Markdown renderer producing exactly the OUTPUT FORMAT in SPEC.md; `-o/--output` flag, default `./<video-id>.md` — verified: render tests pass; pipeline+CLI wired; `hearsay <url>` and `-o` produce correct files (see examples/)
- [x] Friendly errors: no captions ("transcription lands in Phase 2"), private/unavailable/invalid URL — verified: 12 error tests pass; live invalid URL & nonexistent video render a hint + exit 1 with no traceback
- [x] Fixtures: record 2 real transcript+metadata payloads (one video with chapters, one without) into `tests/fixtures/`; unit-test grouping, sectioning, rendering against them — verified: fixtures committed; grouping/sectioning tests run on them; full-chain fixture→render test added
- [x] Run end-to-end on 2 real videos; commit outputs to `examples/` — verified: both ran in <10s (5.0s / 3.6s); outputs in examples/ (293 + 1121 lines)

**Acceptance:** `uv run hearsay https://www.youtube.com/watch?v=<id>` produces a clean file in <10s · tests pass offline · `examples/` contains 2 real outputs.

### Phase 1 Evidence

Run on 2026-06-13 (macOS, uv 0.11.17, Python 3.11.15). An independent 5-lens gate
review (acceptance, spec, core code, I/O code, offline guarantee) verified the
phase; its one confirmed blocker (file-write traceback) and several minors were
fixed before this evidence run — see DECISIONS.md. Tests run offline by
construction (fixtures + injected fetchers) and a `tests/conftest.py` now blocks
non-loopback network to enforce it.

```text
$ uv run pytest
........................................................................ [ 77%]
.....................                                                    [100%]
93 passed in 0.39s

$ uv run ruff check .
All checks passed!

$ uv run mypy
Success: no issues found in 20 source files

$ time uv run hearsay https://www.youtube.com/watch?v=rStL7niR7gs -o /tmp/p1.md
╭─────────────────── hearsay ───────────────────╮
│ ✓ You Would Be a Terrible Leader              │
│ 4 sections · 37 paragraphs · method: captions │
│ → /tmp/p1.md                                  │
╰───────────────────────────────────────────────╯
real 3.29   (well under the 10s budget)

$ head -8 /tmp/p1.md
---
title: "You Would Be a Terrible Leader"
source: "https://www.youtube.com/watch?v=rStL7niR7gs"
channel: "CGP Grey"
duration: "00:18:13"
ingested: "2026-06-13T02:43:06Z"
method: "captions"
language: "en"
---

$ ls examples/
README.md   rStL7niR7gs.md   zjkBMFhNj_g.md
```

## PHASE 2 — Ears: whisper fallback + local files

- [x] `hearsay <file.mp3|.mp4|.wav|.m4a>`: extract audio (faster-whisper's PyAV decodes directly; no ffmpeg step needed) → faster-whisper → same markdown pipeline — verified: `hearsay tests/fixtures/sample.wav` produces correct markdown (method whisper-small)
- [x] `--transcribe` flag to force whisper on YouTube URLs (yt-dlp audio-only download → transcribe); auto-fallback when no captions exist, with a "transcribing locally, this takes a few minutes" notice — verified: live `--transcribe --model tiny` on a 19s video; fallback covered by tests
- [x] `--model` flag (default `small`); rich progress bar during transcription; clean temp-file handling (always delete downloaded audio) — verified: `--model` enum validated by Typer; temp dir auto-deleted (test + live check); progress bar driven by whisper callback
- [x] Add a short (<30s) public-domain audio clip to `tests/fixtures/`; integration test transcribes it with `tiny` — verified: 4.7s OS-TTS WAV; `test_transcribe_sample_with_tiny` passes (offline via cached model + local_files_only)
- [x] Document ffmpeg as a requirement with install commands per OS in README — verified: README Requirements section lists per-OS install commands (brew/apt/dnf/pacman/winget/choco)

**Acceptance:** `uv run hearsay tests/fixtures/sample.wav` produces correct markdown · captionless YouTube video works end-to-end via fallback.

### Phase 2 Evidence

Run on 2026-06-13 (macOS, uv 0.11.17, Python 3.11.15, ffmpeg 8.1). An independent
5-lens gate review (acceptance, spec, transcription code, CLI code, offline
guarantee) verified the phase; acceptance/spec/offline passed with no confirmed
blockers, and the surfaced minors were fixed — see DECISIONS.md.

```text
$ uv run pytest
........................................................................ [ 67%]
...................................                                      [100%]
107 passed in 1.59s

$ uv run ruff check .
All checks passed!

$ uv run mypy
Success: no issues found in 22 source files

$ uv run hearsay tests/fixtures/sample.wav -o /tmp/p2.md
╭───────────────────── hearsay ─────────────────────╮
│ ✓ sample                                          │
│ 1 sections · 1 paragraphs · method: whisper-small │
│ → /tmp/p2.md                                      │
╰───────────────────────────────────────────────────╯

$ head -16 /tmp/p2.md
---
title: "sample"
source: "tests/fixtures/sample.wav"
channel: "Local file"
duration: "00:00:04"
ingested: "..."
method: "whisper-small"
language: "en"
---

# sample

## [00:00:00 – 00:00:04]

**[00:00:00]** The quick brown fox jumps over the lazy dog, hearsay turns audio
into clean markdown.
```

Captionless-fallback acceptance, verified live on a genuinely captionless video
(`aqz-KE-bpKQ`, Big Buck Bunny, captions disabled):

```text
$ uv run hearsay https://www.youtube.com/watch?v=aqz-KE-bpKQ --model tiny -o /tmp/fallback.md
No captions found. Falling back to local transcription.
Downloading audio, then transcribing locally with whisper 'tiny'. This can take a few minutes.
╭───────────────────────────── hearsay ──────────────────────────────╮
│ ✓ Big Buck Bunny 60fps 4K - Official Blender Foundation Short Film │
│ 0 sections · 0 paragraphs · method: whisper-tiny                   │
╰────────────────────────────────────────────────────────────────────╯
```

(Big Buck Bunny is a dialogue-free animation, so the transcript is empty —
correct behaviour. A speech video transcribes with real content: `--transcribe`
on the 19s "Me at the zoo" yields a clean paragraph, method `whisper-tiny`.)

## PHASE 3 — Scale: podcasts, playlists, JSON

- [x] Podcast RSS: `hearsay <feed-url>` lists episodes (rich table); `--latest`, `--episode N`, `--all --limit N` — verified: live listing table on Merriam-Webster WOTD; live `--all --limit 3` ingested 3 episodes
- [x] YouTube playlists: same flags; batch mode writes to `--output-dir` (default `./hearsay-out/`), continues past per-item failures, prints a summary table — verified: playlist fixture parsed; batch-failure test confirms one failure doesn't abort; summary table prints
- [x] `--json` sidecar implementing the `Transcript` pydantic model; export the JSON schema to `docs/schema.json` — verified: live `--json` writes a valid sidecar (37 chunks); `docs/schema.json` committed and a test asserts it stays in sync
- [x] Tests: feed parsing fixture, batch-failure handling, JSON schema validation — verified: 141 tests pass offline (test_feeds, test_batch, test_transcript, playlist parsing)

**Acceptance:** one command ingests 3 podcast episodes into a folder with markdown+json · tests pass.

### Phase 3 Evidence

Run on 2026-06-13 (macOS, uv 0.11.17, Python 3.11.15). An independent 6-lens gate
review (acceptance, spec, feeds/playlist code, batch/CLI code, JSON/schema,
offline) verified the phase; acceptance/spec/JSON/offline passed, and its one
confirmed blocker (feed slug collisions = silent overwrite) plus several minors
were fixed — see DECISIONS.md.

```text
$ uv run pytest
........................................................................ [ 96%]
.....                                                                    [100%]
149 passed in 1.66s

$ uv run ruff check .
All checks passed!

$ uv run mypy
Success: no issues found in 27 source files
```

Acceptance — one command ingests 3 podcast episodes to a folder with markdown+json
(Merriam-Webster Word of the Day, ~2-min episodes, whisper-tiny):

```text
$ uv run hearsay "https://www.merriam-webster.com/wotd/feed/rss2" --all --limit 3 --json --model tiny --output-dir /tmp/pod-out
[1/3] blandishment
[2/3] saturnine
[3/3] foible
                     hearsay batch
┏━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Status ┃ Item         ┃ Output / error               ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ✓      │ blandishment │ /tmp/pod-out/blandishment.md │
│ ✓      │ saturnine    │ /tmp/pod-out/saturnine.md    │
│ ✓      │ foible       │ /tmp/pod-out/foible.md       │
└────────┴──────────────┴──────────────────────────────┘
3 succeeded · → /tmp/pod-out/
$ ls /tmp/pod-out/
blandishment.json  blandishment.md  foible.json  foible.md  saturnine.json  saturnine.md
```

Continue-past-failure also verified live: a feed whose newest episode's media
returned HTTP 404 produced a red ✗ row with the reason, exit 0, no traceback —
the batch did not abort.

## PHASE 4 — Distribution hack: MCP server mode

- [x] Add the official `mcp` Python SDK; `hearsay mcp` starts a stdio MCP server exposing two tools: `ingest_url(url, transcribe?, lang?)` and `ingest_file(path)` — both return the markdown string — verified: `hearsay --help` lists `mcp`; round-trip test lists both tools and gets markdown back
- [x] Keep MCP deps in an optional extra (`pip install hearsay[mcp]`); `hearsay mcp` prints an install hint if absent — verified: `[project.optional-dependencies].mcp`; lazy import in mcp_server with friendly error
- [x] Round-trip test: spawn the server, call `ingest_file` on the fixture clip, assert markdown comes back — verified: `test_mcp_stdio_roundtrip` spawns `python -m hearsay mcp` and asserts markdown (offline via HF_HUB_OFFLINE, skips if model uncached)
- [x] README section: "Give your agent ears" — exact JSON config snippets for Claude Code and Claude Desktop — verified: both snippets parse as JSON (`type: stdio`); `claude mcp add` syntax + filenames confirmed via claude-code-guide

**Acceptance:** MCP round-trip test passes · config snippet verified syntactically.

### Phase 4 Evidence

Run on 2026-06-13 (macOS, uv 0.11.17, Python 3.11.15). An independent 5-lens gate
review (acceptance, spec, MCP protocol, CLI-group regression, offline) passed all
five lenses with zero blockers; the one surfaced minor (options-before-source
regression) was fixed — see DECISIONS.md.

```text
$ uv run pytest
158 passed in 3.28s

$ uv run pytest tests/test_mcp.py -v
tests/test_mcp.py ......                                                 [100%]
6 passed in 1.78s
  (incl. test_mcp_stdio_roundtrip — spawns `python -m hearsay mcp`, calls
   ingest_file on the fixture over stdio, asserts markdown comes back)

$ uv run ruff check . && uv run mypy
All checks passed!
Success: no issues found in 30 source files

$ uv run hearsay --help        # lists both subcommands
Commands:
  mcp     Run the MCP stdio server (give your AI agent ears).
  ingest  crawl4ai for video & audio — turn any YouTube video, podcast episode...

$ uv run hearsay --version
hearsay 0.1.0
```

Config snippets: both README `json` blocks parse via `json.loads` (Claude Code
`.mcp.json` and Claude Desktop `claude_desktop_config.json`, each `type: stdio`);
the `claude mcp add` syntax and config filenames were confirmed against current
Claude Code / Claude Desktop docs.

## PHASE 5 — Launch kit

- [x] Real README: hero one-liner, install (`uv tool install` / `pipx` + from-source), 30-second quickstart, output example, comparison table, badges — verified: renders; both JSON snippets parse; demo gif at top
- [x] `demo/record.sh` using vhs; output `demo.gif` referenced at the top of README — verified: vhs installed; gif generated (208KB); last-frame inspection shows the command + success panel + markdown output
- [x] GitHub Actions CI: ruff + mypy + pytest on 3.11/3.12 — verified: pushed; `gh run` conclusion=success on both 3.11 and 3.12
- [x] `CONTRIBUTING.md` + 3 written-up "good first issue" drafts in `docs/good-first-issues.md` — verified: both files present; 3 scoped issues with files-to-touch + verification
- [x] `uv build` succeeds; dry-run publish checklist in `docs/release.md` (does not publish) — verified: `uv build` produced wheel + sdist; release.md is a dry-run checklist, no auto-publish
- [x] `launch/show-hn.md`: Show HN title + 150-word post + 4-tweet thread — verified: present; plain, demo-first tone

**Acceptance:** fresh-clone test — `git clone` → install → ingest a URL works following only the README · CI green · demo gif renders.

### Phase 5 Evidence

Run on 2026-06-13 (macOS, uv 0.11.17, Python 3.11.15). An independent 5-lens gate
review (fresh-clone, README/docs, CI/build, launch copy, full-project
completeness) passed fresh-clone/README/CI-build/launch; the one flagged item
(this missing evidence block) was the phase-gate step itself and was downgraded
to minor — now added. Its surfaced nits (stale cli docstring, hype words quoted
in the show-hn editorial note, implicit CI ffmpeg) were fixed.

```text
$ uv run pytest
158 passed in 3.43s

$ uv run ruff check . && uv run ruff format --check . && uv run mypy
All checks passed!
32 files already formatted
Success: no issues found in 30 source files

$ uv build
Successfully built dist/hearsay-0.1.0.tar.gz
Successfully built dist/hearsay-0.1.0-py3-none-any.whl

$ file demo/demo.gif
demo/demo.gif: GIF image data, version 89a, 1100 x 720
```

**CI:** GitHub Actions `CI` workflow conclusion `success` on Python 3.11 and 3.12
(ruff + ruff format + mypy + pytest) — re-run on the final commit.

**Fresh-clone acceptance** (verified by the gate review from a clean checkout,
following only the README's from-source block):

```text
$ git clone <repo> /tmp/fc && cd /tmp/fc      # clean: no .venv/dist
$ uv tool install .                            # README path A → installs `hearsay`
$ hearsay --version                            # hearsay 0.1.0
$ uv sync && uv run hearsay --version          # README path B (dev)  → hearsay 0.1.0
$ uv run hearsay "https://www.youtube.com/watch?v=rStL7niR7gs" -o out.md
  ✓ You Would Be a Terrible Leader · 4 sections · 37 paragraphs · method: captions
# out.md: valid YAML frontmatter + "# " heading + "**[00:00:00]**" markers
```

## PHASE 6 — Stretch (DO NOT START without explicit approval)

Parked in `IDEAS.md`: speaker diarization via whisperX · `--frames` keyframe extraction · vector-store export helpers · web URL article fallback.

---

# v0.2 — Dataset Export Mode (TTS/STT)

New, additive mode (`hearsay dataset <SOURCE>`). Design + citations: `docs/dataset-mode-design.md`. Same rules: a box is ticked only after its verification command ran and passed; one commit per task; phase gates paste real evidence then STOP.

## STEP ZERO (Dataset mode)

- [x] Research target formats, alignment, audio cutting, quality filters, diarization, deps/licenses, legality (web-grounded + adversarially verified) — verified: 7-dimension research workflow completed, findings folded into the design memo
- [x] Write `docs/dataset-mode-design.md` (feasibility & design memo) — verified: file present
- [x] Append "v0.2 — Dataset Export Mode (TTS/STT)" to `SPEC.md`, keeping all v0.1 content intact — verified: section appended below PHASE 6
- [x] Add v0.2 phases to `PROGRESS.md` and log key decisions in `DECISIONS.md`; move diarization + vector-store/export helpers out of `IDEAS.md` into this plan — verified: this section; DECISIONS entry added; IDEAS.md updated
- [x] Design approved by the user (2026-06-14, "approved ultracode") — Phase D1 started

## PHASE D1 — Core segmentation engine (the heart)

- [x] Engine-agnostic `Word` model + adapters: faster-whisper `Word(word,start,end,probability)` and merged Parakeet tokens (split on literal leading space) — verified: `Word` in `models.py` (frozen), adapters in `dataset/words.py`; 13 adapter tests pass (merge, blank/None drop, clamp, geometric-mean confidence, defensive timings)
- [x] Add opt-in `word_timestamps` capture to `transcribe.py` (default off → existing behavior, JSON schema, and markdown output all unchanged) — verified: `TranscriptionResult.words` (default None); whisper requests word timings + parakeet merges tokens only when asked; tiny-model test captures fox/dog words; schema-sync test still passes
- [x] Pure `dataset/segmentation.py`: words → `DatasetSegment`s — cut on sentence/pause boundaries, clamp to `[min_s, max_s]`, never split mid-word, repair inverted spans, attach exact start/end + verbatim text — verified: greedy pause-primary (chosen via a 3-design judge panel; see DECISIONS); spans use range *extent* so a glitched interior word can't hide
- [x] Heavy unit tests: long sentences, no punctuation, tiny gaps, inverted/overlapping/out-of-order/NaN spans, one-huge-word, empty input, tiny-tail merge/isolate/rebalance, determinism, plus a 400-case property/fuzz pass over noisy inputs — verified: 34 segmentation tests pass offline

**Acceptance:** segmentation tests pass offline; segments are lossless w.r.t. input words, never cut mid-word, all within `[min,max]` (or flagged when a single word exceeds max).

### Phase D1 Evidence

Run on 2026-06-14 (macOS, uv 0.11.17, Python 3.11.15). The segmentation algorithm
was chosen via a 3-candidate judge panel (pause-primary / sentence-first / DP) whose
adversarial critic caught a real design bug — an endpoint-only duration let a glitched
interior word hide and "launder" the `oversized` flag — fixed by computing every clip's
span from its **range extent**. An independent 4-lens implementation review then
adversarially verified the code: **zero real bugs** (the one substantive segmentation
finding was confirmed *not-a-bug* — tiny-tail merge across a pause is the intended D1
contract; internal-silence drop is a D3 filter), and 7 quality improvements were applied
(dead-constant removal, `target_s` NaN guard, geometric-mean Parakeet confidence,
defensive timing coercion, doc wording).

```text
$ uv run pytest tests/test_dataset_segmentation.py tests/test_dataset_words.py tests/test_transcribe.py
54 passed in 3.90s

$ uv run pytest            # full suite — existing behaviour unchanged
218 passed in 8.10s

$ uv run ruff check . && uv run mypy
All checks passed!
Success: no issues found in 38 source files
```

Invariants asserted by the property/fuzz test (400 noisy random inputs incl.
NaN/inf/inverted/overlapping/out-of-order spans, blanks, degenerate params): lossless
partition by identity & order, never mid-word, non-negative monotone spans, span =
true extent of its words, `oversized` iff duration > max_s, no empty segments, and
determinism. The markdown/JSON path is untouched (`word_timestamps` defaults False;
`docs/schema.json` unchanged).

## PHASE D2 — Audio export + manifest

- [x] `dataset/audio.py`: ffmpeg slice (input-seek `-ss START -i -t DUR`, re-encode, `-ac 1`, `-ar SR`, `pcm_s16le`); ffprobe duration; `ensure_tools()` verifies ffmpeg+ffprobe on PATH (filter-availability checks land with `--normalize` in D6) — verified: real slicing of the fixture; ffprobe confirms mono @ target SR
- [x] `dataset/formats.py`: LJSpeech `metadata.csv` (`id|text|text`) **and** NeMo `manifest.jsonl` **and** `dataset_card.md` (provenance, counts, total duration, language, rights/consent note); optional HF `audiofolder` (`metadata.jsonl`, no collision) — verified: 8 format tests incl. non-ASCII title + pipe-safety
- [x] `dataset/build.py`: orchestrate one source → dataset folder; clip naming `<source_id>_<index4>.wav`; clamp ends to media length, drop past-EOF ghost clips — verified: 8 build tests + live YouTube run
- [x] Tests: manifest/csv shape; audio↔text↔duration alignment on the fixture clip; oversized + past-EOF cases — verified: 16 D2 tests pass offline (ffmpeg on local files)

**Acceptance:** run on one short real video → a folder a TTS/STT pipeline could load; clip count correct, audio+text line up, durations right.

### Phase D2 Evidence

Run on 2026-06-14 (macOS, uv 0.11.17, Python 3.11.15, ffmpeg/ffprobe 8.1). An
independent 4-lens review (ffmpeg correctness / format conformance / build
orchestration / offline+spec) adversarially verified the code and caught **2 real
bugs** — both fixed: (1) `dataset_card.md` emitted `pretty_name`/`language` with
`ensure_ascii=True`, producing lone surrogates in YAML for emoji titles; (2) a clip
whose end ran past the source produced a 0-second "ghost" clip with a `duration: 0.0`
manifest row. Build now clamps clip ends to the probed media length and drops
fully-past-EOF clips. 6 improvements applied (start-clamp consistency, doc wording,
extra regression tests).

```text
$ uv run pytest tests/test_dataset_formats.py tests/test_dataset_build.py
16 passed in 1.69s

$ uv run pytest            # full suite — markdown/JSON path unchanged
234 passed in 11.21s

$ uv run ruff check . && uv run mypy
All checks passed!
Success: no issues found in 43 source files
```

**Live acceptance** — one short real video → a loadable dataset (tiny model):

```text
$ build_dataset_from_youtube("https://www.youtube.com/watch?v=jNQXAC9IVRw", sr=22050, max=10s)
clips: 3 · total: 14.42s · oversized: 0 · lang: en · files: metadata.csv, manifest.jsonl, dataset_card.md

/tmp/hearsay-ds-acc/
  wavs/{jNQXAC9IVRw_0001,_0002,_0003}.wav   # ffprobe: 22050 Hz, mono; 3.34 / 9.22 / 1.86 s
  metadata.csv     # jNQXAC9IVRw_0001|All right, so here we are...|All right, so here we are...
  manifest.jsonl   # {"audio_filepath":"wavs/jNQXAC9IVRw_0001.wav","duration":3.34,"text":"...","offset":0.0}
  dataset_card.md  # YAML (license: unknown, "en", pretty_name "Me at the zoo") + provenance + consent note
```

Manifest durations exactly match the probed WAV durations (audio↔text↔duration aligned).

**Tracked for later phases (from the review, intentionally deferred):**
- Source-sample-rate-below-target **upsampling warning** (design memo §3) → D6 (user-facing channel).
- ffmpeg **filter-availability** check (loudnorm/silenceremove) → D6, with `--normalize`.
- `_safe_id` **collision** across same-titled sources is harmless for single-source D2; D4 batch must dedupe (like `batch.ensure_unique_slugs`).

## PHASE D3 — Quality filtering

- [x] Tier-1 filters (default on, no heavy deps, engine-agnostic): duration `[1,15]s`, internal-silence gap (>2s, glitch-resistant), **confidence** (mean per-word, an engine-agnostic proxy for `avg_logprob`/`no_speech_prob` — clips are word-regrouped & Parakeet has no segment logprobs; see DECISIONS), **compression_ratio** (Whisper's gzip formula, on text), **non_target_script** + chars-per-second (dep-free language proxy; per-language, skipped for CJK/unmapped) — verified: 20 filter tests
- [x] Structured per-clip drop log (`dropped.jsonl`: clip, filter, value, threshold, text) + `BuildReport.dropped_count`/`drops_by_reason`; thresholds config-overridable via `FilterConfig`; `--no-filter` = `FilterConfig(enabled=False)` — verified: build-integration test + live demo
- [x] Optional Tier-2 (off by default): clipping detection (dep-free, stdlib `wave` + numpy; both rails). Silero VAD / SNR deferred to an optional `hearsay[dataset]` extra (VAD needs a bundled model; kept core dep-free) — logged in DECISIONS — verified: clipping tests (incl. negative rail)
- [x] Tests on deliberately bad segments (too short/long, silent join, wrong-script, garbage/repetitive, low-confidence) — verified: 20 filter unit tests + 2 build-integration tests

**Acceptance:** filters drop the bad fixtures with correct logged reasons; summary reports kept/dropped counts.

### Phase D3 Evidence

Run on 2026-06-14 (macOS, uv 0.11.17, Python 3.11.15). The filters were
implemented from design memo §4 (with the Step-Zero corrections) then verified by
an independent 4-lens adversarial review (the false-positive lens hit a transient
content-filter API error and was re-checked by hand — see below). It caught **2 real
bugs**, both fixed: (1) the clipping detector missed the **negative rail**
(`np.abs(int16 -32768)` overflows) — now widened to int32; (2) a glitched far-future
word end **masked a real internal silence** — the word end is now capped at a
plausible max so the join is still caught. 6 improvements applied (deterministic
strict-majority script test, char-rate skipped for unmapped scripts, doc/keying
accuracy, extra tests).

```text
$ uv run pytest tests/test_dataset_filters.py
20 passed in 0.10s

$ uv run pytest            # full suite — markdown/JSON path unchanged
256 passed in 9.90s

$ uv run ruff check . && uv run ruff format --check . && uv run mypy
All checks passed!   45 files already formatted   Success: no issues found in 45 source files
```

**Acceptance — filters drop bad clips with logged reasons + summary** (within the fixture):

```text
SUMMARY: kept 1 · dropped 1 · by reason {'non_target_script': 1}
dropped.jsonl: {"clip":"demo_seg0001","filter":"non_target_script","value":"cjk","threshold":"latin","text":"你好 世界 今天 天气"}
metadata.csv:  demo_0001|Hello there friend|Hello there friend
```

**False-positive check (hand-run, covering the lens that errored):** default Tier-1
filters on the real `sample.wav` speech (tiny model) kept the clip, **dropped 0** — the
confidence (0.30), char-rate (5–25/s) and compression (2.4) defaults don't nuke clean
speech.

## PHASE D4 — Scale: playlists / channels / feeds → one merged dataset

- [ ] Batch every item through D1–D3 into a **single combined dataset** with a shared manifest, continuing past per-item failures
- [ ] Summary table + totals (clips, hours); progress; resumability if feasible (skip already-built clips by source_id+index)
- [ ] Tests: multi-item merge, continue-past-failure, totals

**Acceptance:** a small real playlist → one coherent dataset folder with merged manifest + totals. **STOP.**

## PHASE D5 — Optional diarization (single-speaker TTS)

- [ ] `hearsay[diarize]` extra (pyannote.audio / whisperx, lazy-imported); clean degrade with a clear warning when absent
- [ ] `--per-speaker` (one sub-dataset/manifest per speaker) and `--dominant-speaker` (keep the most-spoken speaker)
- [ ] HF token via `HF_TOKEN`/`--hf-token`; on auth failure print exact remediation (accept conditions on both gated repos + create read token)
- [ ] Tests: mixed-speaker default + degrade path; speaker-label assignment by max temporal overlap (mock/fixture)

**Acceptance:** with the extra, a 2-speaker fixture yields per-speaker (or dominant-only) clips; without it, a clear mixed-speaker warning. **STOP.**

## PHASE D6 — Both front ends + docs

- [ ] CLI flags fully wired: `--out`, `--format`, `--sample-rate`, `--segment-min/max`, `--lang`, `--normalize`, `--diarize`/`--per-speaker`/`--dominant-speaker`, `--vad`, filter toggles
- [ ] Web UI "Dataset mode": choose dataset output + options, run, download dataset as a zip; hint that big playlists are better via CLI
- [ ] README "Build training datasets" section (honest accuracy/rights notes, comparison to markdown mode); comparison table + topics updated

**Acceptance:** CLI + web-UI dataset runs both produce a loadable dataset; README path verified. **STOP.**

## PHASE D7 — Launch-ready polish

- [ ] End-to-end fresh-run test following only the README; CI green
- [ ] Tiny, license-clean example mini-dataset committed so people see the output shape
- [ ] Draft a short "what's new in v0.2" note
- [ ] Final summary: what shipped, what's blocked, sample dataset stats, suggested next ideas. **STOP.**

## Blockers

(none yet)
