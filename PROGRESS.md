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

## PHASE 4 — Distribution hack: MCP server mode

- [ ] Add the official `mcp` Python SDK; `hearsay mcp` starts a stdio MCP server exposing two tools: `ingest_url(url, transcribe?, lang?)` and `ingest_file(path)` — both return the markdown string
- [ ] Keep MCP deps in an optional extra (`pip install hearsay[mcp]`) if they're heavy
- [ ] Round-trip test: spawn the server, call `ingest_file` on the fixture clip, assert markdown comes back
- [ ] README section: "Give your agent ears" — exact JSON config snippets for Claude Code and Claude Desktop

**Acceptance:** MCP round-trip test passes · config snippet verified syntactically.

## PHASE 5 — Launch kit

- [ ] Real README: hero one-liner, install (`uv tool install` / `pipx`), 30-second quickstart, output example, comparison table, badges
- [ ] `demo/record.sh` using vhs (or asciinema fallback); output `demo.gif` referenced at the top of README
- [ ] GitHub Actions CI: ruff + mypy + pytest on 3.11/3.12
- [ ] `CONTRIBUTING.md` + 3 written-up "good first issue" drafts in `docs/good-first-issues.md`
- [ ] `uv build` succeeds; dry-run publish checklist in `docs/release.md` (do not actually publish)
- [ ] `launch/show-hn.md`: a draft Show HN title + 150-word post, and a 4-tweet thread draft

**Acceptance:** fresh-clone test — `git clone` → install → ingest a URL works following only the README · CI green · demo gif renders.

## PHASE 6 — Stretch (DO NOT START without explicit approval)

Parked in `IDEAS.md`: speaker diarization via whisperX · `--frames` keyframe extraction · vector-store export helpers · web URL article fallback.

## Blockers

(none yet)
