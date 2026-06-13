# PROGRESS

Project: **hearsay** (see DECISIONS.md — `earshot` was taken on PyPI).
Rule: a box is ticked only after its verification command ran and passed. One commit per task.

## STEP ZERO

- [x] Save the prompt verbatim to `SPEC.md`
- [x] Create `PROGRESS.md` with every phase and task as checkboxes
- [x] Create `DECISIONS.md` with a header

## PHASE 0 — Scaffold (target: everything runs)

- [x] Check name availability on PyPI; if taken, propose 5 alternatives, pick the best, use it everywhere — `earshot` taken (HTTP 200); picked `hearsay` (HTTP 404 = free); see DECISIONS.md
- [x] `uv init` with `src/hearsay/` layout; pin Python 3.11+ — verified: `uv run python -c "import hearsay"` OK on Python 3.11.15
- [x] Typer CLI skeleton: `hearsay --version`, `hearsay --help` with the one-liner pitch — verified: `uv run hearsay --version` → `hearsay 0.1.0`; `--help` shows pitch
- [x] Configure ruff + mypy + pytest; add one placeholder test — verified: `ruff check` clean, `mypy` clean, `pytest` 2 passed
- [x] MIT `LICENSE`, stub `README.md` (one-liner + "under construction"), `.gitignore`, `git init` + first commit — verified: all files present; .gitignore + git init were done in STEP ZERO (see DECISIONS.md)

**Acceptance:** `uv run hearsay --version` prints version · `uv run pytest` passes · `uv run ruff check .` clean.

## PHASE 1 — The magic moment: YouTube → markdown (captions path)

- [ ] `hearsay <youtube-url>`: fetch metadata via `yt-dlp --dump-json` (title, channel, duration, chapters) — no media download
- [ ] Fetch captions via youtube-transcript-api (prefer manual subs over auto-generated; `--lang` flag, default `en` with fallback to first available)
- [ ] Implement the paragraph-grouping algorithm (pause threshold + sentence boundaries + 40–120 word targets) as a pure, heavily-tested function
- [ ] Chapter-aware sectioning: chapters → `##` headings; no chapters → time-based sections every ~5 minutes
- [ ] Markdown renderer producing exactly the OUTPUT FORMAT in SPEC.md; `-o/--output` flag, default `./<video-id>.md`
- [ ] Friendly errors: no captions ("transcription lands in Phase 2"), private/unavailable/invalid URL
- [ ] Fixtures: record 2 real transcript+metadata payloads (one video with chapters, one without) into `tests/fixtures/`; unit-test grouping, sectioning, rendering against them
- [ ] Run end-to-end on 2 real videos; commit outputs to `examples/`

**Acceptance:** `uv run hearsay https://www.youtube.com/watch?v=<id>` produces a clean file in <10s · tests pass offline · `examples/` contains 2 real outputs.

## PHASE 2 — Ears: whisper fallback + local files

- [ ] `hearsay <file.mp3|.mp4|.wav|.m4a>`: extract audio via ffmpeg if needed → faster-whisper → same markdown pipeline (whisper segments → same grouping function)
- [ ] `--transcribe` flag to force whisper on YouTube URLs (yt-dlp audio-only download → transcribe); auto-fallback to this path when no captions exist, with a clear "transcribing locally, this takes a few minutes" notice
- [ ] `--model` flag (default `small`); rich progress bar during transcription; clean temp-file handling (always delete downloaded audio)
- [ ] Add a short (<30s) public-domain audio clip to `tests/fixtures/`; integration test transcribes it with `tiny`
- [ ] Document ffmpeg as a requirement with install commands per OS in README

**Acceptance:** `uv run hearsay tests/fixtures/sample.wav` produces correct markdown · captionless YouTube video works end-to-end via fallback.

## PHASE 3 — Scale: podcasts, playlists, JSON

- [ ] Podcast RSS: `hearsay <feed-url>` lists episodes (rich table); `--latest`, `--episode N`, `--all --limit N`
- [ ] YouTube playlists: same flags; batch mode writes to `--output-dir` (default `./hearsay-out/`), continues past per-item failures, prints a summary table
- [ ] `--json` sidecar implementing the `Transcript` pydantic model; export the JSON schema to `docs/schema.json`
- [ ] Tests: feed parsing fixture, batch-failure handling, JSON schema validation

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
