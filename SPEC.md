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
