# Contributing to hearsay

Thanks for your interest! hearsay does one thing well — turn video & audio into
great markdown — and aims to keep doing exactly that. Contributions that sharpen
the core (output quality, more sources, fewer dependencies) are very welcome.

## Development setup

hearsay uses [uv](https://docs.astral.sh/uv/). You also need
[ffmpeg](README.md#requirements) on your PATH.

```bash
git clone https://github.com/mudassar531/hearsay
cd hearsay
uv sync                 # creates .venv and installs everything (incl. dev + mcp)
uv run hearsay --help
```

## Before you push

The same three checks CI runs must pass locally:

```bash
uv run ruff check .         # lint
uv run ruff format .        # format (CI checks with --format --check)
uv run mypy                 # types
uv run pytest               # tests
```

CI runs these on Python 3.11, 3.12 and 3.13.

## Ground rules

- **Tests never touch the network.** Record real API/transcript payloads once
  into `tests/fixtures/` (see `scripts/record_fixtures.py`) and test against
  them. A `tests/conftest.py` guard fails any test that tries to open a
  non-loopback connection. The Whisper integration tests load a cached model
  with `local_files_only=True` and **skip** when it isn't cached, so the suite
  is green offline.
- **Type hints and docstrings** on public functions. mypy is configured
  leniently; untyped third-party deps get a scoped override in `pyproject.toml`.
- **Helpful errors.** Every failure should tell the user what to do next (raise
  a `HearsayError` subclass with a `hint`); the CLI prints it without a
  traceback.
- **Minimal dependencies.** Justify any new dependency in `DECISIONS.md`.
- **One thing well.** Summarization, vector stores, GUIs, and re-hosting content
  are explicit non-goals — see `IDEAS.md` for parked ideas.

## How the pipeline fits together

```
source ─▶ metadata + (captions | whisper segments) ─▶ group_segments
       ─▶ sectionize ─▶ Document ─▶ render_markdown / Transcript JSON
```

- `youtube.py` / `feeds.py` — fetch metadata, captions, audio, feeds, playlists
- `captions.py` / `transcribe.py` — turn a source into timed `Segment`s
- `grouping.py` — the core: segments → readable paragraphs (heavily tested)
- `sectioning.py` — chapters or ~5-minute time windows → `Section`s
- `render.py` / `models.py` — `Document` → markdown / `Transcript` JSON
- `pipeline.py` — wires it together; `batch.py` — playlist/feed batch runs
- `cli.py` — the `hearsay` command group; `mcp_server.py` — the MCP server

## Updating the JSON schema

If you change the `Transcript` model, regenerate the committed schema (a test
enforces it stays in sync):

```bash
uv run python scripts/export_schema.py
```

## Pull requests

Keep PRs focused, add tests for new behavior, and update `README.md` /
`DECISIONS.md` when relevant. Conventional commit messages (`feat:`, `fix:`,
`test:`, `docs:`, `chore:`) are appreciated.

## Releasing

Releases are published to PyPI by tag, using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — there is no API token
in the repo, on CI, or on anyone's laptop.

One-time setup on PyPI (owner only): **Manage project → Publishing → Add a new
publisher**, with owner `mudassar531`, repository `hearsay`, workflow
`release.yml`, environment `pypi`.

To cut a release:

```bash
# 1. bump the version in pyproject.toml and add a CHANGELOG entry, then merge to main
# 2. tag it — the tag must match pyproject.toml or the workflow fails before uploading
git tag v0.4.0 && git push origin v0.4.0
```

The workflow re-runs lint, types and tests, builds the wheel and sdist, installs the
wheel into a clean environment and runs it, and only then publishes. PyPI never lets a
version be reused, so every gate runs *before* the upload.
