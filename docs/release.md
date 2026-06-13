# Release checklist

A dry-run checklist for publishing hearsay to PyPI. **Nothing here publishes
automatically** — the final upload step is intentionally left for a human to run.

## Prerequisites

- A clean `main` with CI green.
- A PyPI account and an API token (store it in `~/.pypirc` or pass via env).
- [uv](https://docs.astral.sh/uv/) installed.

## 1. Bump the version

Edit `version` in `pyproject.toml` (e.g. `0.1.0` → `0.1.1`). hearsay reads its
version from the installed package metadata, so no other file needs changing.

```bash
git switch -c release/vX.Y.Z
# edit pyproject.toml version
uv lock                      # refresh uv.lock with the new version
```

## 2. Pre-flight checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/export_schema.py   # ensure docs/schema.json is in sync
git diff --exit-code docs/schema.json     # (no output = in sync)
```

## 3. Build

```bash
rm -rf dist
uv build                     # produces dist/hearsay-X.Y.Z.tar.gz and .whl
ls -la dist
```

Sanity-check the wheel installs and runs in a throwaway environment:

```bash
uv run --isolated --with dist/hearsay-X.Y.Z-py3-none-any.whl hearsay --version
```

## 4. Validate metadata

```bash
uvx twine check dist/*       # README renders, metadata is valid
```

## 5. Dry-run publish (TestPyPI)

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"
# then verify it installs from TestPyPI:
uv tool install --index https://test.pypi.org/simple/ hearsay
```

## 6. Publish (manual — do this deliberately)

```bash
uv publish --token "$PYPI_TOKEN"
```

## 7. Tag and release

```bash
git commit -am "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
gh release create vX.Y.Z --generate-notes
```

## Notes

- The `mcp` extra is optional; `pip install hearsay` stays light, and
  `pip install 'hearsay[mcp]'` adds the MCP server SDK.
- ffmpeg is a documented system requirement, not a Python dependency — it cannot
  be installed from PyPI, so the README install section calls it out per-OS.
