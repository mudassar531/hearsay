# Good first issues

Three self-contained starter tasks for new contributors. Each names the files to
touch and how to verify. Read [CONTRIBUTING.md](../CONTRIBUTING.md) first.

---

## 1. Add a `--quiet` flag to suppress the success panel

**Why:** When scripting hearsay or running it in a pipeline, the rich success
panel is noise. A `--quiet` flag should suppress it (errors still print).

**Where:**
- `src/hearsay/cli.py` — add a `--quiet` option to the `ingest` command; guard
  the `_print_success(...)` call (and the batch summary table) on `not quiet`.
- Thread it through `_Options`.

**Verify:**
- `uv run hearsay <url> --quiet -o out.md` writes the file and prints nothing on
  success.
- Add a test in `tests/test_cli.py` (monkeypatch the ingester) asserting the
  output panel is absent and the file is still written.

**Scope:** small. Touches only the CLI layer.

---

## 2. Recognize more YouTube URL shapes in `extract_video_id`

**Why:** `extract_video_id` (in `src/hearsay/youtube.py`) handles `watch?v=`,
`youtu.be/`, `embed/`, `shorts/`, and `live/`. It does not yet handle
`music.youtube.com` watch URLs or the `/v/<id>` legacy embed form.

**Where:**
- `src/hearsay/youtube.py` — extend `_URL_PATTERNS` (keep the 11-char id with the
  trailing `(?![A-Za-z0-9_-])` boundary so over-long ids are still rejected).

**Verify:**
- Add parametrized cases to `tests/test_youtube.py::test_extract_video_id` for
  the new shapes, and a rejection case for a malformed one.
- `uv run pytest tests/test_youtube.py` passes.

**Scope:** small. Pure function + tests; no network.

---

## 3. Let `--lang` accept a comma-separated preference list for captions

**Why:** A user may want "Spanish, else English, else anything." Today `--lang`
takes one code. Accepting `--lang es,en` would try each in order before falling
back to any available track.

**Where:**
- `src/hearsay/captions.py` — `select_transcript` currently takes one `requested`
  language; generalize it to take an ordered list and pick the first that
  matches (manual preferred over generated within each language).
- `src/hearsay/cli.py` / `pipeline.py` — split the `--lang` value on commas
  before passing it down.

**Verify:**
- Extend `tests/test_captions.py` with a fixture-driven case asserting the
  preference order is honored and the existing single-language behavior is
  unchanged.
- `uv run pytest tests/test_captions.py` passes.

**Scope:** medium. Pure selection logic + a thin CLI change; well covered by the
existing offline transcript fixtures.
