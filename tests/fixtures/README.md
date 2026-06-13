# Test fixtures

Real payloads recorded once from YouTube so the test suite never touches the
network. Re-record with `uv run python scripts/record_fixtures.py` (network
required), then review the diff before committing.

| Video | Files | Why it was chosen |
| --- | --- | --- |
| `zjkBMFhNj_g` — Andrej Karpathy, "[1hr Talk] Intro to Large Language Models" (59:48) | `.meta.json`, `.transcript.json` | 21 chapters → chapter-aware sectioning; auto-generated captions only → generated-caption fallback path |
| `rStL7niR7gs` — CGP Grey, "You Would Be a Terrible Leader" (18:13) | `.meta.json`, `.transcript.json` | No chapters → time-based sectioning; manually created English captions → manual-preferred path |

`*.meta.json` is `yt-dlp --dump-json` output trimmed to the fields hearsay
consumes (the dropped keys are multi-hundred-KB format/thumbnail/caption-URL
blobs irrelevant to tests). `*.transcript.json` is the youtube-transcript-api
fetch result (`to_raw_data()` snippets) plus the list of available transcripts
for caption-selection tests.

Other fixtures:

| File | What | Used by |
| --- | --- | --- |
| `sample.wav` + `sample.txt` | 4.7s OS-TTS speech clip (macOS `say` + ffmpeg) and its expected text | whisper transcription tests |
| `podcast.xml` | Real RSS feed (Talk Python To Me), trimmed to the first 3 `<item>`s | `parse_feed` tests |
| `playlist.json` | Real `yt-dlp -J --flat-playlist` output (Pittsburgh ML Summit '19), trimmed to 4 entries and the fields hearsay uses | `parse_playlist_json` tests |
