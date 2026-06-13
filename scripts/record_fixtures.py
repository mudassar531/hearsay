"""Record real YouTube payloads into tests/fixtures/ (network required).

Usage: uv run python scripts/record_fixtures.py

Tests never touch the network; they run against the JSON files this script
records. Re-run it only to refresh fixtures, then review the diff.
"""

import json
import subprocess
import sys
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# (video_id, transcript language to record, prefer_manual)
VIDEOS = [
    ("zjkBMFhNj_g", "en", False),  # Karpathy: 21 chapters, auto captions only
    ("rStL7niR7gs", "en", True),  # CGP Grey: no chapters, manual captions
]

META_KEYS = [
    "id",
    "title",
    "channel",
    "uploader",
    "channel_id",
    "duration",
    "duration_string",
    "webpage_url",
    "upload_date",
    "language",
    "chapters",
]


def record_metadata(video_id: str) -> None:
    """Run yt-dlp --dump-json and store the fields hearsay consumes."""
    out = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--dump-json", "--no-warnings", "--", video_id],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    raw = json.loads(out)
    meta = {k: raw.get(k) for k in META_KEYS}
    # Keep only the language lists; the full caption dicts are URL blobs.
    meta["subtitles_languages"] = sorted((raw.get("subtitles") or {}).keys())
    meta["automatic_captions_languages"] = sorted((raw.get("automatic_captions") or {}).keys())
    path = FIXTURES / f"{video_id}.meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(meta.get('chapters') or [])} chapters)")


def record_transcript(video_id: str, language: str, prefer_manual: bool) -> None:
    """Fetch one transcript plus the list of available ones."""
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    available = [
        {"language_code": t.language_code, "language": t.language, "is_generated": t.is_generated}
        for t in transcript_list
    ]
    finder = (
        transcript_list.find_manually_created_transcript
        if prefer_manual
        else transcript_list.find_generated_transcript
    )
    transcript = finder([language])
    fetched = transcript.fetch()
    data = {
        "video_id": video_id,
        "language_code": fetched.language_code,
        "is_generated": fetched.is_generated,
        "available_transcripts": available,
        "snippets": fetched.to_raw_data(),
    }
    path = FIXTURES / f"{video_id}.transcript.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(data['snippets'])} snippets, generated={fetched.is_generated})")


def main() -> None:
    """Record metadata + transcript fixtures for both reference videos."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for video_id, language, prefer_manual in VIDEOS:
        record_metadata(video_id)
        record_transcript(video_id, language, prefer_manual)


if __name__ == "__main__":
    main()
