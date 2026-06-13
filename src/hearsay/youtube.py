"""YouTube URL parsing and metadata fetching via yt-dlp (no media download)."""

import json
import re
import subprocess
import sys

from hearsay.errors import (
    HearsayError,
    InvalidSourceError,
    MetadataError,
    VideoUnavailableError,
)
from hearsay.models import Chapter, SourceMetadata

_URL_PATTERNS = [
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:[^#\s]*&)?v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})"),
]

_METADATA_TIMEOUT_S = 60


def extract_video_id(url: str) -> str | None:
    """Return the 11-character video id from a YouTube URL, or None if it isn't one."""
    for pattern in _URL_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def fetch_raw_metadata(url: str) -> dict:
    """Fetch video metadata as a dict via `yt-dlp --dump-json` (no download).

    Raises VideoUnavailableError, InvalidSourceError, or MetadataError with
    actionable messages when yt-dlp fails.
    """
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--dump-json",
                "--no-warnings",
                "--no-playlist",
                "--",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=_METADATA_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise MetadataError(
            f"Timed out fetching metadata for {url} after {_METADATA_TIMEOUT_S}s.",
            hint="Check your network connection and try again.",
        ) from None
    if proc.returncode != 0:
        raise _map_ytdlp_error(url, proc.stderr)
    try:
        raw: dict = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MetadataError(
            f"yt-dlp returned unparseable metadata for {url}.",
            hint="Try updating yt-dlp: uv sync --upgrade-package yt-dlp",
        ) from exc
    return raw


def _map_ytdlp_error(url: str, stderr: str) -> HearsayError:
    """Translate yt-dlp stderr into a friendly hearsay error."""
    lowered = stderr.lower()
    if "private video" in lowered:
        return VideoUnavailableError(
            f"This video is private: {url}",
            hint="Only the uploader can access it; try a public video.",
        )
    if "video unavailable" in lowered or "this video has been removed" in lowered:
        return VideoUnavailableError(
            f"This video is unavailable (removed, region-locked, or never existed): {url}",
            hint="Open it in a browser to confirm it plays, then try again.",
        )
    if "age" in lowered and "confirm" in lowered:
        return VideoUnavailableError(
            f"This video is age-restricted and cannot be fetched anonymously: {url}",
            hint="Try a video that does not require sign-in.",
        )
    if "is not a valid url" in lowered or "unsupported url" in lowered:
        return InvalidSourceError(
            f"Not a URL yt-dlp recognizes: {url}",
            hint="Pass a YouTube video URL like https://www.youtube.com/watch?v=...",
        )
    tail = stderr.strip().splitlines()[-1] if stderr.strip() else "no error output"
    return MetadataError(
        f"yt-dlp failed for {url}: {tail}",
        hint=(
            "If the video plays in a browser, try updating yt-dlp: uv sync --upgrade-package yt-dlp"
        ),
    )


def parse_metadata(raw: dict, source: str) -> SourceMetadata:
    """Build SourceMetadata from a yt-dlp --dump-json payload (pure)."""
    chapters = [
        Chapter(
            title=str(ch.get("title") or "Untitled"),
            start_s=float(ch.get("start_time") or 0.0),
            end_s=float(ch.get("end_time") or 0.0),
        )
        for ch in (raw.get("chapters") or [])
    ]
    return SourceMetadata(
        title=str(raw.get("title") or "Untitled"),
        source=source,
        channel=str(raw.get("channel") or raw.get("uploader") or "Unknown"),
        duration_s=float(raw.get("duration") or 0.0),
        video_id=str(raw["id"]),
        chapters=chapters,
    )
