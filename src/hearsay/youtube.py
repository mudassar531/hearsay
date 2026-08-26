"""YouTube URL parsing and metadata fetching via yt-dlp (no media download)."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import ParseResult, parse_qs, urlparse

from hearsay.errors import (
    AudioDownloadError,
    HearsayError,
    InvalidSourceError,
    MetadataError,
    PlaylistError,
    VideoUnavailableError,
)
from hearsay.models import Chapter, SourceMetadata

# A video id is exactly 11 chars; anything longer is a different video, not a prefix.
_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Hosts whose URLs may be handed to yt-dlp. The id is located by *parsing* the URL and
# matching this set against the real hostname — never by searching the string. A
# substring search matches "youtu.be/<id>" anywhere, including in the path or query of
# an attacker's URL ("http://169.254.169.254/#youtu.be/dQw4w9WgXcQ"), which would let a
# caller aim server-side fetches at cloud metadata or internal hosts.
_YOUTUBE_HOSTS = frozenset({"youtube.com", "youtube-nocookie.com"})
_SHORT_HOSTS = frozenset({"youtu.be"})
# /embed/<id>, /shorts/<id>, /live/<id> — the id is the second path segment.
_PATH_PREFIXES = frozenset({"embed", "shorts", "live"})


def _canonical_host(url: str) -> tuple[str, ParseResult] | None:
    """Parse ``url`` and return its ``(bare_host, parsed)``, or None if it isn't http(s).

    Strips a leading ``www.``/``m.`` so mobile and canonical links compare equal.
    """
    try:
        parsed = urlparse(url)
    except ValueError:  # malformed IPv6 literal, etc.
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    for prefix in ("www.", "m."):
        host = host.removeprefix(prefix)
    return host, parsed


def _valid_id(candidate: str) -> str | None:
    return candidate if _ID.match(candidate) else None


_METADATA_TIMEOUT_S = 60
_DOWNLOAD_TIMEOUT_S = 600


_PLAYLIST_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class PlaylistEntry(NamedTuple):
    """One video in a playlist: its id, title, and watch URL."""

    video_id: str
    title: str
    url: str


def extract_video_id(url: str) -> str | None:
    """Return the 11-character video id from a YouTube URL, or None if it isn't one.

    The hostname is parsed and matched exactly (modulo ``www.``/``m.``), so a URL that
    merely *contains* a YouTube-looking substring is rejected — this function is the
    gate that decides whether a caller-supplied URL gets fetched server-side.
    """
    resolved = _canonical_host(url)
    if resolved is None:
        return None
    host, parsed = resolved
    segments = [s for s in parsed.path.split("/") if s]
    if host in _SHORT_HOSTS:
        # youtu.be/<id>
        return _valid_id(segments[0]) if len(segments) == 1 else None
    if host not in _YOUTUBE_HOSTS:
        return None
    if len(segments) == 1 and segments[0] == "watch":
        values = parse_qs(parsed.query).get("v") or []
        return _valid_id(values[0]) if values else None
    if len(segments) == 2 and segments[0] in _PATH_PREFIXES:
        return _valid_id(segments[1])
    return None


def extract_playlist_id(url: str) -> str | None:
    """Return the playlist id from a canonical /playlist?list=... URL, else None.

    The path must be exactly ``/playlist`` (parsed, not substring-matched), so a
    ``watch?v=...&list=...`` URL — or a watch URL with ``/playlist`` buried in a
    query param — is treated as a single video and ingests just that video.
    """
    parsed = urlparse(url)
    if parsed.path.rstrip("/") != "/playlist":
        return None
    values = parse_qs(parsed.query).get("list")
    if not values:
        return None
    candidate = values[0]
    return candidate if _PLAYLIST_ID.match(candidate) else None


def fetch_playlist(url: str) -> tuple[str, list[PlaylistEntry]]:
    """List a playlist's entries via `yt-dlp --flat-playlist` (no per-video fetch).

    Returns (playlist_title, entries). Raises PlaylistError on failure.
    """
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "-J",
                "--flat-playlist",
                "--no-warnings",
                "--",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_METADATA_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise PlaylistError(
            f"Timed out listing playlist {url} after {_METADATA_TIMEOUT_S}s.",
            hint="Check your network connection and try again.",
        ) from None
    if proc.returncode != 0:
        mapped = _map_ytdlp_error(url, proc.stderr)
        raise PlaylistError(mapped.message, hint=mapped.hint) from None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PlaylistError(
            f"yt-dlp returned unparseable playlist data for {url}.",
            hint="Try updating yt-dlp: uv sync --upgrade-package yt-dlp",
        ) from exc
    return parse_playlist_json(data, url)


def parse_playlist_json(data: dict, url: str) -> tuple[str, list[PlaylistEntry]]:
    """Extract (playlist_title, entries) from a yt-dlp `-J --flat-playlist` dict (pure)."""
    entries: list[PlaylistEntry] = []
    for raw in data.get("entries") or []:
        if not raw or not raw.get("id"):
            continue
        video_id = str(raw["id"])
        entries.append(
            PlaylistEntry(
                video_id=video_id,
                title=str(raw.get("title") or video_id),
                url=str(raw.get("url") or f"https://www.youtube.com/watch?v={video_id}"),
            )
        )
    if not entries:
        raise PlaylistError(
            f"No videos found in playlist {url}.",
            hint="Check the playlist is public and not empty.",
        )
    return str(data.get("title") or "Playlist"), entries


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
            encoding="utf-8",  # yt-dlp emits UTF-8 JSON; do not trust the locale
            errors="replace",
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


def download_audio(url: str, dest_dir: Path) -> Path:
    """Download the best audio-only stream into ``dest_dir`` and return its path.

    Prefers an m4a stream (no re-encode, no ffmpeg needed); falls back to any
    best audio (e.g. webm/opus), which faster-whisper decodes directly. The
    caller owns ``dest_dir`` and is responsible for deleting it.
    """
    out_template = str(dest_dir / "%(id)s.%(ext)s")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "-f",
                "bestaudio[ext=m4a]/bestaudio/best",
                "--no-playlist",
                "--no-warnings",
                "-o",
                out_template,
                "--",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DOWNLOAD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise AudioDownloadError(
            f"Timed out downloading audio for {url} after {_DOWNLOAD_TIMEOUT_S}s.",
            hint="Check your network connection and try again.",
        ) from None
    if proc.returncode != 0:
        mapped = _map_ytdlp_error(url, proc.stderr)
        raise AudioDownloadError(mapped.message, hint=mapped.hint) from None
    audio = _pick_downloaded_audio(dest_dir)
    if audio is None:
        raise AudioDownloadError(
            f"yt-dlp reported success but produced no audio file for {url}.",
            hint="Try updating yt-dlp: uv sync --upgrade-package yt-dlp",
        )
    return audio


# Partial/sidecar files yt-dlp may leave next to the real media.
_NON_AUDIO_SUFFIXES = {".part", ".ytdl", ".json", ".jpg", ".jpeg", ".png", ".webp", ".vtt", ".srt"}


def _pick_downloaded_audio(dest_dir: Path) -> Path | None:
    """Choose the real media file in ``dest_dir`` (largest non-sidecar file).

    yt-dlp normally writes exactly one file, but a leftover ``.part`` or a
    sidecar (thumbnail/info/subtitle) can appear; ``iterdir`` order is
    arbitrary, so we filter known sidecars and pick the largest remaining file.
    """
    candidates = [
        p for p in dest_dir.iterdir() if p.is_file() and p.suffix.lower() not in _NON_AUDIO_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def parse_metadata(raw: dict, source: str) -> SourceMetadata:
    """Build SourceMetadata from a yt-dlp --dump-json payload (pure)."""
    video_id = raw.get("id")
    if not video_id:
        raise MetadataError(
            f"yt-dlp returned metadata with no video id for {source}.",
            hint="Try updating yt-dlp: uv sync --upgrade-package yt-dlp",
        )
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
        video_id=str(video_id),
        chapters=chapters,
    )
