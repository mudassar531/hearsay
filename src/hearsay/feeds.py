"""Podcast RSS feed parsing (feedparser) and episode audio download (urllib)."""

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import feedparser

from hearsay.errors import AudioDownloadError, FeedError

_USER_AGENT = "hearsay/0.1 (+https://github.com/mudassar531/hearsay)"
_DOWNLOAD_TIMEOUT_S = 600
# Content-type -> file extension for naming the downloaded episode.
_AUDIO_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


class Episode(NamedTuple):
    """One podcast episode: its title, audio URL, and (best-effort) duration."""

    title: str
    audio_url: str | None
    duration_s: float
    guid: str


class Feed(NamedTuple):
    """A parsed podcast feed: the show title and its episodes (feed order)."""

    title: str
    episodes: list[Episode]


def parse_feed(content: str | bytes) -> Feed:
    """Parse raw RSS/Atom feed content into a Feed (pure; no network)."""
    return _extract(feedparser.parse(content))


def fetch_feed(url: str) -> Feed:
    """Fetch and parse a podcast feed URL (network).

    Raises FeedError when the URL cannot be fetched or has no episodes.
    """
    try:
        parsed = feedparser.parse(url, agent=_USER_AGENT)
    except Exception as exc:  # feedparser usually self-handles, but never leak a traceback
        raise FeedError(
            f"Could not fetch feed {url}: {exc}",
            hint="Check the URL is reachable and points to an RSS/Atom feed.",
        ) from exc
    if not parsed.entries:
        reason = ""
        if getattr(parsed, "bozo", 0) and getattr(parsed, "bozo_exception", None):
            reason = f" ({parsed.bozo_exception})"
        raise FeedError(
            f"No podcast episodes found at {url}{reason}.",
            hint="Check the URL points to an RSS/Atom feed (often a .xml or .rss link).",
        )
    return _extract(parsed)


def _extract(parsed: feedparser.FeedParserDict) -> Feed:
    show = str(parsed.feed.get("title") or "Podcast")
    episodes = [
        Episode(
            title=str(entry.get("title") or "Untitled episode"),
            audio_url=_enclosure_url(entry),
            duration_s=_duration_seconds(entry),
            guid=str(entry.get("id") or entry.get("link") or ""),
        )
        for entry in parsed.entries
    ]
    return Feed(title=show, episodes=episodes)


def _enclosure_url(entry: feedparser.FeedParserDict) -> str | None:
    """The first audio enclosure URL for an entry, or any enclosure as fallback."""
    enclosures = entry.get("enclosures") or []
    for enc in enclosures:
        if str(enc.get("type", "")).startswith("audio") and enc.get("href"):
            return str(enc["href"])
    for enc in enclosures:
        if enc.get("href"):
            return str(enc["href"])
    return None


def _duration_seconds(entry: feedparser.FeedParserDict) -> float:
    """Parse itunes:duration, which may be seconds or H:MM:SS / MM:SS."""
    raw = entry.get("itunes_duration")
    if not raw:
        return 0.0
    raw = str(raw).strip()
    try:
        if ":" in raw:
            parts = [float(p) for p in raw.split(":")]
            total = 0.0
            for part in parts:
                total = total * 60 + part
        else:
            total = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, total)  # a malformed/negative duration falls back to "unknown"


def download_episode(url: str, dest_dir: Path) -> Path:
    """Download an episode's audio into ``dest_dir`` and return its path.

    The caller owns ``dest_dir`` and must delete it. Raises AudioDownloadError.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            suffix = _suffix_for(url, response.headers.get("Content-Type"))
            expected = _content_length(response.headers.get("Content-Length"))
            dest = dest_dir / f"episode{suffix}"
            with dest.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise AudioDownloadError(
            f"Could not download episode audio from {url}: {exc}",
            hint="Check the feed's media URL is reachable, then try again.",
        ) from exc
    written = dest.stat().st_size
    if written == 0:
        raise AudioDownloadError(
            f"Downloaded an empty file from {url}.",
            hint="The episode media URL may be broken; try another episode.",
        )
    if expected is not None and written < expected:
        raise AudioDownloadError(
            f"Incomplete download from {url}: got {written} of {expected} bytes.",
            hint="The connection dropped mid-download; try again.",
        )
    return dest


def _content_length(value: str | None) -> int | None:
    """Parse a Content-Length header into a positive int, or None if absent/bad."""
    if value and value.isdigit():
        size = int(value)
        return size if size > 0 else None
    return None


def _suffix_for(url: str, content_type: str | None) -> str:
    """Pick a file extension from the Content-Type, falling back to the URL/.mp3."""
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base in _AUDIO_EXTENSIONS:
            return _AUDIO_EXTENSIONS[base]
    url_suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if url_suffix in set(_AUDIO_EXTENSIONS.values()):
        return url_suffix
    return ".mp3"
