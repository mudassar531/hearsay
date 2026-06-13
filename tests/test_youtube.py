"""Tests for YouTube URL parsing and metadata parsing (offline, fixture-driven)."""

import json
from pathlib import Path

import pytest

from hearsay.errors import PlaylistError
from hearsay.youtube import (
    _pick_downloaded_audio,
    extract_playlist_id,
    extract_video_id,
    parse_metadata,
    parse_playlist_json,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_meta(video_id: str) -> dict:
    return json.loads((FIXTURES / f"{video_id}.meta.json").read_text())


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("http://youtube.com/watch?v=zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("https://www.youtube.com/watch?list=PL123&v=zjkBMFhNj_g&t=42", "zjkBMFhNj_g"),
        ("https://youtu.be/rStL7niR7gs", "rStL7niR7gs"),
        ("https://youtu.be/rStL7niR7gs?t=120", "rStL7niR7gs"),
        ("https://www.youtube.com/shorts/rStL7niR7gs", "rStL7niR7gs"),
        ("https://www.youtube.com/embed/rStL7niR7gs", "rStL7niR7gs"),
        ("https://www.youtube.com/live/rStL7niR7gs", "rStL7niR7gs"),
        ("https://www.youtube-nocookie.com/embed/rStL7niR7gs", "rStL7niR7gs"),
    ],
)
def test_extract_video_id(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/watch?v=zjkBMFhNj_g",
        "https://www.youtube.com/playlist?list=PL123",
        "not a url at all",
        "zjkBMFhNj_g",  # a bare id is not a URL
        "https://youtu.be/tooshort",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQextra",  # 11+ chars: reject, don't truncate
        "",
    ],
)
def test_extract_video_id_rejects_non_video_urls(url: str) -> None:
    assert extract_video_id(url) is None


def test_parse_metadata_with_chapters() -> None:
    raw = load_meta("zjkBMFhNj_g")
    meta = parse_metadata(raw, "https://www.youtube.com/watch?v=zjkBMFhNj_g")
    assert meta.title == "[1hr Talk] Intro to Large Language Models"
    assert meta.channel == "Andrej Karpathy"
    assert meta.video_id == "zjkBMFhNj_g"
    assert meta.duration_s == 3588.0
    assert len(meta.chapters) == 21
    first = meta.chapters[0]
    assert first.start_s == 0.0
    assert first.end_s > first.start_s
    # Chapters must be in order and contiguous enough to section against.
    starts = [c.start_s for c in meta.chapters]
    assert starts == sorted(starts)


def test_parse_metadata_without_chapters() -> None:
    raw = load_meta("rStL7niR7gs")
    meta = parse_metadata(raw, "https://www.youtube.com/watch?v=rStL7niR7gs")
    assert meta.title == "You Would Be a Terrible Leader"
    assert meta.channel == "CGP Grey"
    assert meta.chapters == []
    assert meta.duration_s == 1093.0


def test_parse_metadata_fills_safe_defaults() -> None:
    meta = parse_metadata({"id": "abc123def45", "chapters": None}, "src")
    assert meta.title == "Untitled"
    assert meta.channel == "Unknown"
    assert meta.duration_s == 0.0
    assert meta.chapters == []


def test_parse_metadata_missing_id_raises_friendly_error() -> None:
    from hearsay.errors import MetadataError

    with pytest.raises(MetadataError) as excinfo:
        parse_metadata({"title": "no id here"}, "https://youtu.be/x")
    assert excinfo.value.hint  # tells the user what to do next


def test_pick_downloaded_audio_ignores_sidecars_and_picks_largest(tmp_path: Path) -> None:
    (tmp_path / "vid.webp").write_bytes(b"thumbnail")
    (tmp_path / "vid.info.json").write_bytes(b"{}")
    (tmp_path / "vid.m4a.part").write_bytes(b"partial-download")
    audio = tmp_path / "vid.m4a"
    audio.write_bytes(b"the real audio payload is the biggest")
    assert _pick_downloaded_audio(tmp_path) == audio


def test_pick_downloaded_audio_none_when_only_sidecars(tmp_path: Path) -> None:
    (tmp_path / "vid.webp").write_bytes(b"x")
    assert _pick_downloaded_audio(tmp_path) is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/playlist?list=PLabc123", "PLabc123"),
        ("https://youtube.com/playlist?list=PLxyz&foo=1", "PLxyz"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc", None),  # video, not playlist
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", None),
        ("https://m.youtube.com/playlist?list=PLabc", "PLabc"),
        ("https://example.com/playlist?list=PLabc", "PLabc"),
        # A watch URL with "/playlist" buried in a query param must NOT misroute.
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&next=/playlist?list=EVIL", None),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&foo=/playlist&list=PLXYZ", None),
    ],
)
def test_extract_playlist_id(url: str, expected: str | None) -> None:
    assert extract_playlist_id(url) == expected


def test_parse_playlist_json_fixture() -> None:
    data = json.loads((FIXTURES / "playlist.json").read_text())
    title, entries = parse_playlist_json(data, "https://www.youtube.com/playlist?list=PL")
    assert title == "Pittsburgh ML Summit ‘19"
    assert len(entries) == 4
    first = entries[0]
    assert first.video_id == "CvTApw9X8aA"
    assert first.title.startswith("Welcome")
    assert first.url == "https://www.youtube.com/watch?v=CvTApw9X8aA" or first.url.startswith(
        "http"
    )


def test_parse_playlist_json_empty_raises() -> None:
    with pytest.raises(PlaylistError):
        parse_playlist_json({"title": "Empty", "entries": []}, "url")
