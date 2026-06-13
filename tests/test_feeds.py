"""Tests for podcast feed parsing (offline, against a recorded RSS fixture)."""

from pathlib import Path

from hearsay.feeds import _duration_seconds, _suffix_for, parse_feed

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_real_feed_fixture() -> None:
    feed = parse_feed((FIXTURES / "podcast.xml").read_bytes())
    assert feed.title == "Talk Python To Me"
    assert len(feed.episodes) == 3
    for episode in feed.episodes:
        assert episode.title
        assert episode.audio_url and episode.audio_url.startswith("http")
        assert episode.duration_s > 0  # itunes:duration present on this feed


def test_parse_feed_handles_missing_enclosure() -> None:
    rss = """<?xml version="1.0"?>
    <rss version="2.0"><channel><title>Empty Show</title>
      <item><title>No media here</title><guid>abc</guid></item>
    </channel></rss>"""
    feed = parse_feed(rss)
    assert feed.title == "Empty Show"
    assert len(feed.episodes) == 1
    assert feed.episodes[0].audio_url is None


def test_parse_feed_prefers_audio_enclosure() -> None:
    rss = """<?xml version="1.0"?>
    <rss version="2.0"><channel><title>Mixed</title>
      <item><title>Ep</title>
        <enclosure url="http://x/cover.jpg" type="image/jpeg"/>
        <enclosure url="http://x/ep.mp3" type="audio/mpeg"/>
      </item>
    </channel></rss>"""
    feed = parse_feed(rss)
    assert feed.episodes[0].audio_url == "http://x/ep.mp3"


def test_duration_parsing() -> None:
    assert _duration_seconds({"itunes_duration": "90"}) == 90.0
    assert _duration_seconds({"itunes_duration": "1:30"}) == 90.0
    assert _duration_seconds({"itunes_duration": "1:02:03"}) == 3723.0
    assert _duration_seconds({}) == 0.0
    assert _duration_seconds({"itunes_duration": "garbage"}) == 0.0


def test_suffix_for_uses_content_type_then_url() -> None:
    assert _suffix_for("http://x/ep", "audio/mpeg") == ".mp3"
    assert _suffix_for("http://x/ep.m4a", None) == ".m4a"
    assert _suffix_for("http://x/ep?token=1", "application/octet-stream") == ".mp3"
