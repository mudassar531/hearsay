"""Tests for friendly error mapping and messages (offline)."""

import pytest

from hearsay.captions import _no_captions_message
from hearsay.errors import (
    InvalidSourceError,
    MetadataError,
    VideoUnavailableError,
)
from hearsay.youtube import _map_ytdlp_error

URL = "https://www.youtube.com/watch?v=abcdefghijk"


@pytest.mark.parametrize(
    ("stderr", "expected_type", "needle"),
    [
        ("ERROR: [youtube] abc: Private video. Sign in", VideoUnavailableError, "private"),
        ("ERROR: [youtube] abc: Video unavailable", VideoUnavailableError, "unavailable"),
        (
            "ERROR: [youtube] abc: This video has been removed by the uploader",
            VideoUnavailableError,
            "unavailable",
        ),
        (
            "ERROR: Sign in to confirm your age. This video may be inappropriate",
            VideoUnavailableError,
            "age-restricted",
        ),
        ("ERROR: 'foo' is not a valid URL", InvalidSourceError, "YouTube"),
        ("ERROR: Unsupported URL: ftp://x", InvalidSourceError, "YouTube"),
        ("ERROR: HTTP Error 503: Service Unavailable", MetadataError, "yt-dlp"),
    ],
)
def test_map_ytdlp_error(stderr: str, expected_type: type, needle: str) -> None:
    error = _map_ytdlp_error(URL, stderr)
    assert isinstance(error, expected_type)
    assert error.hint  # every error tells the user what to do next
    assert needle.lower() in (error.message + " " + error.hint).lower()


def test_map_ytdlp_error_handles_empty_stderr() -> None:
    error = _map_ytdlp_error(URL, "")
    assert isinstance(error, MetadataError)
    assert error.hint


def test_no_captions_message_points_to_transcription() -> None:
    message, hint = _no_captions_message("abcdefghijk")
    assert "no captions" in message.lower()
    assert "--transcribe" in hint
