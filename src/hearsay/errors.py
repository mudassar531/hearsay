"""Error types with user-facing messages.

Every error carries a `hint` telling the user what to do next; the CLI
prints message + hint and exits nonzero without a traceback.
"""


class HearsayError(Exception):
    """Base class for all expected failures."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class InvalidSourceError(HearsayError):
    """The given source is not something hearsay can ingest."""


class VideoUnavailableError(HearsayError):
    """The video is private, deleted, region-locked, or otherwise unreachable."""


class NoCaptionsError(HearsayError):
    """The video has no captions in any language."""


class MetadataError(HearsayError):
    """yt-dlp could not fetch or parse video metadata."""
