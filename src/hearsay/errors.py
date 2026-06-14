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


class CaptionsError(HearsayError):
    """Captions exist but could not be retrieved (e.g. blocked network)."""


class MetadataError(HearsayError):
    """yt-dlp could not fetch or parse video metadata."""


class OutputWriteError(HearsayError):
    """The rendered markdown could not be written to the output path."""


class TranscriptionError(HearsayError):
    """Local whisper transcription failed (model load or audio decode)."""


class AudioDownloadError(HearsayError):
    """yt-dlp could not download the audio stream for transcription."""


class FeedError(HearsayError):
    """A podcast RSS feed could not be fetched or parsed."""


class PlaylistError(HearsayError):
    """A YouTube playlist could not be listed."""


class AudioExportError(HearsayError):
    """ffmpeg/ffprobe was unavailable or a clip could not be sliced/probed."""
