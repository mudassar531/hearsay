"""Caption fetching and selection via youtube-transcript-api."""

import html
import re
from typing import NamedTuple

from youtube_transcript_api import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from hearsay.errors import CaptionsError, NoCaptionsError, VideoUnavailableError
from hearsay.models import Segment

_WHITESPACE = re.compile(r"\s+")
# Standalone cues like [Music], (applause), [Laughter] carry no transcript text.
_NOISE_CUE = re.compile(r"[\[(][^\])]{0,40}[\])]")


class TranscriptInfo(NamedTuple):
    """The identity of one available transcript track."""

    language_code: str
    is_generated: bool


class CaptionResult(NamedTuple):
    """Fetched captions plus what was actually selected."""

    segments: list[Segment]
    language_code: str
    is_generated: bool


def select_transcript(available: list[TranscriptInfo], requested: str) -> TranscriptInfo | None:
    """Pick the best transcript track (pure).

    Preference order: manual in the requested language, generated in the
    requested language, any manual, any generated. Language matching accepts
    regional/named variants (`en` matches `en-US` and `en-eEY6OEpapPo`).
    """

    def matches(info: TranscriptInfo) -> bool:
        return info.language_code == requested or info.language_code.startswith(requested + "-")

    manual = [t for t in available if not t.is_generated]
    generated = [t for t in available if t.is_generated]
    for pool in (
        [t for t in manual if matches(t)],
        [t for t in generated if matches(t)],
        manual,
        generated,
    ):
        if pool:
            return pool[0]
    return None


def normalize_snippets(snippets: list[dict]) -> list[Segment]:
    """Clean raw snippets into Segments (pure).

    Unescapes HTML entities, collapses internal whitespace/newlines, drops
    empty snippets and standalone noise cues like ``[Music]``.
    """
    segments: list[Segment] = []
    for snippet in snippets:
        text = html.unescape(str(snippet["text"]))
        text = _WHITESPACE.sub(" ", text).strip()
        if not text or not _NOISE_CUE.sub("", text).strip():
            continue
        start = float(snippet["start"])
        duration = float(snippet.get("duration") or 0.0)
        segments.append(Segment(text=text, start_s=start, end_s=start + duration))
    return segments


def fetch_captions(video_id: str, language: str) -> CaptionResult:
    """Fetch the best available captions for a video (network).

    Raises NoCaptionsError when the video has no captions at all, and
    VideoUnavailableError / CaptionsError for access problems.
    """
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
        available = [TranscriptInfo(t.language_code, t.is_generated) for t in transcript_list]
        choice = select_transcript(available, language)
        if choice is None:
            raise NoCaptionsError(*_no_captions_message(video_id))
        transcript = next(
            t
            for t in transcript_list
            if (t.language_code, t.is_generated) == (choice.language_code, choice.is_generated)
        )
        fetched = transcript.fetch()
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        raise NoCaptionsError(*_no_captions_message(video_id)) from exc
    except VideoUnavailable as exc:
        raise VideoUnavailableError(
            f"Video {video_id} is unavailable (private, removed, or region-locked).",
            hint="Open it in a browser to confirm it plays, then try again.",
        ) from exc
    except RequestBlocked as exc:
        raise CaptionsError(
            "YouTube is blocking transcript requests from this network.",
            hint="Wait a few minutes and retry, or switch networks (cloud IPs are often blocked).",
        ) from exc
    return CaptionResult(
        segments=normalize_snippets(fetched.to_raw_data()),
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
    )


def _no_captions_message(video_id: str) -> tuple[str, str]:
    return (
        f"This video has no captions in any language: {video_id}",
        "Local Whisper transcription lands in Phase 2; until then, pick a captioned video.",
    )
