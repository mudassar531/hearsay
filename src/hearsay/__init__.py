"""hearsay — crawl4ai for video & audio.

Turn any YouTube video, podcast episode, or local recording into clean,
timestamped, LLM-ready markdown.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hearsay")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"
