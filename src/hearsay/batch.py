"""Batch ingestion: episode/entry selection and a failure-tolerant run loop.

Shared by podcast feeds and YouTube playlists. The pieces here are pure or
injectable so batch behaviour (selection, continue-past-failure) is tested
offline without touching the network.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hearsay.errors import InvalidSourceError
from hearsay.models import Document

_SLUG_STRIP = re.compile(r"[^\w\s-]")
_SLUG_SPACES = re.compile(r"[\s_-]+")


@dataclass
class BatchItem:
    """One unit of batch work: how to label, name, and produce its document."""

    title: str
    slug: str
    ingest: Callable[[], Document]


@dataclass
class BatchResult:
    """The outcome of ingesting one BatchItem."""

    title: str
    ok: bool
    output: Path | None = None
    error: str | None = None


def slugify(text: str, *, fallback: str = "episode") -> str:
    """Turn a title into a filesystem-safe, lowercase, hyphenated stem."""
    cleaned = _SLUG_STRIP.sub("", text).strip().lower()
    slug = _SLUG_SPACES.sub("-", cleaned).strip("-")
    return (slug or fallback)[:80]


def select(
    items: list[BatchItem],
    *,
    latest: bool,
    episode: int | None,
    all_: bool,
    limit: int | None,
) -> list[BatchItem]:
    """Pick which items to ingest from a selection of flags (pure).

    Precedence: ``episode`` (1-indexed) > ``latest`` (first item) > ``all_``
    (optionally capped by ``limit``). With no flag, returns [] — the caller
    lists the items instead of ingesting.
    """
    count = len(items)
    if episode is not None:
        if episode < 1 or episode > count:
            raise InvalidSourceError(
                f"--episode {episode} is out of range; this source has {count} item(s).",
                hint=f"Pass a number from 1 to {count}, or use --latest / --all.",
            )
        return [items[episode - 1]]
    if latest:
        return items[:1]
    if all_:
        return items[:limit] if limit is not None else items
    return []


def run_batch(
    items: list[BatchItem],
    write: Callable[[Document, str], Path],
    *,
    on_item: Callable[[int, int, BatchItem], None] | None = None,
) -> list[BatchResult]:
    """Ingest each item, writing its output, and never stop on a single failure.

    ``write(document, slug)`` persists one document and returns its path.
    Returns a result per item (in order) for a summary table.
    """
    results: list[BatchResult] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        if on_item is not None:
            on_item(index, total, item)
        try:
            document = item.ingest()
            output = write(document, item.slug)
            results.append(BatchResult(title=item.title, ok=True, output=output))
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep going; one bad item must not abort the batch
            message = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
            results.append(BatchResult(title=item.title, ok=False, error=message))
    return results
