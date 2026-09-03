"""Batch ingestion: episode/entry selection and a failure-tolerant run loop.

Shared by podcast feeds and YouTube playlists. The pieces here are pure or
injectable so batch behaviour (selection, continue-past-failure) is tested
offline without touching the network.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hearsay.errors import InvalidSourceError
from hearsay.models import Document

_SLUG_SPACES = re.compile(r"[\s_-]+")


def _slug_char(ch: str) -> bool:
    # ``\w`` drops combining marks, and an abugida title without its vowel signs is a
    # different word: "বাংলা সংবাদ" came out as "বল-সবদ". Marks (category M*) stay.
    return ch.isalnum() or ch.isspace() or ch in "-_" or unicodedata.category(ch)[0] == "M"


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
    cleaned = "".join(ch for ch in text if _slug_char(ch)).strip().lower()
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
    if limit is not None and limit < 1:
        raise InvalidSourceError(
            f"--limit must be 1 or more (got {limit}).",
            hint="Use --limit N with N >= 1, or omit it to ingest everything.",
        )
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


def ensure_unique_slugs(items: list[BatchItem]) -> None:
    """Make every item's slug unique in place, suffixing collisions -2, -3, ...

    Distinct items can slugify to the same stem (e.g. two episode titles that
    differ only in punctuation); without this they would overwrite each other's
    output files. The first occurrence keeps the base slug.
    """
    used: set[str] = set()
    for item in items:
        slug = item.slug
        n = 1
        while slug in used:
            n += 1
            slug = f"{item.slug}-{n}"
        used.add(slug)
        item.slug = slug


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
