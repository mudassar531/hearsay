"""Tests for batch selection and the failure-tolerant run loop (offline)."""

from pathlib import Path

import pytest

from hearsay.batch import BatchItem, ensure_unique_slugs, run_batch, select, slugify
from hearsay.errors import InvalidSourceError, TranscriptionError
from hearsay.models import Document, SourceMetadata


def _doc(title: str) -> Document:
    meta = SourceMetadata(title=title, source="x", channel="c", duration_s=1.0, video_id=title)
    return Document(meta=meta, method="whisper-tiny", language="en", ingested_at="t")


def _items(n: int) -> list[BatchItem]:
    return [
        BatchItem(title=f"item {i}", slug=f"s{i}", ingest=lambda: _doc("x"))
        for i in range(1, n + 1)
    ]


# --- selection ------------------------------------------------------------


def test_select_none_returns_empty() -> None:
    assert select(_items(5), latest=False, episode=None, all_=False, limit=None) == []


def test_select_latest_returns_first() -> None:
    chosen = select(_items(5), latest=True, episode=None, all_=False, limit=None)
    assert [i.slug for i in chosen] == ["s1"]


def test_select_episode_is_one_indexed() -> None:
    chosen = select(_items(5), latest=False, episode=3, all_=False, limit=None)
    assert [i.slug for i in chosen] == ["s3"]


def test_select_all_with_limit() -> None:
    chosen = select(_items(5), latest=False, episode=None, all_=True, limit=2)
    assert [i.slug for i in chosen] == ["s1", "s2"]


def test_select_all_without_limit() -> None:
    chosen = select(_items(3), latest=False, episode=None, all_=True, limit=None)
    assert len(chosen) == 3


@pytest.mark.parametrize("bad", [0, -1, 6])
def test_select_episode_out_of_range(bad: int) -> None:
    with pytest.raises(InvalidSourceError) as excinfo:
        select(_items(5), latest=False, episode=bad, all_=False, limit=None)
    assert "out of range" in excinfo.value.message


@pytest.mark.parametrize("bad", [0, -1])
def test_select_rejects_non_positive_limit(bad: int) -> None:
    with pytest.raises(InvalidSourceError) as excinfo:
        select(_items(5), latest=False, episode=None, all_=True, limit=bad)
    assert "--limit" in excinfo.value.message


# --- slug uniqueness ------------------------------------------------------


def test_ensure_unique_slugs_disambiguates_collisions() -> None:
    items = [
        BatchItem(title="A", slug="dup", ingest=lambda: _doc("x")),
        BatchItem(title="B", slug="dup", ingest=lambda: _doc("x")),
        BatchItem(title="C", slug="dup", ingest=lambda: _doc("x")),
        BatchItem(title="D", slug="other", ingest=lambda: _doc("x")),
    ]
    ensure_unique_slugs(items)
    assert [i.slug for i in items] == ["dup", "dup-2", "dup-3", "other"]


# --- run loop -------------------------------------------------------------


def test_run_batch_continues_past_failures(tmp_path: Path) -> None:
    def good() -> Document:
        return _doc("ok")

    def bad() -> Document:
        raise TranscriptionError("model exploded", hint="...")

    items = [
        BatchItem(title="first", slug="first", ingest=good),
        BatchItem(title="second", slug="second", ingest=bad),
        BatchItem(title="third", slug="third", ingest=good),
    ]
    written: list[str] = []

    def write(document: Document, slug: str) -> Path:
        written.append(slug)
        return tmp_path / f"{slug}.md"

    results = run_batch(items, write)
    assert [r.ok for r in results] == [True, False, True]
    assert written == ["first", "third"]  # the failing item is skipped, not fatal
    assert results[1].error == "model exploded"  # friendly message preserved


def test_run_batch_reports_write_failure(tmp_path: Path) -> None:
    def write(document: Document, slug: str) -> Path:
        raise OSError("disk full")

    results = run_batch([BatchItem(title="x", slug="x", ingest=lambda: _doc("x"))], write)
    assert results[0].ok is False
    assert results[0].error is not None
    assert "disk full" in results[0].error


# --- slugify --------------------------------------------------------------


def test_slugify() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  Multiple   spaces ") == "multiple-spaces"
    assert slugify("#551: Stroll Down Startup Lane") == "551-stroll-down-startup-lane"
    assert slugify("", fallback="ep-7") == "ep-7"
    assert slugify("!!!", fallback="ep-7") == "ep-7"
    assert len(slugify("x" * 200)) <= 80


def test_slugify_keeps_combining_marks() -> None:
    # ``\\w`` dropped the vowel signs of abugida scripts, mangling the title.
    assert slugify("বাংলা সংবাদ") == "বাংলা-সংবাদ"
    assert slugify("اردو پوڈکاسٹ") == "اردو-پوڈکاسٹ"
    assert slugify("Hello, World!") == "hello-world"  # unchanged for Latin
