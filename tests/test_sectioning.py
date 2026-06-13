"""Tests for chapter-aware and time-based sectioning (offline)."""

import json
from pathlib import Path

from hearsay.models import Chapter, Paragraph
from hearsay.sectioning import sectionize

FIXTURES = Path(__file__).parent / "fixtures"


def para(start_s: float, end_s: float, text: str = "words") -> Paragraph:
    return Paragraph(text=text, start_s=start_s, end_s=end_s)


def test_empty_paragraphs_yield_no_sections() -> None:
    assert sectionize([], []) == []
    assert sectionize([], [Chapter(title="A", start_s=0, end_s=10)]) == []


def test_chapters_become_sections_in_order() -> None:
    chapters = [
        Chapter(title="Intro", start_s=0, end_s=60),
        Chapter(title="Middle", start_s=60, end_s=120),
        Chapter(title="End", start_s=120, end_s=180),
    ]
    paragraphs = [para(5, 50), para(60, 90), para(95, 119), para(150, 170)]
    sections = sectionize(paragraphs, chapters)
    assert [s.title for s in sections] == ["Intro", "Middle", "End"]
    assert [len(s.paragraphs) for s in sections] == [1, 2, 1]
    assert sections[1].start_s == 60


def test_paragraph_exactly_at_chapter_start_joins_that_chapter() -> None:
    chapters = [
        Chapter(title="A", start_s=0, end_s=100),
        Chapter(title="B", start_s=100, end_s=200),
    ]
    sections = sectionize([para(100, 110)], chapters)
    assert [s.title for s in sections] == ["B"]


def test_chapters_without_paragraphs_are_dropped() -> None:
    chapters = [
        Chapter(title="Silent", start_s=0, end_s=100),
        Chapter(title="Spoken", start_s=100, end_s=200),
    ]
    sections = sectionize([para(150, 160)], chapters)
    assert [s.title for s in sections] == ["Spoken"]


def test_paragraphs_before_first_chapter_get_span_section() -> None:
    chapters = [Chapter(title="Late start", start_s=100, end_s=200)]
    sections = sectionize([para(0, 30), para(120, 130)], chapters)
    assert sections[0].title == "[00:00:00 – 00:00:30]"
    assert sections[1].title == "Late start"


def test_real_chapter_fixture_covers_all_paragraphs() -> None:
    raw = json.loads((FIXTURES / "zjkBMFhNj_g.meta.json").read_text())
    chapters = [
        Chapter(title=c["title"], start_s=c["start_time"], end_s=c["end_time"])
        for c in raw["chapters"]
    ]
    paragraphs = [para(float(t), float(t) + 30) for t in range(0, 3580, 40)]
    sections = sectionize(paragraphs, chapters)
    assert sum(len(s.paragraphs) for s in sections) == len(paragraphs)
    titles = [s.title for s in sections]
    assert titles == [c.title for c in chapters if c.title in titles]  # source order kept


def test_time_based_sections_split_about_every_window() -> None:
    paragraphs = [para(float(t), float(t) + 25, text="x " * 50) for t in range(0, 1080, 30)]
    sections = sectionize(paragraphs, [], window_s=300.0)
    assert len(sections) == 4
    assert sum(len(s.paragraphs) for s in sections) == len(paragraphs)
    # Every section spans at most ~the window.
    for section in sections:
        starts = [p.start_s for p in section.paragraphs]
        assert max(starts) - min(starts) < 300.0
    assert sections[0].title == "[00:00:00 – 00:04:55]"
    assert sections[1].start_s == 300.0


def test_time_based_single_short_video_is_one_section() -> None:
    paragraphs = [para(0, 50), para(60, 110)]
    sections = sectionize(paragraphs, [])
    assert len(sections) == 1
    assert sections[0].title == "[00:00:00 – 00:01:50]"
