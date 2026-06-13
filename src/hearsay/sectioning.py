"""Group paragraphs into document sections.

Chapters from the source become `##` headings; without chapters, paragraphs
fall into time-based sections of roughly five minutes, titled with their
actual time span (we never invent topic titles — that would be summarization).
"""

from hearsay.models import Chapter, Paragraph, Section
from hearsay.timefmt import format_timestamp

DEFAULT_WINDOW_S = 300.0


def sectionize(
    paragraphs: list[Paragraph],
    chapters: list[Chapter],
    *,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[Section]:
    """Assign paragraphs to chapter sections, or to ~window_s time slices (pure).

    Sections that would contain no paragraphs are dropped. Paragraphs that
    start before the first chapter get a time-span section of their own.
    """
    if not paragraphs:
        return []
    if chapters:
        return _by_chapters(paragraphs, chapters)
    return _by_time(paragraphs, window_s)


def _span_title(paragraphs: list[Paragraph]) -> str:
    end_s = max(p.end_s for p in paragraphs)  # last by start order may not end last
    return f"[{format_timestamp(paragraphs[0].start_s)} – {format_timestamp(end_s)}]"


def _by_chapters(paragraphs: list[Paragraph], chapters: list[Chapter]) -> list[Section]:
    ordered = sorted(chapters, key=lambda c: c.start_s)
    buckets: list[list[Paragraph]] = [[] for _ in ordered]
    before_first: list[Paragraph] = []
    for paragraph in paragraphs:
        index = _chapter_index(ordered, paragraph.start_s)
        if index is None:
            before_first.append(paragraph)
        else:
            buckets[index].append(paragraph)
    sections: list[Section] = []
    if before_first:
        sections.append(
            Section(
                title=_span_title(before_first),
                start_s=before_first[0].start_s,
                paragraphs=before_first,
            )
        )
    sections.extend(
        Section(title=chapter.title, start_s=chapter.start_s, paragraphs=bucket)
        for chapter, bucket in zip(ordered, buckets, strict=True)
        if bucket
    )
    return sections


def _chapter_index(ordered: list[Chapter], start_s: float) -> int | None:
    """Index of the last chapter starting at or before start_s, or None."""
    index = None
    for i, chapter in enumerate(ordered):
        if chapter.start_s <= start_s:
            index = i
        else:
            break
    return index


def _by_time(paragraphs: list[Paragraph], window_s: float) -> list[Section]:
    sections: list[Section] = []
    current: list[Paragraph] = []
    window_end = paragraphs[0].start_s + window_s
    for paragraph in paragraphs:
        if current and paragraph.start_s >= window_end:
            sections.append(_time_section(current))
            current = []
            window_end = paragraph.start_s + window_s
        current.append(paragraph)
    sections.append(_time_section(current))
    return sections


def _time_section(paragraphs: list[Paragraph]) -> Section:
    return Section(
        title=_span_title(paragraphs),
        start_s=paragraphs[0].start_s,
        paragraphs=paragraphs,
    )
