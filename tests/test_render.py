"""Tests for the markdown renderer — the output format is the product."""

from hearsay.models import Chapter, Document, Paragraph, Section, SourceMetadata
from hearsay.render import render_markdown
from hearsay.timefmt import format_timestamp


def make_doc(**overrides: object) -> Document:
    meta = SourceMetadata(
        title="A Test Video",
        source="https://www.youtube.com/watch?v=abc123def45",
        channel="Test Channel",
        duration_s=3753.0,
        video_id="abc123def45",
        chapters=[Chapter(title="One", start_s=0, end_s=10)],
    )
    base: dict = {
        "meta": meta,
        "method": "captions",
        "language": "en",
        "ingested_at": "2026-06-13T10:00:00Z",
        "sections": [
            Section(
                title="One",
                start_s=0,
                paragraphs=[
                    Paragraph(text="Hello there.", start_s=12.4, end_s=14.0),
                    Paragraph(text="Second paragraph.", start_s=105.0, end_s=110.0),
                ],
            )
        ],
    }
    base.update(overrides)
    return Document(**base)


def test_exact_output_format() -> None:
    expected = """---
title: "A Test Video"
source: "https://www.youtube.com/watch?v=abc123def45"
channel: "Test Channel"
duration: "01:02:33"
ingested: "2026-06-13T10:00:00Z"
method: "captions"
language: "en"
---

# A Test Video

## One

**[00:00:12]** Hello there.

**[00:01:45]** Second paragraph.
"""
    assert render_markdown(make_doc()) == expected


def test_yaml_escaping_of_quotes_and_backslashes() -> None:
    doc = make_doc()
    doc.meta.title = 'He said "hi" \\ bye'
    output = render_markdown(doc)
    assert 'title: "He said \\"hi\\" \\\\ bye"' in output
    # The body heading keeps the raw title.
    assert '# He said "hi" \\ bye' in output


def test_long_paragraphs_wrap_at_80_columns() -> None:
    words = " ".join(f"word{i}" for i in range(60))
    doc = make_doc(
        sections=[
            Section(
                title="One",
                start_s=0,
                paragraphs=[Paragraph(text=words, start_s=0, end_s=60)],
            )
        ]
    )
    output = render_markdown(doc)
    body = output.split("## One\n\n")[1]
    lines = body.strip().splitlines()
    assert len(lines) > 1  # actually wrapped
    assert all(len(line) <= 80 for line in lines)
    assert lines[0].startswith("**[00:00:00]** word0")
    assert not lines[1].startswith(" ")  # continuation lines are flush left


def test_multiple_sections_render_in_order() -> None:
    doc = make_doc(
        sections=[
            Section(title="One", start_s=0, paragraphs=[Paragraph(text="a", start_s=0, end_s=1)]),
            Section(
                title="[00:05:00 – 00:09:58]",
                start_s=300,
                paragraphs=[Paragraph(text="b", start_s=300, end_s=301)],
            ),
        ]
    )
    output = render_markdown(doc)
    assert output.index("## One") < output.index("## [00:05:00 – 00:09:58]")
    assert output.endswith("\n")


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3723.9) == "01:02:03"
    assert format_timestamp(86399) == "23:59:59"
    assert format_timestamp(-5) == "00:00:00"
