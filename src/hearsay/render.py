"""Render a Document to the hearsay markdown format (the product)."""

import textwrap

from hearsay.models import Document
from hearsay.timefmt import format_timestamp

_WRAP_WIDTH = 80


def render_markdown(doc: Document) -> str:
    """Render the document exactly per the SPEC.md output format (pure)."""
    lines = [
        "---",
        f'title: "{_escape(doc.meta.title)}"',
        f'source: "{_escape(doc.meta.source)}"',
        f'channel: "{_escape(doc.meta.channel)}"',
        f'duration: "{format_timestamp(doc.meta.duration_s)}"',
        f'ingested: "{doc.ingested_at}"',
        f'method: "{doc.method}"',
        f'language: "{doc.language}"',
        "---",
        "",
        f"# {doc.meta.title}",
    ]
    for section in doc.sections:
        lines.append("")
        lines.append(f"## {section.title}")
        for paragraph in section.paragraphs:
            stamp = f"**[{format_timestamp(paragraph.start_s)}]** "
            lines.append("")
            lines.append(
                textwrap.fill(
                    paragraph.text,
                    width=_WRAP_WIDTH,
                    initial_indent=stamp,
                    subsequent_indent="",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    """Escape a string for a YAML double-quoted scalar."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
