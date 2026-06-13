"""Render a Document to the hearsay markdown format (the product)."""

import re
import textwrap

from hearsay.models import Document
from hearsay.timefmt import format_timestamp

_WRAP_WIDTH = 80
# Any run of whitespace (including newlines) collapses to a single space.
_WHITESPACE = re.compile(r"\s+")
# C0/C1 control characters that have no place in titles or YAML scalars.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def render_markdown(doc: Document) -> str:
    """Render the document exactly per the SPEC.md output format (pure)."""
    lines = [
        "---",
        f'title: "{_escape(doc.meta.title)}"',
        f'source: "{_escape(doc.meta.source)}"',
        f'channel: "{_escape(doc.meta.channel)}"',
        f'duration: "{format_timestamp(doc.meta.duration_s)}"',
        f'ingested: "{_escape(doc.ingested_at)}"',
        f'method: "{_escape(doc.method)}"',
        f'language: "{_escape(doc.language)}"',
        "---",
        "",
        f"# {_oneline(doc.meta.title)}",
    ]
    for section in doc.sections:
        lines.append("")
        lines.append(f"## {_oneline(section.title)}")
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


def _oneline(value: str) -> str:
    """Collapse whitespace and drop control chars so a heading stays one line.

    Markdown headings cannot span lines; a title or chapter title carrying a
    newline would otherwise inject extra structure into the document body.
    """
    return _CONTROL_CHARS.sub("", _WHITESPACE.sub(" ", value)).strip()


def _escape(value: str) -> str:
    """Escape a string for a YAML double-quoted scalar.

    Handles backslash, double-quote, and the whitespace/control characters a
    double-quoted scalar may not contain literally (newlines, tabs, and other
    C0/C1 control chars) so the frontmatter always parses and never lets a
    crafted title break out of its quotes.
    """
    out = []
    for char in value.replace("\\", "\\\\").replace('"', '\\"'):
        if char in "\n\r\t" or _CONTROL_CHARS.match(char):
            out.append(f"\\x{ord(char):02x}" if ord(char) <= 0xFF else f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)
