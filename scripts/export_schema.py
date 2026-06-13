"""Export the Transcript JSON Schema to docs/schema.json.

Run after changing the Transcript model:  uv run python scripts/export_schema.py
A test (tests/test_transcript.py) asserts the committed file stays in sync.
"""

from pathlib import Path

from hearsay.models import transcript_schema_json

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "schema.json"


def main() -> None:
    """Write the schema to docs/schema.json."""
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(transcript_schema_json(), encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
