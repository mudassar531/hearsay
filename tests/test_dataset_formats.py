"""Tests for the dataset index/card writers (pure; no ffmpeg)."""

import json
from pathlib import Path

from hearsay.dataset.formats import (
    _size_category,
    write_dataset_card,
    write_hf_audiofolder,
    write_indices,
    write_jsonl_manifest,
    write_ljspeech,
)
from hearsay.dataset.models import BuildReport, DatasetClip
from hearsay.models import SourceMetadata


def _clips() -> list[DatasetClip]:
    return [
        DatasetClip(
            id="vid_0001",
            audio_path="wavs/vid_0001.wav",
            text="Hello there, world.",
            start_s=0.0,
            end_s=2.0,
            duration_s=2.0,
        ),
        DatasetClip(
            id="vid_0002",
            audio_path="wavs/vid_0002.wav",
            text="A pipe | and a newline\nin here",
            start_s=2.5,
            end_s=5.0,
            duration_s=2.5,
        ),
    ]


def test_ljspeech_metadata_csv(tmp_path: Path) -> None:
    name = write_ljspeech(tmp_path, _clips())
    assert name == "metadata.csv"
    lines = (tmp_path / name).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # no header row
    cols = lines[0].split("|")
    assert cols == ["vid_0001", "Hello there, world.", "Hello there, world."]  # id|text|text
    # pipes and newlines in text are sanitized so the row stays single-line, 3-col
    assert len(lines[1].split("|")) == 3
    assert "\n" not in lines[1]
    assert "and a newline in here" in lines[1]


def test_jsonl_manifest(tmp_path: Path) -> None:
    name = write_jsonl_manifest(tmp_path, _clips())
    assert name == "manifest.jsonl"
    rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0] == {
        "audio_filepath": "wavs/vid_0001.wav",
        "duration": 2.0,
        "text": "Hello there, world.",
        "offset": 0.0,
    }
    assert isinstance(rows[0]["duration"], float)  # seconds, float
    assert rows[0]["audio_filepath"].startswith("wavs/")  # relative to the manifest


def test_hf_audiofolder_metadata(tmp_path: Path) -> None:
    name = write_hf_audiofolder(tmp_path, _clips())
    assert name == "metadata.jsonl"  # never collides with LJSpeech metadata.csv
    rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert rows[0] == {"file_name": "wavs/vid_0001.wav", "transcription": "Hello there, world."}


def test_write_indices_respects_selection(tmp_path: Path) -> None:
    files = write_indices(tmp_path, _clips(), ["jsonl"])
    assert files == ["manifest.jsonl"]
    assert (tmp_path / "manifest.jsonl").exists()
    assert not (tmp_path / "metadata.csv").exists()


def test_empty_clips_write_empty_indices(tmp_path: Path) -> None:
    write_ljspeech(tmp_path, [])
    write_jsonl_manifest(tmp_path, [])
    assert (tmp_path / "metadata.csv").read_text() == ""
    assert (tmp_path / "manifest.jsonl").read_text() == ""


def test_size_category() -> None:
    assert _size_category(0) == "n<1K"
    assert _size_category(999) == "n<1K"
    assert _size_category(1000) == "1K<n<10K"
    assert _size_category(50_000) == "10K<n<100K"


def test_dataset_card(tmp_path: Path) -> None:
    clips = _clips()
    meta = SourceMetadata(
        title="My Talk",
        source="https://youtu.be/abc",
        channel="Some Channel",
        duration_s=300.0,
        video_id="abc",
    )
    report = BuildReport(
        out_dir=str(tmp_path),
        source=meta.source,
        clip_count=len(clips),
        total_duration_s=4.5,
        oversized_count=0,
        sample_rate=22050,
        language="en",
        formats=["ljspeech", "jsonl"],
        clips=clips,
    )
    name = write_dataset_card(
        tmp_path,
        report,
        meta,
        version="0.2.0",
        generated_at="2026-06-14T10:00:00Z",
        source_platform="youtube",
    )
    card = (tmp_path / name).read_text(encoding="utf-8")
    assert card.startswith("---\n")  # YAML front matter
    assert "license: unknown" in card  # lowercase id, honest default
    assert "- text-to-speech" in card and "- automatic-speech-recognition" in card
    assert '- "en"' in card  # language as a quoted YAML scalar
    assert "https://youtu.be/abc" in card  # provenance
    assert "Some Channel" in card
    assert "2026-06-14" in card  # retrieval date
    assert "informed consent" in card  # the rights/consent note
    assert "not legal advice" in card
    assert "2 clips" in card or "**Clips:** 2" in card


def test_dataset_card_non_ascii_title_and_language(tmp_path: Path) -> None:
    # An emoji (astral-plane) title must be written as the real character, not a
    # \u surrogate pair, so the YAML front matter has no lone surrogates.
    meta = SourceMetadata(
        title="My 🎙 Podcast",
        source="local.wav",
        channel="Me",
        duration_s=10.0,
        video_id="x",
    )
    report = BuildReport(
        out_dir=str(tmp_path),
        source="local.wav",
        clip_count=1,
        total_duration_s=2.0,
        oversized_count=0,
        sample_rate=22050,
        language="en",
        formats=["jsonl"],
    )
    write_dataset_card(tmp_path, report, meta, version="0.2.0", generated_at="2026-06-14T00:00:00Z")
    card = (tmp_path / "dataset_card.md").read_text(encoding="utf-8")
    assert "🎙" in card  # the actual emoji, intact
    front = card.split("---", 2)[1]
    front.encode("utf-8")  # would raise UnicodeEncodeError on a lone surrogate
