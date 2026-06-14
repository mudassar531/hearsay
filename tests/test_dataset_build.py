"""Integration tests for the dataset build (real ffmpeg slicing of the fixture clip).

Offline by construction: ffmpeg/ffprobe operate on the local committed WAV, opening
no network. A synthetic word list makes the build deterministic without a model; a
separate end-to-end test transcribes the clip with the tiny model and skips if it
is not cached.
"""

import json
import subprocess
from pathlib import Path

import pytest

from hearsay.dataset.build import build_dataset, build_dataset_from_file
from hearsay.dataset.models import DatasetConfig
from hearsay.errors import AudioExportError, InvalidSourceError
from hearsay.models import SourceMetadata, Word
from hearsay.transcribe import TranscriptionError, transcribe_audio

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"  # ~4.71s, 16 kHz mono, "...quick brown fox...lazy dog..."


def _synthetic_words() -> list[Word]:
    """Word spans within the 4.71s clip, sized to yield a few clips at min_s=1,max_s=2."""
    return [
        Word(text="The", start_s=0.0, end_s=0.4),
        Word(text="quick", start_s=0.4, end_s=0.9),
        Word(text="brown", start_s=0.9, end_s=1.4),
        Word(text="fox.", start_s=1.4, end_s=1.9),
        Word(text="Jumps", start_s=2.0, end_s=2.5),
        Word(text="over", start_s=2.5, end_s=3.0),
        Word(text="the", start_s=3.0, end_s=3.4),
        Word(text="lazy", start_s=3.4, end_s=3.9),
        Word(text="dog.", start_s=3.9, end_s=4.4),
    ]


def _meta() -> SourceMetadata:
    return SourceMetadata(
        title="sample", source=str(SAMPLE), channel="Local file",
        duration_s=4.71, video_id="sample",
    )


def _probe_stream(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    return int(stream["sample_rate"]), int(stream["channels"])


def test_build_dataset_end_to_end(tmp_path: Path) -> None:
    config = DatasetConfig(
        out_dir=tmp_path, sample_rate=22050, segment_min_s=1.0, segment_max_s=2.0
    )
    report = build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config, language="en")

    # clip count matches the segments, and every WAV physically exists
    assert report.clip_count >= 2
    wavs = sorted((tmp_path / "wavs").glob("*.wav"))
    assert len(wavs) == report.clip_count == len(report.clips)
    for clip in report.clips:
        assert (tmp_path / clip.audio_path).exists()

    # audio <-> text <-> duration line up: manifest duration ~= probed WAV duration
    # ~= the segment's span, and the WAV is mono at the requested sample rate.
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == report.clip_count
    for clip, row in zip(report.clips, manifest, strict=True):
        wav = tmp_path / clip.audio_path
        assert row["text"] == clip.text
        assert row["audio_filepath"] == clip.audio_path
        assert row["duration"] == pytest.approx(clip.duration_s, abs=0.05)
        assert clip.duration_s == pytest.approx(clip.end_s - clip.start_s, abs=0.1)
        sr, channels = _probe_stream(wav)
        assert sr == 22050 and channels == 1

    # LJSpeech csv and the card are written; ids join the wavs
    csv_lines = (tmp_path / "metadata.csv").read_text().splitlines()
    assert len(csv_lines) == report.clip_count
    assert csv_lines[0].split("|")[0] == report.clips[0].id
    assert (tmp_path / "dataset_card.md").exists()
    assert "dataset_card.md" in report.files
    assert report.total_duration_s == pytest.approx(sum(c.duration_s for c in report.clips))


def test_build_only_requested_formats(tmp_path: Path) -> None:
    config = DatasetConfig(
        out_dir=tmp_path, formats=["jsonl"], segment_min_s=1.0, segment_max_s=2.0
    )
    build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)
    assert (tmp_path / "manifest.jsonl").exists()
    assert not (tmp_path / "metadata.csv").exists()
    assert (tmp_path / "dataset_card.md").exists()  # card is always written


def test_build_empty_words_yields_empty_dataset(tmp_path: Path) -> None:
    config = DatasetConfig(out_dir=tmp_path)
    report = build_dataset(SAMPLE, [], _meta(), config=config)
    assert report.clip_count == 0
    assert report.total_duration_s == 0.0
    assert (tmp_path / "manifest.jsonl").read_text() == ""
    assert (tmp_path / "dataset_card.md").exists()  # still documents a (0-clip) build
    assert list((tmp_path / "wavs").glob("*.wav")) == []


def test_oversized_clip_surfaced_in_manifest(tmp_path: Path) -> None:
    # A single word longer than max_s is unbreakable -> emitted oversized, and the
    # manifest must reflect its real (clamped-to-source) duration.
    words = [Word(text="loooong", start_s=0.0, end_s=3.0)]
    config = DatasetConfig(out_dir=tmp_path, segment_min_s=0.5, segment_max_s=1.0)
    report = build_dataset(SAMPLE, words, _meta(), config=config)
    assert report.clip_count == 1
    assert report.oversized_count == 1
    assert report.clips[0].oversized is True
    assert (tmp_path / report.clips[0].audio_path).exists()
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["duration"] > 1.0  # real ~3s clip, not a ghost


def test_clips_past_source_end_are_clamped_or_dropped(tmp_path: Path) -> None:
    # Source is 4.71s. A clip that overruns the end is truncated to real audio; a
    # clip entirely past EOF is dropped — never a duration:0.0 ghost row.
    words = [
        Word(text="near", start_s=4.4, end_s=4.9),  # overruns 4.71 -> truncated
        Word(text="past", start_s=6.0, end_s=7.0),  # fully past EOF -> dropped
        Word(text="gone", start_s=8.0, end_s=9.0),  # fully past EOF -> dropped
    ]
    config = DatasetConfig(out_dir=tmp_path, segment_min_s=0.2, segment_max_s=2.0)
    report = build_dataset(SAMPLE, words, _meta(), config=config)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert all(row["duration"] > 0 for row in manifest)  # no ghost clips
    assert len(manifest) == report.clip_count
    for clip in report.clips:
        wav = tmp_path / clip.audio_path
        assert wav.stat().st_size > 100  # real audio, not a 78-byte header-only WAV
        assert clip.end_s <= 4.72  # clamped to the source length


def test_unknown_format_rejected(tmp_path: Path) -> None:
    config = DatasetConfig(out_dir=tmp_path, formats=["bogus"])
    with pytest.raises(AudioExportError) as excinfo:
        build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)
    assert "bogus" in excinfo.value.message


def test_build_from_file_missing_path(tmp_path: Path) -> None:
    config = DatasetConfig(out_dir=tmp_path)
    with pytest.raises(InvalidSourceError):
        build_dataset_from_file(tmp_path / "nope.wav", config=config)


def test_build_from_file_with_tiny_model(tmp_path: Path) -> None:
    # End-to-end through the real engine; offline via local_files_only, skips if uncached.
    def offline_transcriber(path: Path, **kwargs: object):
        return transcribe_audio(path, local_files_only=True, **kwargs)  # type: ignore[arg-type]

    config = DatasetConfig(
        out_dir=tmp_path, sample_rate=16000, segment_min_s=0.5, segment_max_s=15.0
    )
    try:
        report = build_dataset_from_file(
            SAMPLE, config=config, model_size="tiny", transcriber=offline_transcriber
        )
    except TranscriptionError:
        pytest.skip("tiny whisper model not cached")
    assert report.clip_count >= 1
    all_text = " ".join(c.text for c in report.clips).lower()
    assert "fox" in all_text and "dog" in all_text
    for clip in report.clips:
        assert (tmp_path / clip.audio_path).exists()
