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
from pydantic import ValidationError

import hearsay.dataset.build as ds_build
from hearsay.dataset.build import build_dataset, build_dataset_from_file
from hearsay.dataset.models import DatasetConfig, FilterConfig
from hearsay.errors import AudioExportError, InvalidSourceError
from hearsay.models import SourceMetadata, Word
from hearsay.transcribe import TranscriptionError, transcribe_audio

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"  # ~4.71s, 16 kHz mono, "...quick brown fox...lazy dog..."
_NO_FILTER = FilterConfig(enabled=False)  # isolate D2 behaviour from D3 quality filtering


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
        title="sample",
        source=str(SAMPLE),
        channel="Local file",
        duration_s=4.71,
        video_id="sample",
    )


def _probe_stream(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
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
    config = DatasetConfig(
        out_dir=tmp_path, segment_min_s=0.5, segment_max_s=1.0, filters=_NO_FILTER
    )
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
    config = DatasetConfig(
        out_dir=tmp_path, segment_min_s=0.2, segment_max_s=2.0, filters=_NO_FILTER
    )
    report = build_dataset(SAMPLE, words, _meta(), config=config)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert all(row["duration"] > 0 for row in manifest)  # no ghost clips
    assert len(manifest) == report.clip_count
    for clip in report.clips:
        wav = tmp_path / clip.audio_path
        assert wav.stat().st_size > 100  # real audio, not a 78-byte header-only WAV
        assert clip.end_s <= 4.72  # clamped to the source length


def test_build_filters_drop_bad_clips_and_log(tmp_path: Path) -> None:
    # Two clips within the 4.71s fixture, split by a >2s pause: a wrong-script
    # (CJK, target "en") clip is dropped; a clean English clip is kept.
    words = [
        Word(text="你好", start_s=0.0, end_s=0.3, confidence=0.9),
        Word(text="世界", start_s=0.3, end_s=0.6, confidence=0.9),
        Word(text="今天", start_s=0.6, end_s=1.0, confidence=0.9),
        Word(text="Hello", start_s=3.5, end_s=3.8, confidence=0.9),
        Word(text="there", start_s=3.8, end_s=4.1, confidence=0.9),
        Word(text="friend", start_s=4.1, end_s=4.5, confidence=0.9),
    ]
    config = DatasetConfig(out_dir=tmp_path, segment_min_s=0.5, segment_max_s=10.0)
    report = build_dataset(SAMPLE, words, _meta(), config=config, language="en")

    assert report.clip_count == 1  # only the English clip survives
    assert "Hello there friend" in report.clips[0].text
    assert report.dropped_count == 1
    assert report.drops_by_reason == {"non_target_script": 1}

    dropped = (tmp_path / "dropped.jsonl").read_text().splitlines()
    assert len(dropped) == 1
    rec = json.loads(dropped[0])
    assert rec["filter"] == "non_target_script" and rec["value"] == "cjk"
    assert "dropped.jsonl" in report.files


def test_build_no_filter_keeps_all(tmp_path: Path) -> None:
    # Same input, filtering disabled -> both clips kept, no drops.
    words = [
        Word(text="你好", start_s=0.0, end_s=0.5, confidence=0.9),
        Word(text="世界", start_s=0.5, end_s=1.0, confidence=0.9),
        Word(text="Hello", start_s=3.5, end_s=4.0, confidence=0.9),
        Word(text="there", start_s=4.0, end_s=4.5, confidence=0.9),
    ]
    config = DatasetConfig(
        out_dir=tmp_path, segment_min_s=0.5, segment_max_s=10.0, filters=_NO_FILTER
    )
    report = build_dataset(SAMPLE, words, _meta(), config=config)
    assert report.clip_count == 2
    assert report.dropped_count == 0


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
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=0.5,
        segment_max_s=15.0,
        filters=_NO_FILTER,
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


def test_config_rejects_inverted_segment_bounds() -> None:
    # Each bound is individually valid, so without a cross-field check the build runs
    # to completion and emits a near-empty dataset — a bad flag must fail fast instead.
    DatasetConfig(out_dir=Path("x"), segment_min_s=1.0, segment_max_s=15.0)  # valid
    with pytest.raises(ValidationError, match="--segment-min"):
        DatasetConfig(out_dir=Path("x"), segment_min_s=20.0, segment_max_s=5.0)


def test_config_allows_equal_segment_bounds() -> None:
    # A fixed clip length is a legitimate request, not an inverted window.
    cfg = DatasetConfig(out_dir=Path("x"), segment_min_s=5.0, segment_max_s=5.0)
    assert cfg.segment_min_s == cfg.segment_max_s == 5.0


# --- word-alignment warning ------------------------------------------------


def _warning_config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(out_dir=tmp_path, sample_rate=16000, filters=FilterConfig(enabled=False))


@pytest.mark.parametrize("method", ["whisper-tiny", "whisper-base"])
def test_small_models_warn_about_word_alignment(tmp_path: Path, method: str) -> None:
    # tiny/base can omit an audible word from a clip's transcript, which is silent
    # audio/text misalignment — no downstream filter can see it, so it must be said.
    report = build_dataset(
        SAMPLE,
        _synthetic_words(),
        _meta(),
        config=_warning_config(tmp_path),
        transcription_method=method,
    )
    assert any("word alignment" in w for w in report.warnings)
    assert any(method in w for w in report.warnings)


@pytest.mark.parametrize("method", ["whisper-small", "whisper-large-v3", "parakeet-tdt-0.6b-v3"])
def test_capable_models_do_not_warn(tmp_path: Path, method: str) -> None:
    report = build_dataset(
        SAMPLE,
        _synthetic_words(),
        _meta(),
        config=_warning_config(tmp_path),
        transcription_method=method,
    )
    assert not any("word alignment" in w for w in report.warnings)


def test_alignment_warning_absent_when_method_unknown(tmp_path: Path) -> None:
    # Callers that don't report a method (older/injected paths) must not be warned at.
    report = build_dataset(SAMPLE, _synthetic_words(), _meta(), config=_warning_config(tmp_path))
    assert not any("word alignment" in w for w in report.warnings)


def test_segment_bounds_flow_into_the_duration_filter() -> None:
    # The segmenter cuts to the user's window, then the filter re-checks the result.
    # Left at their standalone defaults the filter rejected everything the segmenter
    # produced, so `--segment-max 30` reported "0 clips" under a green success tick.
    cfg = DatasetConfig(out_dir=Path("x"), segment_min_s=16.0, segment_max_s=30.0)
    assert cfg.filters.min_duration_s == 16.0
    assert cfg.filters.max_duration_s == 30.0


def test_explicit_filter_bounds_are_not_overridden() -> None:
    # A caller who sets filter bounds deliberately keeps them.
    cfg = DatasetConfig(
        out_dir=Path("x"),
        segment_min_s=16.0,
        segment_max_s=30.0,
        filters=FilterConfig(min_duration_s=2.0, max_duration_s=25.0),
    )
    assert cfg.filters.min_duration_s == 2.0
    assert cfg.filters.max_duration_s == 25.0


def test_widened_segment_window_actually_yields_clips(tmp_path: Path) -> None:
    # End-to-end guard for the same trap, with the quality filters ON.
    config = DatasetConfig(out_dir=tmp_path, segment_min_s=3.0, segment_max_s=30.0)
    report = build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)
    assert report.clip_count >= 1
    assert not any(d.filter == "duration" for d in report.drops)


def test_hf_and_ljspeech_together_warn(tmp_path: Path) -> None:
    # Verified against datasets 4.x: both indexes in one tree raises
    # "Found metadata files with different extensions: ['.csv', '.jsonl']",
    # so the HuggingFace index the user asked for is unloadable.
    config = DatasetConfig(out_dir=tmp_path, formats=["ljspeech", "hf"])
    report = build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)
    assert any("cannot share a dataset folder" in w for w in report.warnings)


def test_hf_alone_does_not_warn(tmp_path: Path) -> None:
    config = DatasetConfig(out_dir=tmp_path, formats=["hf", "jsonl"])
    report = build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)
    assert not any("cannot share a dataset folder" in w for w in report.warnings)


def test_one_bad_slice_does_not_discard_the_whole_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable stretch of a long recording must cost one clip, not all of them."""
    real_slice = ds_build.slice_clip
    calls = {"n": 0}

    def flaky_slice(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise AudioExportError("ffmpeg could not slice clip", hint="…")
        real_slice(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ds_build, "slice_clip", flaky_slice)
    config = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.0,
        filters=_NO_FILTER,
    )
    report = build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)

    assert calls["n"] >= 2, "fixture must produce enough clips to hit the failing one"
    assert report.clip_count >= 1
    assert any(d.filter == "slice_failed" for d in report.drops)
    # ids stay dense: the failed clip did not leave a hole in the numbering
    assert [c.id for c in report.clips] == [
        f"sample_{i:04d}" for i in range(1, report.clip_count + 1)
    ]


def test_a_thoroughly_broken_source_still_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tolerating every failure would ship a silently decimated dataset.
    def always_fails(*args: object, **kwargs: object) -> None:
        raise AudioExportError("ffmpeg could not slice clip", hint="…")

    monkeypatch.setattr(ds_build, "slice_clip", always_fails)
    config = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=0.3,
        segment_max_s=0.6,
        filters=_NO_FILTER,
    )
    with pytest.raises(AudioExportError):
        build_dataset(SAMPLE, _synthetic_words(), _meta(), config=config)


def test_stock_whisper_on_an_unusable_language_warns(tmp_path: Path) -> None:
    """Whisper's own FLEURS table puts Uzbek at ~90% WER and Pashto at ~93% — worse than
    transcribing nothing. Clips would ship paired with text that is largely wrong, and no
    downstream filter can see it, so the build has to say so."""
    report = build_dataset(
        SAMPLE,
        _synthetic_words(),
        _meta(),
        config=DatasetConfig(out_dir=tmp_path, filters=_NO_FILTER),
        language="uz",
        transcription_method="whisper-large-v3",
    )
    assert any("cannot really transcribe" in w for w in report.warnings)


def test_a_language_specific_model_is_not_warned_about(tmp_path: Path) -> None:
    # A custom model IS the fine-tune the warning asks for.
    report = build_dataset(
        SAMPLE,
        _synthetic_words(),
        _meta(),
        config=DatasetConfig(out_dir=tmp_path, filters=_NO_FILTER),
        language="uz",
        transcription_method="org/uzbek-ct2",
    )
    assert not any("cannot really transcribe" in w for w in report.warnings)


def test_well_served_languages_are_not_warned_about(tmp_path: Path) -> None:
    report = build_dataset(
        SAMPLE,
        _synthetic_words(),
        _meta(),
        config=DatasetConfig(out_dir=tmp_path, filters=_NO_FILTER),
        language="en",
        transcription_method="whisper-small",
    )
    assert not any("cannot really transcribe" in w for w in report.warnings)
