"""Tests for optional diarization — pure assignment, modes, and clean degrade (offline).

A fake diarizer (a list of speaker turns) exercises the per-speaker / dominant /
cross-speaker logic. The "extra not installed" path is exercised through the
``no_pyannote`` fixture, which makes the import fail on demand rather than relying on
pyannote being absent from the environment — installing the documented ``[diarize]``
extra must not turn these into failures (and, once installed, it would otherwise reach
for a gated model over the network).
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from hearsay.dataset.audio import decode_to_wav
from hearsay.dataset.build import (
    DatasetSource,
    _produce_clips,
    build_combined_dataset,
    build_dataset,
)
from hearsay.dataset.diarize import (
    PyannoteDiarizer,
    SpeakerTurn,
    assign_speaker,
    dominant_speaker,
    resolve_token,
)
from hearsay.dataset.models import DatasetConfig, DiarizeConfig, FilterConfig
from hearsay.errors import DiarizationError
from hearsay.models import SourceMetadata, Word

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"  # ~4.71s


@pytest.fixture
def no_pyannote(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make ``pyannote.audio`` unimportable for the duration of a test.

    ``None`` in ``sys.modules`` is the documented way to force a failed import: the
    import machinery raises ImportError, and ``find_spec`` raises ValueError — the two
    signals hearsay treats as "the extra isn't installed".
    """
    for name in ("pyannote", "pyannote.audio"):
        monkeypatch.setitem(sys.modules, name, None)
    yield


# Two speakers: SPEAKER_00 owns [0, 2.5], SPEAKER_01 owns [2.5, 4.5].
_TURNS = [SpeakerTurn("SPEAKER_00", 0.0, 2.5), SpeakerTurn("SPEAKER_01", 2.5, 4.5)]


# --- pure assignment -------------------------------------------------------


def test_assign_speaker_max_overlap() -> None:
    speaker, purity = assign_speaker(0.0, 2.0, _TURNS)
    assert speaker == "SPEAKER_00" and purity == pytest.approx(1.0)
    speaker, purity = assign_speaker(2.6, 4.1, _TURNS)
    assert speaker == "SPEAKER_01" and purity == pytest.approx(1.0)


def test_assign_speaker_cross_speaker_purity() -> None:
    # [1.5, 3.5]: 1.0s of SPEAKER_00 + 1.0s of SPEAKER_01 -> purity 0.5.
    speaker, purity = assign_speaker(1.5, 3.5, _TURNS)
    assert speaker in {"SPEAKER_00", "SPEAKER_01"}
    assert purity == pytest.approx(0.5)


def test_assign_speaker_no_overlap() -> None:
    assert assign_speaker(10.0, 11.0, _TURNS) == (None, 0.0)


def test_dominant_speaker() -> None:
    assert dominant_speaker(_TURNS) == "SPEAKER_00"  # 2.5s > 2.0s
    assert dominant_speaker([]) is None


def test_resolve_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert resolve_token(DiarizeConfig()) is None
    assert resolve_token(DiarizeConfig(hf_token="explicit")) == "explicit"  # explicit wins
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert resolve_token(DiarizeConfig()) == "from-env"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy")
    assert resolve_token(DiarizeConfig()) == "legacy"


def test_pyannote_not_installed_raises_friendly_error(no_pyannote: None) -> None:
    # With the extra unavailable, the first call must raise a clear, actionable
    # DiarizationError (not a bare ImportError).
    diarizer = PyannoteDiarizer(DiarizeConfig())
    with pytest.raises(DiarizationError) as excinfo:
        diarizer(SAMPLE)
    assert "pyannote" in excinfo.value.message.lower()
    assert "hearsay[diarize]" in excinfo.value.hint


def test_diarize_config_rejects_unknown_mode() -> None:
    DiarizeConfig(mode="dominant")  # valid
    with pytest.raises(ValueError, match="mode must be one of"):
        DiarizeConfig(mode="dominent")  # typo -> fail fast, not silent tag


class _Segment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _Annotation:
    """Stand-in for a pyannote Annotation: itertracks yields (segment, track, label)."""

    def __init__(self, turns: list[tuple[str, float, float]]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        for i, (label, start, end) in enumerate(self._turns):
            yield _Segment(start, end), f"track{i}", label


class _DiarizeOutput:  # mimics pyannote 4.x: the Annotation lives at .speaker_diarization
    def __init__(self, annotation: _Annotation) -> None:
        self.speaker_diarization = annotation


def _diarizer_with_pipeline(result_for) -> PyannoteDiarizer:
    d = PyannoteDiarizer(DiarizeConfig())
    d._pipeline = lambda _path, **_kw: result_for  # bypass _load (no pyannote needed)
    return d


def test_pyannote_parses_4x_diarize_output() -> None:
    out = _DiarizeOutput(_Annotation([("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 1.0, 2.0)]))
    turns = _diarizer_with_pipeline(out)(SAMPLE)
    assert turns == [SpeakerTurn("SPEAKER_00", 0.0, 1.0), SpeakerTurn("SPEAKER_01", 1.0, 2.0)]


def test_pyannote_parses_3x_bare_annotation() -> None:
    ann = _Annotation([("SPEAKER_00", 0.0, 1.5)])  # 3.x returned a bare Annotation
    turns = _diarizer_with_pipeline(ann)(SAMPLE)
    assert turns == [SpeakerTurn("SPEAKER_00", 0.0, 1.5)]


def test_pyannote_unexpected_output_raises_friendly_error() -> None:
    # A drifted/unknown return shape degrades to an actionable DiarizationError.
    with pytest.raises(DiarizationError) as excinfo:
        _diarizer_with_pipeline(object())(SAMPLE)
    assert "Unexpected diarization output" in excinfo.value.message


# --- build integration with a fake diarizer (offline) ----------------------


def _fake_diarizer(turns: list[SpeakerTurn]):
    return lambda _path: list(turns)


def _meta() -> SourceMetadata:
    return SourceMetadata(
        title="sample", source=str(SAMPLE), channel="c", duration_s=4.71, video_id="sample"
    )


def _config(tmp_path: Path, mode: str) -> DatasetConfig:
    return DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.5,
        filters=FilterConfig(enabled=False),
        diarize=DiarizeConfig(enabled=True, mode=mode),
    )


def _two_speaker_words() -> list[Word]:
    return [
        Word(text="Hello", start_s=0.0, end_s=0.5, confidence=0.9),
        Word(text="there", start_s=0.5, end_s=1.0, confidence=0.9),
        Word(text="friend", start_s=1.0, end_s=1.5, confidence=0.9),
        Word(text="today", start_s=1.5, end_s=2.0, confidence=0.9),
        Word(text="Goodbye", start_s=2.6, end_s=3.1, confidence=0.9),
        Word(text="now", start_s=3.1, end_s=3.6, confidence=0.9),
        Word(text="everyone", start_s=3.6, end_s=4.1, confidence=0.9),
    ]


def _manifest(tmp_path: Path) -> list[dict]:
    path = tmp_path / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_tag_mode_labels_clips(tmp_path: Path) -> None:
    report = build_dataset(
        SAMPLE,
        _two_speaker_words(),
        _meta(),
        config=_config(tmp_path, "tag"),
        diarizer=_fake_diarizer(_TURNS),
    )
    speakers = {c.speaker for c in report.clips}
    assert speakers == {"sample:SPEAKER_00", "sample:SPEAKER_01"}  # namespaced per source
    assert all(row.get("speaker") for row in _manifest(tmp_path))  # speaker in the manifest


def test_dominant_mode_keeps_only_top_speaker(tmp_path: Path) -> None:
    report = build_dataset(
        SAMPLE,
        _two_speaker_words(),
        _meta(),
        config=_config(tmp_path, "dominant"),
        diarizer=_fake_diarizer(_TURNS),
    )
    assert {c.speaker for c in report.clips} == {"sample:SPEAKER_00"}  # the most-spoken speaker
    assert any(d.filter == "non_dominant_speaker" for d in report.drops)
    # the dropped speaker's WAV is removed (tree matches manifest)
    wavs = {p.name for p in (tmp_path / "wavs").glob("*.wav")}
    assert wavs == {row["audio_filepath"].split("/")[-1] for row in _manifest(tmp_path)}


def test_per_speaker_mode_writes_per_speaker_indices(tmp_path: Path) -> None:
    report = build_dataset(
        SAMPLE,
        _two_speaker_words(),
        _meta(),
        config=_config(tmp_path, "per_speaker"),
        diarizer=_fake_diarizer(_TURNS),
    )
    assert (tmp_path / "manifest.sample-SPEAKER_00.jsonl").exists()
    assert (tmp_path / "manifest.sample-SPEAKER_01.jsonl").exists()
    assert "manifest.sample-SPEAKER_00.jsonl" in report.files


def test_cross_speaker_clip_dropped(tmp_path: Path) -> None:
    # One clip spanning the speaker boundary ~50/50 -> below min_purity -> dropped.
    words = [
        Word(text="straddle", start_s=1.5, end_s=2.0, confidence=0.9),
        Word(text="the", start_s=2.0, end_s=2.5, confidence=0.9),
        Word(text="boundary", start_s=2.5, end_s=3.5, confidence=0.9),
    ]
    report = build_dataset(
        SAMPLE,
        words,
        _meta(),
        config=_config(tmp_path, "tag"),
        diarizer=_fake_diarizer(_TURNS),
    )
    assert report.clip_count == 0
    assert any(d.filter == "cross_speaker" for d in report.drops)
    assert list((tmp_path / "wavs").glob("*.wav")) == []  # cross-speaker WAV removed


def test_degrades_to_mixed_speaker_when_not_installed(tmp_path: Path, no_pyannote: None) -> None:
    # diarize requested, no diarizer injected, pyannote not installed -> mixed-speaker
    # dataset with a clear warning (never a crash).
    report = build_dataset(
        SAMPLE, _two_speaker_words(), _meta(), config=_config(tmp_path, "dominant")
    )
    assert report.clip_count >= 1
    assert all(c.speaker is None for c in report.clips)  # mixed, unlabeled
    assert any("MIXED speakers" in w for w in report.warnings)


def test_combined_per_speaker_namespaced_across_sources(tmp_path: Path) -> None:
    # Two sources each have a "SPEAKER_00"; namespacing by source id must keep them
    # separate in a combined per-speaker export (they are different people).
    cfg = DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.5,
        filters=FilterConfig(enabled=False),
        diarize=DiarizeConfig(enabled=True, mode="per_speaker"),
    )
    one_speaker = _fake_diarizer([SpeakerTurn("SPEAKER_00", 0.0, 5.0)])

    def make_source(sid: str) -> DatasetSource:
        def run(out_dir, config, source_id):
            clips, drops = _produce_clips(
                SAMPLE,
                _two_speaker_words()[:4],
                out_dir=out_dir,
                source_id=source_id,
                config=config,
                diarizer=one_speaker,
            )
            return clips, drops, _meta()

        return DatasetSource(source_id=sid, label=sid, run=run)

    build_combined_dataset([make_source("vidA"), make_source("vidB")], cfg, title="Mix", source="s")
    speakers = {row["speaker"] for row in _manifest(tmp_path)}
    assert speakers == {"vidA:SPEAKER_00", "vidB:SPEAKER_00"}  # not merged
    assert (tmp_path / "manifest.vidA-SPEAKER_00.jsonl").exists()
    assert (tmp_path / "manifest.vidB-SPEAKER_00.jsonl").exists()


# --- the backend always diarizes decoded PCM, never the original container ----


class _FakePipeline:
    """Stand-in pyannote pipeline that records the path it was handed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, path: str, **kwargs: object) -> _Annotation:
        self.seen.append(path)
        return _Annotation([("SPEAKER_00", 0.0, 1.0)])


def test_diarizer_decodes_compressed_source_to_wav(tmp_path: Path) -> None:
    """A compressed source must be decoded before it reaches pyannote.

    pyannote 4.x reads fixed windows sized from the reported duration; an MP3/M4A
    decodes to a frame-quantized sample count that can fall short of the last window,
    which it raises on ("resulted in N samples instead of the expected M"). Since every
    podcast enclosure is an MP3, handing it the raw path broke the headline
    podcast -> single-voice-TTS build outright.
    """
    mp3 = tmp_path / "episode.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(SAMPLE), str(mp3)],
        check=True,
    )
    diarizer = PyannoteDiarizer(DiarizeConfig())
    fake = _FakePipeline()
    diarizer._pipeline = fake

    turns = diarizer(mp3)

    assert turns == [SpeakerTurn("SPEAKER_00", 0.0, 1.0)]
    assert len(fake.seen) == 1
    assert fake.seen[0].endswith(".wav"), "pyannote must receive decoded PCM, not the .mp3"
    assert not Path(fake.seen[0]).exists(), "the decoded scratch file is cleaned up"


def test_decode_to_wav_produces_mono_16k_pcm(tmp_path: Path) -> None:
    out = decode_to_wav(SAMPLE, tmp_path / "out.wav")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "pcm_s16le,16000,1"
