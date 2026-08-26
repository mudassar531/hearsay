"""Tests for combined (playlist/feed) dataset builds — merge, failures, totals, resume.

Offline: source ``run`` closures slice the committed fixture WAV via the real
ffmpeg path; network/transcription is never touched (injected fetchers).
"""

import json
from pathlib import Path

import pytest

from hearsay.dataset.build import (
    DatasetSource,
    _produce_clips,
    build_combined_dataset,
    build_dataset_from_playlist,
)
from hearsay.dataset.models import DatasetConfig, DiarizeConfig, FilterConfig
from hearsay.models import SourceMetadata, Word
from hearsay.transcribe import TranscriptionResult
from hearsay.youtube import PlaylistEntry

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.wav"  # ~4.71s, 16 kHz mono


def _words() -> list[Word]:
    return [
        Word(text="Hello", start_s=0.0, end_s=0.4, confidence=0.9),
        Word(text="there", start_s=0.4, end_s=0.9, confidence=0.9),
        Word(text="friend", start_s=0.9, end_s=1.4, confidence=0.9),
        Word(text="today", start_s=1.4, end_s=1.9, confidence=0.9),
    ]


def _config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        out_dir=tmp_path,
        sample_rate=16000,
        segment_min_s=1.0,
        segment_max_s=2.0,
        filters=FilterConfig(enabled=False),  # isolate orchestration from quality filtering
    )


def _source(
    source_id: str,
    label: str,
    *,
    fail: bool = False,
    slice_then_fail: bool = False,
    empty: bool = False,
    calls: list | None = None,
) -> DatasetSource:
    def run(out_dir, config, sid):
        if calls is not None:
            calls.append(sid)
        meta = SourceMetadata(
            title=label, source=f"src://{sid}", channel="c", duration_s=4.71, video_id=sid
        )
        if slice_then_fail:  # write WAVs into the shared tree, then die
            _produce_clips(SAMPLE, _words(), out_dir=out_dir, source_id=sid, config=config)
            raise RuntimeError("boom after slicing")
        if fail:
            raise RuntimeError("boom")
        if empty:
            return [], [], meta
        clips, drops = _produce_clips(
            SAMPLE, _words(), out_dir=out_dir, source_id=sid, config=config
        )
        return clips, drops, meta

    return DatasetSource(source_id=source_id, label=label, run=run)


def _manifest(tmp_path: Path) -> list[dict]:
    path = tmp_path / "manifest.jsonl"
    text = path.read_text() if path.exists() else ""
    return [json.loads(line) for line in text.splitlines()]


def _wav_names(tmp_path: Path) -> set[str]:
    return {p.name for p in (tmp_path / "wavs").glob("*.wav")}


def _manifest_wav_names(tmp_path: Path) -> set[str]:
    return {row["audio_filepath"].split("/")[-1] for row in _manifest(tmp_path)}


def test_combined_merges_sources_into_one_dataset(tmp_path: Path) -> None:
    sources = [_source("alpha", "A"), _source("beta", "B")]
    report = build_combined_dataset(sources, _config(tmp_path), title="Mix", source="src://mix")

    assert report.succeeded == 2 and report.failed == 0
    assert report.clip_count >= 2
    # one shared wavs/ tree and one merged manifest covering all clips
    manifest = _manifest(tmp_path)
    assert len(manifest) == report.clip_count
    prefixes = {row["audio_filepath"].split("/")[1].rsplit("_", 1)[0] for row in manifest}
    assert prefixes == {"alpha", "beta"}  # clips namespaced per source, no collision
    for row in manifest:
        assert (tmp_path / row["audio_filepath"]).exists()
    assert report.total_duration_s == pytest.approx(
        sum(row["duration"] for row in manifest), abs=0.05
    )
    assert (tmp_path / "dataset_card.md").exists()


def test_combined_continues_past_failure(tmp_path: Path) -> None:
    sources = [
        _source("ok1", "Good one"),
        _source("bad", "Broken", fail=True),
        _source("ok2", "Good two"),
    ]
    report = build_combined_dataset(sources, _config(tmp_path), title="Mix", source="src://mix")

    assert report.succeeded == 2 and report.failed == 1
    broken = next(s for s in report.sources if s.source_id == "bad")
    assert broken.ok is False and broken.error == "boom"
    # only the two good sources' clips are in the merged manifest
    prefixes = {
        row["audio_filepath"].split("/")[1].rsplit("_", 1)[0] for row in _manifest(tmp_path)
    }
    assert prefixes == {"ok1", "ok2"}


def test_combined_dedupes_colliding_source_ids(tmp_path: Path) -> None:
    sources = [_source("dup", "First"), _source("dup", "Second")]
    report = build_combined_dataset(sources, _config(tmp_path), title="Mix", source="src://mix")
    assert {s.source_id for s in report.sources} == {"dup", "dup-2"}  # deduped in place
    wav_prefixes = {p.name.rsplit("_", 1)[0] for p in (tmp_path / "wavs").glob("*.wav")}
    assert wav_prefixes == {"dup", "dup-2"}  # no overwrite


def test_combined_card_lists_sources(tmp_path: Path) -> None:
    sources = [_source("alpha", "Episode Alpha"), _source("beta", "Episode Beta", fail=True)]
    build_combined_dataset(sources, _config(tmp_path), title="My Show", source="src://show")
    card = (tmp_path / "dataset_card.md").read_text(encoding="utf-8")
    assert "My Show" in card
    assert "`alpha`" in card  # per-source table row
    assert "Skipped sources" in card and "Episode Beta" in card  # failures surfaced


def test_combined_resume_skips_completed_and_adds_new(tmp_path: Path) -> None:
    calls1: list = []
    first = [_source("a", "A", calls=calls1), _source("b", "B", calls=calls1)]
    r1 = build_combined_dataset(first, _config(tmp_path), title="Mix", source="src://mix")
    assert calls1 == ["a", "b"]

    # Re-run with the same two sources + a new one: a and b are cached (their WAVs
    # exist), so their run() is never called again; only c runs.
    calls2: list = []
    again = [
        _source("a", "A", calls=calls2),
        _source("b", "B", calls=calls2),
        _source("c", "C", calls=calls2),
    ]
    r2 = build_combined_dataset(again, _config(tmp_path), title="Mix", source="src://mix")
    assert calls2 == ["c"]  # a, b skipped
    assert r2.clip_count > r1.clip_count  # c's clips added
    assert {s.source_id for s in r2.sources} == {"a", "b", "c"}


def test_combined_resume_invalidated_by_config_change(tmp_path: Path) -> None:
    # Re-running with a different sample rate must NOT reuse the old clips.
    calls: list = []
    cfg1 = _config(tmp_path)
    build_combined_dataset([_source("a", "A", calls=calls)], cfg1, title="M", source="s")
    cfg2 = cfg1.model_copy(update={"sample_rate": 22050})
    build_combined_dataset([_source("a", "A", calls=calls)], cfg2, title="M", source="s")
    assert calls == ["a", "a"]  # config changed -> source re-run, not served from state


def test_combined_resume_invalidated_by_diarize_change(tmp_path: Path) -> None:
    # Enabling diarization changes which clips survive, so it must invalidate resume.
    calls: list = []
    cfg1 = _config(tmp_path)  # diarize disabled
    build_combined_dataset([_source("a", "A", calls=calls)], cfg1, title="M", source="s")
    cfg2 = cfg1.model_copy(update={"diarize": DiarizeConfig(enabled=True, mode="tag")})
    build_combined_dataset([_source("a", "A", calls=calls)], cfg2, title="M", source="s")
    assert calls == ["a", "a"]  # diarize config changed -> source re-run


def test_combined_no_resume_reruns_everything(tmp_path: Path) -> None:
    calls: list = []
    build_combined_dataset(
        [_source("a", "A", calls=calls)], _config(tmp_path), title="M", source="s"
    )
    build_combined_dataset(
        [_source("a", "A", calls=calls)], _config(tmp_path), title="M", source="s", resume=False
    )
    assert calls == ["a", "a"]  # resume=False re-runs


def test_build_from_playlist_wiring(tmp_path: Path) -> None:
    # Full playlist -> combined path with everything injected (offline).
    entries = [
        PlaylistEntry(video_id="vidA", title="Vid A", url="https://yt/a"),
        PlaylistEntry(video_id="vidB", title="Vid B", url="https://yt/b"),
    ]

    def fake_playlist(url: str):
        return "My Playlist", entries

    def fake_meta(url: str) -> dict:
        return {"id": url.rsplit("/", 1)[-1], "title": "t", "duration": 4.71, "channel": "C"}

    def fake_download(url: str, dest: Path) -> Path:
        return SAMPLE  # stand in the fixture for the "downloaded" audio

    def fake_transcribe(path, **kwargs):
        return TranscriptionResult(
            segments=[], language="en", duration_s=4.71, model_size="x", method="x", words=_words()
        )

    report = build_dataset_from_playlist(
        "https://yt/playlist",
        config=_config(tmp_path),
        playlist_fetcher=fake_playlist,
        metadata_fetcher=fake_meta,
        audio_downloader=fake_download,
        transcriber=fake_transcribe,
    )
    assert report.title == "My Playlist"
    assert report.succeeded == 2 and report.clip_count >= 2
    prefixes = {
        row["audio_filepath"].split("/")[1].rsplit("_", 1)[0] for row in _manifest(tmp_path)
    }
    assert prefixes == {"vidA", "vidB"}


def test_failed_source_leaves_no_orphan_wavs(tmp_path: Path) -> None:
    # A source that slices clips then raises must not leave orphan WAVs behind.
    sources = [_source("good", "Good"), _source("bad", "Bad", slice_then_fail=True)]
    build_combined_dataset(sources, _config(tmp_path), title="Mix", source="s")
    assert _wav_names(tmp_path) == _manifest_wav_names(tmp_path)  # tree matches manifest exactly
    assert not any(n.startswith("bad_") for n in _wav_names(tmp_path))


def test_rerun_with_fewer_clips_leaves_no_stale_wavs(tmp_path: Path) -> None:
    # First build produces several short clips; a re-run with a coarser config
    # (fingerprint change) produces fewer — the stale extras must be removed.
    fine = _config(tmp_path).model_copy(update={"segment_min_s": 0.3, "segment_max_s": 0.6})
    build_combined_dataset([_source("a", "A")], fine, title="M", source="s")
    coarse = _config(tmp_path).model_copy(update={"segment_min_s": 1.0, "segment_max_s": 2.0})
    build_combined_dataset([_source("a", "A")], coarse, title="M", source="s")
    assert _wav_names(tmp_path) == _manifest_wav_names(tmp_path)  # no stale higher-numbered WAVs


def test_combined_card_non_ascii_source_label(tmp_path: Path) -> None:
    sources = [_source("s1", "Episode du Café 🎧"), _source("s2", "Bad 💥 One", fail=True)]
    build_combined_dataset(sources, _config(tmp_path), title="My 🎙 Show", source="s")
    card = (tmp_path / "dataset_card.md").read_text(encoding="utf-8")
    assert "Episode du Café 🎧" in card and "Bad 💥 One" in card  # real chars, not surrogates
    card.split("---", 2)[1].encode("utf-8")  # YAML front matter has no lone surrogates


def test_zero_clip_source_is_ok(tmp_path: Path) -> None:
    sources = [_source("empty", "Empty", empty=True), _source("good", "Good")]
    report = build_combined_dataset(sources, _config(tmp_path), title="M", source="s")
    assert report.succeeded == 2 and report.failed == 0
    empty = next(s for s in report.sources if s.source_id == "empty")
    assert empty.ok is True and empty.clip_count == 0
    assert report.clip_count >= 1  # the good source's clips


def test_empty_playlist_yields_empty_dataset(tmp_path: Path) -> None:
    report = build_dataset_from_playlist(
        "https://yt/playlist",
        config=_config(tmp_path),
        playlist_fetcher=lambda url: ("Empty Playlist", []),
    )
    assert report.succeeded == 0 and report.clip_count == 0
    assert _manifest(tmp_path) == []
    assert "0 source(s)" in (tmp_path / "dataset_card.md").read_text(encoding="utf-8")


def test_user_owned_wavs_in_the_output_dir_are_never_deleted(tmp_path: Path) -> None:
    """--out may point at a folder that already holds the user's own recordings.

    Reconciliation used to sweep every unreferenced WAV under <out>/wavs/, which
    silently destroyed them. Only hearsay's own "<source_id>_NNNN.wav" clips are
    eligible for cleanup.
    """
    wavs = tmp_path / "wavs"
    wavs.mkdir(parents=True)
    mine = wavs / "my_precious_recording.wav"
    also_mine = wavs / "session-2024-05-01.wav"
    for path in (mine, also_mine):
        path.write_bytes(SAMPLE.read_bytes())

    build_combined_dataset([_source("a", "A")], _config(tmp_path), title="M", source="s")

    assert mine.exists() and also_mine.exists()
    # hearsay's own clips are still written and still reconciled against the manifest.
    owned = {n for n in _wav_names(tmp_path) if n.startswith("a_")}
    assert owned and owned == {n for n in _manifest_wav_names(tmp_path) if n.startswith("a_")}


def test_stale_hearsay_clips_are_still_cleaned_up(tmp_path: Path) -> None:
    # The protection above must not stop hearsay from removing its own stale output.
    wavs = tmp_path / "wavs"
    wavs.mkdir(parents=True)
    stale = wavs / "a_9999.wav"
    stale.write_bytes(SAMPLE.read_bytes())
    build_combined_dataset([_source("a", "A")], _config(tmp_path), title="M", source="s")
    assert not stale.exists()
