"""Build a TTS/STT dataset from one source: transcribe -> segment -> slice -> index.

Orchestrates the dataset pipeline for a single audio source (a local file or one
YouTube video). Network/transcription steps are injected as callables so the
build logic is testable offline against fixtures. Phase D4 layers batch merging
(playlists/feeds) on top of ``build_dataset``.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hearsay import __version__
from hearsay.dataset.audio import (
    ensure_filter,
    ensure_tools,
    probe_duration,
    probe_sample_rate,
    slice_clip,
)
from hearsay.dataset.diarize import Diarizer, assign_speaker, dominant_speaker, load_diarizer
from hearsay.dataset.filters import detect_clipping_drops, filter_segments
from hearsay.dataset.formats import (
    write_combined_card,
    write_dataset_card,
    write_dropped,
    write_indices,
    write_per_speaker_indices,
)
from hearsay.dataset.models import (
    DATASET_FORMATS,
    BuildReport,
    CombinedReport,
    DatasetClip,
    DatasetConfig,
    DatasetSegment,
    DiarizeConfig,
    DropRecord,
    SourceResult,
)
from hearsay.dataset.segmentation import segment_words
from hearsay.errors import AudioDownloadError, AudioExportError, HearsayError, InvalidSourceError
from hearsay.feeds import Episode, Feed, download_episode, fetch_feed
from hearsay.models import SourceMetadata, Word
from hearsay.transcribe import (
    DEFAULT_MODEL,
    UNKNOWN_LANGUAGE,
    UNUSABLE_WITHOUT_FINE_TUNE,
    TranscriptionResult,
    resolve_method,
    transcribe_audio,
)
from hearsay.youtube import (
    PlaylistEntry,
    download_audio,
    fetch_playlist,
    fetch_raw_metadata,
    parse_metadata,
)

Transcriber = Callable[..., TranscriptionResult]
MetadataFetcher = Callable[[str], dict]
AudioDownloader = Callable[[str, Path], Path]
EpisodeDownloader = Callable[[str, Path], Path]
SourceRunner = Callable[
    [Path, DatasetConfig, str],
    tuple[list[DatasetClip], list[DropRecord], SourceMetadata],
]
_STATE_FILE = "_state.json"

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
# The clip filenames hearsay generates: "<source_id>_NNNN.wav" (see _produce_clips).
# Used to tell hearsay's own output apart from a user's files in the same folder.
_CLIP_NAME = re.compile(r"^(.+)_\d{4}\.wav$")
# A clip shorter than this (after slicing) is a near-empty ffmpeg artifact, not audio.
_MIN_CLIP_S = 0.01
# A few unreadable stretches in a long recording are normal and are logged as drops; a
# source where slicing fails this often is broken, and failing loudly beats emitting a
# quietly decimated dataset.
_MAX_SLICE_FAILURE_RATE = 0.2
_MIN_SLICE_FAILURES = 3
# Whisper's smallest checkpoints are usable for reading a transcript but not for
# building one: with word_timestamps=True their alignment pass can omit a word that is
# plainly audible, and the clip then ships paired with a transcript missing it. That is
# silent audio/text misalignment — exactly what corrupts a TTS/STT training run — and
# no downstream filter can see it, because the text that survives is well-formed.
_ALIGNMENT_RISK_MODELS = frozenset({"whisper-tiny", "whisper-base"})
_ALIGNMENT_WARNING = (
    "{method} is too small for reliable word alignment: it can drop audible words from "
    "a clip's transcript, silently misaligning audio and text. Use --model small (or "
    "larger, or parakeet on Apple Silicon) for datasets you intend to train on."
)
_MIXED_SPEAKER_WARNING = (
    "diarization not installed: this dataset contains MIXED speakers — fine for STT, "
    'but not single-voice TTS. Install "hearsay[diarize]" and set HF_TOKEN to enable '
    "per-speaker / dominant-speaker export."
)


def _with_detected_language(config: DatasetConfig, language: str) -> DatasetConfig:
    """Pin the filters to ``language`` when the user didn't name one with ``--lang``.

    The script and character-rate filters are language-specific. Leaving them on a
    hardcoded "en" meant an Urdu, Arabic, Russian, Hindi or Chinese source had every
    clip dropped as ``non_target_script`` — a silently empty dataset from a perfectly
    good recording. Transcription already detects the language, so use it.
    """
    if config.filters.target_language is not None:
        return config
    return config.model_copy(
        update={"filters": config.filters.model_copy(update={"target_language": language})}
    )


_HF_LJSPEECH_CLASH = (
    "formats 'hf' and 'ljspeech' cannot share a dataset folder: HuggingFace "
    "`audiofolder` scans the tree for a metadata index and refuses a mix of "
    "extensions (\"Found metadata files with different extensions: ['.csv', "
    "'.jsonl']\"), so metadata.csv makes metadata.jsonl unloadable. Build the "
    "HuggingFace index on its own: --format hf (optionally with --format jsonl)."
)


def _format_warnings(formats: list[str]) -> list[str]:
    """Warn about index formats that cannot coexist in one dataset folder."""
    chosen = set(formats)
    return [_HF_LJSPEECH_CLASH] if {"hf", "ljspeech"} <= chosen else []


_LOW_RESOURCE_WARNING = (
    "stock Whisper cannot really transcribe {language}: its own FLEURS table puts this "
    "language above 90% word error, and for some it transliterates into a neighbouring "
    "language instead of failing, so the text looks fluent and is wrong. Clips will be "
    "paired with text that is largely wrong. Point --model at a CTranslate2 fine-tune "
    "for this language; hearsay accepts a Hugging Face id or a local path."
)


def _low_resource_warnings(model_size: str, language: str | None) -> list[str]:
    """Warn when a stock Whisper is being used on a language it cannot really read.

    A custom model is assumed to be the fine-tune this is asking for, so it is exempt.
    """
    if not language or language.split("-")[0] not in UNUSABLE_WITHOUT_FINE_TUNE:
        return []
    if "/" in model_size or Path(model_size).is_dir():
        return []
    return [_LOW_RESOURCE_WARNING.format(language=language)]


def _alignment_warnings(model_size: str) -> list[str]:
    """Warn up front when the requested model is too small for word alignment.

    Combined builds transcribe per source, so the warning is derived from the
    *requested* model (resolving ``auto``) rather than from a result, letting it reach
    the user before a long multi-source build rather than after it.
    """
    try:
        method = resolve_method(model_size)
    except HearsayError:  # unknown model — the transcriber reports it far better
        return []
    return [_ALIGNMENT_WARNING.format(method=method)] if method in _ALIGNMENT_RISK_MODELS else []


def _resolve_diarizer(
    config: DatasetConfig, injected: Diarizer | None
) -> tuple[Diarizer | None, list[str]]:
    """Resolve the diarizer to use, degrading to a warning when the extra is absent.

    An injected diarizer (tests) wins. Otherwise, if diarization is requested but
    ``pyannote.audio`` is not installed, return ``(None, [warning])`` so the build
    proceeds with a mixed-speaker dataset (SPEC: degrade cleanly). When it IS installed,
    return a real diarizer; an auth/gating error then surfaces (actionably) on first use.
    """
    if injected is not None:
        return injected, []
    if not config.diarize.enabled:
        return None, []
    from importlib.util import find_spec

    try:
        available = find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        available = False
    if not available:
        return None, [_MIXED_SPEAKER_WARNING]
    return load_diarizer(config.diarize), []


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_id(text: str) -> str:
    """A filesystem-safe, collision-resistant id stem from a video id or title."""
    cleaned = _UNSAFE.sub("-", text).strip("-")
    return cleaned[:60] or "clip"


def _validate_formats(formats: list[str]) -> None:
    unknown = [f for f in formats if f not in DATASET_FORMATS]
    if unknown:
        raise AudioExportError(
            f"Unknown dataset format(s): {', '.join(unknown)}.",
            hint=f"Choose from: {', '.join(DATASET_FORMATS)}.",
        )


def build_dataset(
    source_audio: Path,
    words: list[Word],
    meta: SourceMetadata,
    *,
    config: DatasetConfig,
    language: str = "en",
    source_platform: str = "local",
    diarizer: Diarizer | None = None,
    transcription_method: str | None = None,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
) -> BuildReport:
    """Segment ``words``, slice ``source_audio`` into clips, and write the dataset.

    Writes ``wavs/<id>_NNNN.wav`` plus the requested index formats and a
    ``dataset_card.md`` into ``config.out_dir``. Returns a :class:`BuildReport`.
    Raises AudioExportError when ffmpeg/ffprobe are missing or a slice fails.
    """
    ensure_tools()
    _validate_formats(config.formats)
    if config.normalize:
        ensure_filter("loudnorm")
    _make_wavs_dir(config.out_dir)
    diarizer, warnings = _resolve_diarizer(config, diarizer)
    warnings += _format_warnings(config.formats)
    warnings += _low_resource_warnings(transcription_method or "", language)
    if transcription_method in _ALIGNMENT_RISK_MODELS:
        warnings.append(_ALIGNMENT_WARNING.format(method=transcription_method))
    src_rate = probe_sample_rate(source_audio)
    if 0 < src_rate < config.sample_rate:
        warnings.append(
            f"source audio is {src_rate} Hz but the target is {config.sample_rate} Hz; "
            "upsampling adds no real high-frequency content."
        )
    source_id = _safe_id(meta.video_id or meta.title)
    clips, drops = _produce_clips(
        source_audio,
        words,
        out_dir=config.out_dir,
        source_id=source_id,
        config=_with_detected_language(config, language),
        diarizer=diarizer,
    )

    out_dir = config.out_dir
    files = write_indices(out_dir, clips, config.formats)
    files.append(write_dropped(out_dir, drops))
    if config.diarize.enabled and config.diarize.mode == "per_speaker":
        files.extend(write_per_speaker_indices(out_dir, clips, config.formats))
    report = BuildReport(
        out_dir=str(out_dir),
        source=meta.source,
        clip_count=len(clips),
        total_duration_s=sum(c.duration_s for c in clips),
        oversized_count=sum(1 for c in clips if c.oversized),
        sample_rate=config.sample_rate,
        language=language,
        formats=list(config.formats),
        files=files,
        clips=clips,
        dropped_count=len(drops),
        drops_by_reason=dict(Counter(d.filter for d in drops)),
        drops=drops,
        warnings=warnings,
    )
    card = write_dataset_card(
        out_dir, report, meta, version=version, generated_at=now(), source_platform=source_platform
    )
    report.files.append(card)
    return report


def _make_wavs_dir(out_dir: Path) -> None:
    try:
        (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioExportError(
            f"Could not create the dataset directory {out_dir}: {exc.strerror or exc}",
            hint="Pick a writable --out path whose parent is not an existing file.",
        ) from exc


def _produce_clips(
    source_audio: Path,
    words: list[Word],
    *,
    out_dir: Path,
    source_id: str,
    config: DatasetConfig,
    diarizer: Diarizer | None = None,
) -> tuple[list[DatasetClip], list[DropRecord]]:
    """Segment, filter, and slice one source's audio into ``out_dir/wavs`` (no index write).

    Clip ids are ``<source_id>_NNNN`` so several sources can share one ``wavs/`` tree
    without colliding (the combined builder ensures ``source_id`` is unique). Returns
    ``(clips, drops)`` for the caller to merge and index. Assumes ``out_dir/wavs`` exists.
    """
    segments = segment_words(words, min_s=config.segment_min_s, max_s=config.segment_max_s)

    # Clamp clip ends to the true media length so a segment whose last word ends a
    # hair past the audio (common) yields a real, shorter clip — and one entirely
    # past EOF is skipped rather than written as a 0-second "ghost" clip.
    source_duration = probe_duration(source_audio)

    # Pre-slice candidates (clamped span); Tier-1 filters run here so we never
    # waste an ffmpeg slice on a clip we are going to drop.
    pad = config.edge_pad_s
    candidates: list[tuple[str, DatasetSegment, float]] = []
    spans: dict[str, tuple[float, float]] = {}
    for seg_index, seg in enumerate(segments, start=1):
        start = seg.start_s
        end = min(seg.end_s, source_duration) if source_duration > 0 else seg.end_s
        if end - start <= 0:
            continue  # nothing left to slice (segment lies past the source end)
        ref = f"{source_id}_seg{seg_index:04d}"
        # Filter on the verbatim word extent (unpadded), so padding never changes which
        # clips survive the duration/rate filters — it only adds audio to what is kept.
        # (The `oversized` flag and the max-duration filter therefore both describe the
        # word extent; a kept clip's written WAV can be up to 2*pad seconds longer.)
        candidates.append((ref, seg, end - start))
        # Pad the slice *window* so onset/offset phonemes the ASR word-timestamps clip
        # are captured, with a little edge silence/context (clamped to the media bounds).
        # When the source duration is unknown (ffprobe failed) we do not extend past the
        # unverifiable end — over-running EOF would yield a short WAV whose trailing fade
        # (anchored at the requested duration) never fires, leaving the very click we add
        # the fade to remove.
        pstart = max(0.0, start - pad)
        pend = min(end + pad, source_duration) if source_duration > 0 else end
        spans[ref] = (pstart, pend)

    # Tier-1 quality filters (default on). Note an "oversized" clip (longer than
    # segment_max_s) is dropped here by the duration filter whenever
    # filters.max_duration_s == segment_max_s (the defaults) — intended.
    kept, drops = filter_segments(candidates, config.filters)

    clips: list[DatasetClip] = []
    index = 0
    slice_failures = 0
    for ref, seg, _duration in kept:
        start, end = spans[ref]
        index += 1
        clip_id = f"{source_id}_{index:04d}"
        rel = f"wavs/{clip_id}.wav"
        dest = out_dir / rel
        try:
            slice_clip(
                source_audio,
                start,
                end,
                dest,
                sample_rate=config.sample_rate,
                normalize=config.normalize,
                fade_s=config.fade_s,
            )
        except AudioExportError as exc:
            # One unreadable stretch of a long recording used to abort the whole source,
            # throwing away every clip already cut from it. Log the clip and keep going;
            # a genuinely broken file trips the failure ceiling below instead.
            dest.unlink(missing_ok=True)
            index -= 1
            slice_failures += 1
            drops.append(
                DropRecord(
                    clip=ref,
                    filter="slice_failed",
                    value=str(exc.message)[:80],
                    threshold="ffmpeg ok",
                    text=seg.text[:80],
                )
            )
            if slice_failures > max(_MIN_SLICE_FAILURES, len(kept) * _MAX_SLICE_FAILURE_RATE):
                raise
            continue
        duration_s = probe_duration(dest)
        if duration_s <= _MIN_CLIP_S:
            dest.unlink(missing_ok=True)  # empty/near-empty slice — don't record it
            index -= 1
            drops.append(
                DropRecord(
                    clip=ref,
                    filter="empty_slice",
                    value="0.00s",
                    threshold=">0s",
                    text=seg.text[:80],
                )
            )
            continue
        clips.append(
            DatasetClip(
                id=clip_id,
                audio_path=rel,
                text=seg.text,
                start_s=start,
                end_s=end,
                duration_s=duration_s,
                oversized=seg.oversized,
            )
        )

    # Tier-2 (opt-in): drop hard-clipped clips and remove their WAVs.
    clips, clip_drops = detect_clipping_drops(clips, out_dir, config.filters)
    for drop in clip_drops:
        (out_dir / "wavs" / f"{drop.clip}.wav").unlink(missing_ok=True)
    drops.extend(clip_drops)

    # Optional diarization: label each clip's speaker, drop cross-speaker clips, and
    # (in dominant mode) keep only this source's most-spoken speaker.
    if diarizer is not None and config.diarize.enabled:
        clips, diar_drops = _assign_speakers(
            clips, source_audio, out_dir, source_id, diarizer, config.diarize
        )
        drops.extend(diar_drops)
    return clips, drops


def _assign_speakers(
    clips: list[DatasetClip],
    source_audio: Path,
    out_dir: Path,
    source_id: str,
    diarizer: Diarizer,
    dcfg: DiarizeConfig,
) -> tuple[list[DatasetClip], list[DropRecord]]:
    """Tag each clip with its dominant speaker; drop cross-speaker (and non-dominant) clips."""
    turns = diarizer(source_audio)  # may raise DiarizationError (auth/gating) — actionable
    keep_only = dominant_speaker(turns) if dcfg.mode == "dominant" else None
    kept: list[DatasetClip] = []
    drops: list[DropRecord] = []
    for clip in clips:
        speaker, purity = assign_speaker(clip.start_s, clip.end_s, turns)
        reason: tuple[str, str, str] | None = None
        if speaker is None:
            reason = ("no_speaker", "0.00", ">0 overlap")
        elif purity < dcfg.min_purity:
            reason = ("cross_speaker", f"{purity:.2f}", f">={dcfg.min_purity}")
        elif keep_only is not None and speaker != keep_only:
            reason = ("non_dominant_speaker", speaker, keep_only)
        if reason is not None:
            (out_dir / clip.audio_path).unlink(missing_ok=True)
            drops.append(
                DropRecord(
                    clip=clip.id,
                    filter=reason[0],
                    value=reason[1],
                    threshold=reason[2],
                    text=clip.text[:80],
                )
            )
            continue
        clip.speaker = f"{source_id}:{speaker}"  # namespace per source (labels are per-file)
        kept.append(clip)
    return kept, drops


def build_dataset_from_file(
    path: Path,
    *,
    config: DatasetConfig,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    vad_filter: bool = True,
    transcriber: Transcriber = transcribe_audio,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
) -> BuildReport:
    """Transcribe a local file (with word timings) and build a dataset from it."""
    if not path.exists():
        raise InvalidSourceError(
            f"File not found: {path}", hint="Check the path, or pass a YouTube URL."
        )
    result = transcriber(
        path, model_size=model_size, language=language, vad_filter=vad_filter, word_timestamps=True
    )
    words = _require_words(result)
    meta = SourceMetadata(
        title=path.stem,
        source=str(path),
        channel="Local file",
        duration_s=result.duration_s,
        video_id=path.stem,
    )
    return build_dataset(
        path,
        words,
        meta,
        config=config,
        language=result.language,
        source_platform="local",
        transcription_method=result.method,
        version=version,
        now=now,
    )


def build_dataset_from_media_url(
    url: str,
    *,
    config: DatasetConfig,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    vad_filter: bool = True,
    metadata_fetcher: MetadataFetcher = fetch_raw_metadata,
    audio_downloader: AudioDownloader = download_audio,
    transcriber: Transcriber = transcribe_audio,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
) -> BuildReport:
    """Download one media URL's audio, transcribe it, and build a dataset.

    Any site yt-dlp supports works — YouTube, Dailymotion, SoundCloud, Twitch and
    ~1800 others — because both the metadata and the audio come from yt-dlp, which
    takes the URL verbatim. Nothing on this path is YouTube-specific.

    The downloaded audio is always deleted; the dataset is built before the temp
    directory is removed (clips are sliced from that audio).
    """
    raw = metadata_fetcher(url)
    meta = parse_metadata(raw, url)
    with tempfile.TemporaryDirectory(prefix="hearsay-ds-") as tmp:
        audio_path = audio_downloader(url, Path(tmp))
        result = transcriber(
            audio_path,
            model_size=model_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
        )
        words = _require_words(result)
        return build_dataset(
            audio_path,
            words,
            meta,
            config=config,
            language=result.language,
            source_platform="youtube",
            transcription_method=result.method,
            version=version,
            now=now,
        )


def _require_words(result: TranscriptionResult) -> list[Word]:
    """Return the result's words, or raise a friendly error if there are none."""
    if not result.words:
        raise AudioExportError(
            "Transcription produced no word-level timings, so no clips can be cut.",
            hint="The audio may be silent or non-speech; try a clearer file or another --model.",
        )
    return result.words


# --- Combined builds: playlists / channels / feeds -> one merged dataset (D4) ---


@dataclass
class DatasetSource:
    """One item of a combined build: a unique clip-id namespace + how to produce it.

    ``run(out_dir, config, source_id)`` downloads/transcribes this source and slices
    its clips into the shared ``out_dir/wavs`` under ``<source_id>_NNNN``, returning
    ``(clips, drops, meta)``. ``source_id`` is assigned by the combined builder after
    de-duplication, so it may differ from the value set here.
    """

    source_id: str
    label: str
    run: SourceRunner


def _ensure_unique_source_ids(sources: list[DatasetSource]) -> None:
    """Make each source's id unique in place (suffix collisions -2, -3, ...).

    Distinct sources can derive the same id (e.g. two episode titles that slugify
    alike); without this their clips would overwrite each other in the shared tree.
    """
    used: set[str] = set()
    for src in sources:
        base = src.source_id
        candidate = base
        n = 1
        while candidate in used:
            n += 1
            candidate = f"{base}-{n}"
        used.add(candidate)
        src.source_id = candidate


def _error_message(exc: Exception) -> str:
    return getattr(exc, "message", None) or str(exc) or exc.__class__.__name__


def _state_fingerprint(config: DatasetConfig) -> str:
    """A stable signature of the clip-affecting config, to invalidate stale resume state.

    Sample rate, segment bounds, normalize, edge padding, the de-click fade, filters,
    and diarization all change the WAVs / which clips survive, so resuming across a
    change of any of these would mix incompatible clips. (Output formats are excluded —
    they only affect index files, which are always rewritten.)
    """
    return json.dumps(
        {
            "sample_rate": config.sample_rate,
            "segment_min_s": config.segment_min_s,
            "segment_max_s": config.segment_max_s,
            "normalize": config.normalize,
            "edge_pad_s": config.edge_pad_s,
            "fade_s": config.fade_s,
            "filters": config.filters.model_dump(),
            # diarization changes which clips survive / how they're labelled; the token
            # is excluded (it doesn't change the output and shouldn't leak into state).
            "diarize": {k: v for k, v in config.diarize.model_dump().items() if k != "hf_token"},
        },
        sort_keys=True,
    )


def _load_state(out_dir: Path, fingerprint: str) -> dict:
    """Load the per-source resume state, or {} if absent/unreadable/config-changed."""
    try:
        data = json.loads((out_dir / _STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("fingerprint") != fingerprint:
        return {}  # config changed since the last run — re-build everything
    sources = data.get("sources")
    return sources if isinstance(sources, dict) else {}


def _save_state(out_dir: Path, fingerprint: str, sources: dict) -> None:
    payload = json.dumps({"fingerprint": fingerprint, "sources": sources})
    (out_dir / _STATE_FILE).write_text(payload, encoding="utf-8")


def _valid_cache_entry(cached: dict) -> bool:
    """True if a per-source resume record has the expected shape (else treat as a miss)."""
    return (
        isinstance(cached.get("result"), dict)
        and "ok" in cached["result"]
        and isinstance(cached.get("clips"), list)
        and isinstance(cached.get("drops"), list)
    )


def _clips_present(out_dir: Path, clips: list[dict]) -> bool:
    """True if every cached clip's WAV still exists on disk (safe to reuse on resume)."""
    return all((out_dir / c["audio_path"]).exists() for c in clips)


def _write_merged(
    out_dir: Path, clips: list[DatasetClip], drops: list[DropRecord], formats: list[str]
) -> list[str]:
    """Write the merged indices + drop log over all accumulated clips. Returns filenames."""
    files = write_indices(out_dir, clips, formats)
    files.append(write_dropped(out_dir, drops))
    return files


def _reconcile_wavs(out_dir: Path, clips: list[DatasetClip], owned_ids: set[str]) -> None:
    """Delete WAVs *this dataset wrote* that the final manifest no longer references.

    Keeps the dataset's strong invariant — every WAV is referenced and every clip
    has a WAV — by removing orphans left when a source fails after slicing, or stale
    higher-numbered clips from a previous run that produced more clips.

    Deletion is restricted to the ``<source_id>_NNNN.wav`` names hearsay generates for
    ``owned_ids``. ``--out`` is an ordinary directory a user may well point at a folder
    that already holds their own recordings, and an unrestricted sweep silently erased
    every one of them — so anything hearsay did not create is left untouched.
    """
    referenced = {c.audio_path for c in clips}
    wavs_dir = out_dir / "wavs"
    if not wavs_dir.is_dir():
        return
    for wav in wavs_dir.glob("*.wav"):
        if f"wavs/{wav.name}" in referenced:
            continue
        match = _CLIP_NAME.match(wav.name)
        if match and match.group(1) in owned_ids:
            wav.unlink(missing_ok=True)


def build_combined_dataset(
    sources: list[DatasetSource],
    config: DatasetConfig,
    *,
    title: str,
    source: str,
    language: str = "en",
    source_platform: str = "youtube",
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
    on_item: Callable[[int, int, DatasetSource], None] | None = None,
    resume: bool = True,
    warnings: list[str] | None = None,
) -> CombinedReport:
    """Merge many sources into one dataset (shared ``wavs/`` + manifests).

    Each source runs through transcribe -> segment -> filter -> slice; per-item
    failures are recorded and the build continues. The merged indices are rewritten
    after every source (crash-safe), and a ``_state.json`` lets a re-run skip sources
    already completed (clips present on disk). Returns a :class:`CombinedReport`.
    """
    ensure_tools()
    _validate_formats(config.formats)
    if config.normalize:
        ensure_filter("loudnorm")
    _make_wavs_dir(config.out_dir)
    out_dir = config.out_dir
    _ensure_unique_source_ids(sources)
    fingerprint = _state_fingerprint(config)
    state = _load_state(out_dir, fingerprint) if resume else {}

    results: list[SourceResult] = []
    all_clips: list[DatasetClip] = []
    all_drops: list[DropRecord] = []
    total = len(sources)
    for index, src in enumerate(sources, start=1):
        if on_item is not None:
            on_item(index, total, src)
        cached = state.get(src.source_id) if resume else None
        if (
            cached
            and _valid_cache_entry(cached)
            and cached["result"]["ok"]
            and _clips_present(out_dir, cached["clips"])
        ):
            clips = [DatasetClip(**c) for c in cached["clips"]]
            drops = [DropRecord(**d) for d in cached["drops"]]
            results.append(SourceResult(**cached["result"]))
        else:
            try:
                clips, drops, _meta = src.run(out_dir, config, src.source_id)
                result = SourceResult(
                    source_id=src.source_id,
                    label=src.label,
                    ok=True,
                    clip_count=len(clips),
                    duration_s=sum(c.duration_s for c in clips),
                    dropped=len(drops),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # one bad source must not abort the batch
                clips, drops = [], []
                result = SourceResult(
                    source_id=src.source_id, label=src.label, ok=False, error=_error_message(exc)
                )
            results.append(result)
            state[src.source_id] = {
                "clips": [c.model_dump() for c in clips],
                "drops": [d.model_dump() for d in drops],
                "result": result.model_dump(),
            }
            _save_state(out_dir, fingerprint, state)
        all_clips.extend(clips)
        all_drops.extend(drops)
        _write_merged(out_dir, all_clips, all_drops, config.formats)  # crash-safe incremental write

    # Remove orphan/stale WAVs so the on-disk tree matches the merged manifest exactly.
    # Only ids this build (or a previous run recorded in state) produced are candidates.
    _reconcile_wavs(out_dir, all_clips, {s.source_id for s in sources} | set(state))
    files = _write_merged(out_dir, all_clips, all_drops, config.formats)
    if config.diarize.enabled and config.diarize.mode == "per_speaker":
        files.extend(write_per_speaker_indices(out_dir, all_clips, config.formats))
    report = CombinedReport(
        out_dir=str(out_dir),
        source=source,
        title=title,
        clip_count=len(all_clips),
        total_duration_s=sum(c.duration_s for c in all_clips),
        oversized_count=sum(1 for c in all_clips if c.oversized),
        dropped_count=len(all_drops),
        drops_by_reason=dict(Counter(d.filter for d in all_drops)),
        sample_rate=config.sample_rate,
        language=language,
        formats=list(config.formats),
        sources=results,
        warnings=warnings or [],
        succeeded=sum(1 for r in results if r.ok),
        failed=sum(1 for r in results if not r.ok),
    )
    card = write_combined_card(
        out_dir, report, version=version, generated_at=now(), source_platform=source_platform
    )
    report.files = [*files, card]
    return report


def _youtube_source(
    entry: PlaylistEntry,
    *,
    model_size: str,
    language: str | None,
    vad_filter: bool,
    metadata_fetcher: MetadataFetcher,
    audio_downloader: AudioDownloader,
    transcriber: Transcriber,
    diarizer: Diarizer | None = None,
) -> DatasetSource:
    def run(
        out_dir: Path, config: DatasetConfig, source_id: str
    ) -> tuple[list[DatasetClip], list[DropRecord], SourceMetadata]:
        raw = metadata_fetcher(entry.url)
        meta = parse_metadata(raw, entry.url)
        with tempfile.TemporaryDirectory(prefix="hearsay-ds-") as tmp:
            audio_path = audio_downloader(entry.url, Path(tmp))
            result = transcriber(
                audio_path,
                model_size=model_size,
                language=language,
                vad_filter=vad_filter,
                word_timestamps=True,
            )
            words = _require_words(result)
            clips, drops = _produce_clips(
                audio_path,
                words,
                out_dir=out_dir,
                source_id=source_id,
                config=_with_detected_language(config, result.language),
                diarizer=diarizer,
            )
        return clips, drops, meta

    return DatasetSource(source_id=_safe_id(entry.video_id), label=entry.title, run=run)


def _episode_source(
    episode: Episode,
    show_title: str,
    *,
    model_size: str,
    language: str | None,
    vad_filter: bool,
    episode_downloader: EpisodeDownloader,
    transcriber: Transcriber,
    diarizer: Diarizer | None = None,
) -> DatasetSource:
    def run(
        out_dir: Path, config: DatasetConfig, source_id: str
    ) -> tuple[list[DatasetClip], list[DropRecord], SourceMetadata]:
        if not episode.audio_url:
            raise AudioDownloadError(
                f"Episode '{episode.title}' has no audio enclosure.",
                hint="This feed entry has no downloadable media; try another episode.",
            )
        with tempfile.TemporaryDirectory(prefix="hearsay-ds-") as tmp:
            audio_path = episode_downloader(episode.audio_url, Path(tmp))
            result = transcriber(
                audio_path,
                model_size=model_size,
                language=language,
                vad_filter=vad_filter,
                word_timestamps=True,
            )
            words = _require_words(result)
            clips, drops = _produce_clips(
                audio_path,
                words,
                out_dir=out_dir,
                source_id=source_id,
                config=_with_detected_language(config, result.language),
                diarizer=diarizer,
            )
        meta = SourceMetadata(
            title=episode.title,
            source=episode.audio_url,
            channel=show_title,
            duration_s=episode.duration_s or result.duration_s,
            video_id=source_id,
        )
        return clips, drops, meta

    return DatasetSource(source_id=_safe_id(episode.title), label=episode.title, run=run)


def build_dataset_from_playlist(
    url: str,
    *,
    config: DatasetConfig,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    vad_filter: bool = True,
    limit: int | None = None,
    playlist_fetcher: Callable[[str], tuple[str, list[PlaylistEntry]]] = fetch_playlist,
    metadata_fetcher: MetadataFetcher = fetch_raw_metadata,
    audio_downloader: AudioDownloader = download_audio,
    transcriber: Transcriber = transcribe_audio,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
    on_item: Callable[[int, int, DatasetSource], None] | None = None,
    resume: bool = True,
) -> CombinedReport:
    """List a YouTube playlist/channel and merge its videos into one dataset."""
    title, entries = playlist_fetcher(url)
    if limit is not None:
        entries = entries[:limit]
    diarizer, warnings = _resolve_diarizer(config, None)
    warnings += _format_warnings(config.formats) + _alignment_warnings(model_size)
    sources = [
        _youtube_source(
            entry,
            model_size=model_size,
            language=language,
            vad_filter=vad_filter,
            metadata_fetcher=metadata_fetcher,
            audio_downloader=audio_downloader,
            transcriber=transcriber,
            diarizer=diarizer,
        )
        for entry in entries
    ]
    return build_combined_dataset(
        sources,
        config,
        title=title,
        source=url,
        language=language or UNKNOWN_LANGUAGE,
        source_platform="youtube",
        version=version,
        now=now,
        on_item=on_item,
        resume=resume,
        warnings=warnings,
    )


def build_dataset_from_feed(
    url: str,
    *,
    config: DatasetConfig,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    vad_filter: bool = True,
    limit: int | None = None,
    feed_fetcher: Callable[[str], Feed] = fetch_feed,
    episode_downloader: EpisodeDownloader = download_episode,
    transcriber: Transcriber = transcribe_audio,
    version: str = __version__,
    now: Callable[[], str] = _utc_now_iso,
    on_item: Callable[[int, int, DatasetSource], None] | None = None,
    resume: bool = True,
) -> CombinedReport:
    """Fetch a podcast feed and merge its episodes into one dataset."""
    feed = feed_fetcher(url)
    episodes = list(feed.episodes)
    if limit is not None:
        episodes = episodes[:limit]
    diarizer, warnings = _resolve_diarizer(config, None)
    warnings += _format_warnings(config.formats) + _alignment_warnings(model_size)
    sources = [
        _episode_source(
            episode,
            feed.title,
            model_size=model_size,
            language=language,
            vad_filter=vad_filter,
            episode_downloader=episode_downloader,
            transcriber=transcriber,
            diarizer=diarizer,
        )
        for episode in episodes
    ]
    return build_combined_dataset(
        sources,
        config,
        title=feed.title,
        source=url,
        language=language or UNKNOWN_LANGUAGE,
        source_platform="podcast",
        version=version,
        now=now,
        on_item=on_item,
        resume=resume,
        warnings=warnings,
    )
