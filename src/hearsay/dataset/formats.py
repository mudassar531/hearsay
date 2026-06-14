"""Write dataset index files and the dataset card (pure file writers).

Three index formats over one shared ``wavs/`` tree (see docs/dataset-mode-design.md
section 1):

* **LJSpeech** ``metadata.csv`` — pipe-delimited, no header, ``id|text|text`` (the
  verbatim transcript is duplicated into the transcription and normalized columns
  so the file loads in both Coqui — which reads column 3 — and Piper — which reads
  the last column — without a multi-speaker misparse).
* **NeMo-style** ``manifest.jsonl`` — one JSON object per line with
  ``audio_filepath`` (relative to the manifest), ``duration`` (seconds, float),
  ``text``, and ``offset`` (0.0, for forward-compat with segmented clips).
* **HuggingFace ``audiofolder``** ``metadata.jsonl`` — ``file_name`` +
  ``transcription`` (named ``metadata.jsonl`` so it never collides with the
  LJSpeech ``metadata.csv``).

Plus ``dataset_card.md``: HuggingFace-compatible YAML front matter (lowercase
license id, defaulting to ``unknown``) and standard markdown sections recording
provenance and an honest rights/consent note. Hearsay custom provenance keys live
in the markdown body, not the YAML (where unknown keys are ignored).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hearsay.dataset.models import BuildReport, CombinedReport, DatasetClip, DropRecord
from hearsay.models import SourceMetadata
from hearsay.timefmt import format_timestamp

_WS = re.compile(r"\s+")


def _clean_pipe_text(text: str) -> str:
    """Make text safe for a pipe-delimited, one-line CSV cell."""
    return _WS.sub(" ", text.replace("|", " ")).strip()


def write_ljspeech(out_dir: Path, clips: list[DatasetClip], name: str = "metadata.csv") -> str:
    """Write LJSpeech CSV (``id|text|text``, no header). Returns the filename."""
    lines = [f"{c.id}|{_clean_pipe_text(c.text)}|{_clean_pipe_text(c.text)}" for c in clips]
    body = "\n".join(lines)
    (out_dir / name).write_text(body + ("\n" if body else ""), encoding="utf-8")
    return name


def _manifest_obj(clip: DatasetClip) -> dict:
    obj: dict = {
        "audio_filepath": clip.audio_path,
        "duration": round(clip.duration_s, 3),
        "text": clip.text,
        "offset": 0.0,
    }
    if clip.speaker is not None:  # NeMo manifests carry an optional speaker field
        obj["speaker"] = clip.speaker
    return obj


def write_jsonl_manifest(
    out_dir: Path, clips: list[DatasetClip], name: str = "manifest.jsonl"
) -> str:
    """Write a NeMo-style JSONL manifest (``manifest.jsonl`` by default). Returns its name."""
    lines = [json.dumps(_manifest_obj(c), ensure_ascii=False) for c in clips]
    body = "\n".join(lines)
    (out_dir / name).write_text(body + ("\n" if body else ""), encoding="utf-8")
    return name


def write_hf_audiofolder(out_dir: Path, clips: list[DatasetClip]) -> str:
    """Write a HuggingFace ``audiofolder`` ``metadata.jsonl``. Returns its name."""
    lines = [
        json.dumps({"file_name": c.audio_path, "transcription": c.text}, ensure_ascii=False)
        for c in clips
    ]
    body = "\n".join(lines)
    (out_dir / "metadata.jsonl").write_text(body + ("\n" if body else ""), encoding="utf-8")
    return "metadata.jsonl"


_WRITERS = {
    "ljspeech": write_ljspeech,
    "jsonl": write_jsonl_manifest,
    "hf": write_hf_audiofolder,
}


def write_indices(out_dir: Path, clips: list[DatasetClip], formats: list[str]) -> list[str]:
    """Write each requested index format; return the relative filenames written."""
    return [_WRITERS[fmt](out_dir, clips) for fmt in formats if fmt in _WRITERS]


def write_dropped(out_dir: Path, drops: list[DropRecord]) -> str:
    """Write the per-clip drop log ``dropped.jsonl`` (one record per line). Returns its name."""
    lines = [json.dumps(d.model_dump(), ensure_ascii=False) for d in drops]
    body = "\n".join(lines)
    (out_dir / "dropped.jsonl").write_text(body + ("\n" if body else ""), encoding="utf-8")
    return "dropped.jsonl"


_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _speaker_slug(speaker: str) -> str:
    """A filesystem-safe slug for a per-speaker index filename."""
    return _SLUG_UNSAFE.sub("-", speaker).strip("-") or "unknown"


def write_per_speaker_indices(
    out_dir: Path, clips: list[DatasetClip], formats: list[str]
) -> list[str]:
    """Write a per-speaker index (over the shared ``wavs/``) for each speaker. Returns filenames.

    Gives a drop-in single-voice corpus per speaker without duplicating audio:
    ``manifest.<speaker>.jsonl`` (always) and ``metadata.<speaker>.csv`` (when LJSpeech
    is requested), each listing only that speaker's clips.
    """
    by_speaker: dict[str, list[DatasetClip]] = {}
    for clip in clips:
        if clip.speaker is not None:
            by_speaker.setdefault(clip.speaker, []).append(clip)
    files: list[str] = []
    for speaker, spk_clips in sorted(by_speaker.items()):
        slug = _speaker_slug(speaker)
        files.append(write_jsonl_manifest(out_dir, spk_clips, name=f"manifest.{slug}.jsonl"))
        if "ljspeech" in formats:
            files.append(write_ljspeech(out_dir, spk_clips, name=f"metadata.{slug}.csv"))
    return files


def _size_category(n: int) -> str:
    """HuggingFace ``size_categories`` bucket for a clip count."""
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    return "n>1M"


def write_dataset_card(
    out_dir: Path,
    report: BuildReport,
    meta: SourceMetadata,
    *,
    version: str,
    generated_at: str,
    source_platform: str = "local",
) -> str:
    """Write ``dataset_card.md`` with HF YAML front matter + provenance/consent note."""
    retrieval_date = generated_at.split("T", 1)[0]
    oversized_note = (
        f" ({report.oversized_count} over the segment-length cap)" if report.oversized_count else ""
    )
    card = f"""---
license: unknown
language:
- {json.dumps(report.language, ensure_ascii=False)}
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: {json.dumps(meta.title, ensure_ascii=False)}
size_categories:
- {_size_category(report.clip_count)}
tags:
- hearsay
- speech
---

# {meta.title} — hearsay dataset

Generated by **hearsay {version}** on {retrieval_date}. This is a machine-built
TTS/STT training dataset: audio sliced into clips, each paired with its verbatim
transcript and timing. Indexed as {", ".join(report.formats)}.

## Dataset summary

- **Clips:** {report.clip_count}{oversized_note}
- **Total audio:** {format_timestamp(report.total_duration_s)} ({report.total_duration_s:.1f} s)
- **Language:** {report.language}
- **Audio:** mono 16-bit PCM WAV @ {report.sample_rate} Hz, under `wavs/`

## Source data

- **source_url:** {meta.source}
- **source_platform:** {source_platform}
- **channel_or_creator:** {meta.channel}
- **title:** {meta.title}
- **retrieval_date:** {retrieval_date}
- **generated_by:** hearsay {version}

## Personal and sensitive information

This audio may contain a real, identifiable person's voice. Voice and voiceprints
can be regulated as biometric or personal data (e.g. GDPR, U.S. state biometric
laws such as Illinois BIPA) and by voice/likeness laws (e.g. the Tennessee ELVIS
Act). **If you intend to clone or synthesize a voice, you are responsible for
obtaining that person's informed consent.** The EU AI Act requires synthetic
audio to be disclosed/labelled (from 2 Aug 2026).

- **consent_status:** unknown — user must verify
- **intended_use:** TTS/STT training (user-defined)

## Licensing information

- **source_license:** unknown — user must verify

hearsay is a local, user-side tool: it processed this media on your machine and
wrote these files locally. It does not host, distribute, or download content on
your behalf, and ships no datasets. **You are solely responsible** for ensuring you
have the rights to this audio and for complying with the source platform's terms
(extracting audio from services such as YouTube may violate their Terms of Service).
Audio and transcripts are typically copyrighted; building a personal/research
dataset differs from redistributing one, which generally needs the rights holder's
permission. *This note is informational, not legal advice.*
"""
    (out_dir / "dataset_card.md").write_text(card, encoding="utf-8")
    return "dataset_card.md"


def write_combined_card(
    out_dir: Path,
    report: CombinedReport,
    *,
    version: str,
    generated_at: str,
    source_platform: str,
) -> str:
    """Write ``dataset_card.md`` for a merged multi-source build, with a per-source table."""
    retrieval_date = generated_at.split("T", 1)[0]
    rows = "\n".join(
        f"| {json.dumps(s.label, ensure_ascii=False)} | `{s.source_id}` "
        f"| {s.clip_count} | {s.dropped} |"
        for s in report.sources
        if s.ok
    )
    failed = [s for s in report.sources if not s.ok]
    failed_rows = "\n".join(
        f"- {json.dumps(s.label, ensure_ascii=False)} — {s.error}" for s in failed
    )
    failed_section = (
        f"\n\n### Skipped sources ({report.failed})\n\n{failed_rows}\n" if failed_rows else ""
    )
    oversized_note = (
        f" ({report.oversized_count} over the segment-length cap)" if report.oversized_count else ""
    )
    dropped_note = f" — {report.drops_by_reason}" if report.drops_by_reason else ""
    card = f"""---
license: unknown
language:
- {json.dumps(report.language, ensure_ascii=False)}
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: {json.dumps(report.title, ensure_ascii=False)}
size_categories:
- {_size_category(report.clip_count)}
tags:
- hearsay
- speech
---

# {report.title} — hearsay dataset

Generated by **hearsay {version}** on {retrieval_date}. A machine-built TTS/STT
training dataset merged from **{report.succeeded} source(s)** ({report.failed} skipped).
Indexed as {", ".join(report.formats)}.

## Dataset summary

- **Clips:** {report.clip_count}{oversized_note}
- **Total audio:** {format_timestamp(report.total_duration_s)} ({report.total_duration_s:.1f} s)
- **Dropped by filters:** {report.dropped_count}{dropped_note}
- **Language:** {report.language}
- **Audio:** mono 16-bit PCM WAV @ {report.sample_rate} Hz, under `wavs/`

## Source data

- **source:** {report.source}
- **source_platform:** {source_platform}
- **retrieval_date:** {retrieval_date}
- **generated_by:** hearsay {version}

| Source | Clip id prefix | Clips | Dropped |
| --- | --- | --- | --- |
{rows}{failed_section}

## Personal and sensitive information

This audio may contain real, identifiable people's voices. Voice and voiceprints
can be regulated as biometric or personal data (e.g. GDPR, U.S. state biometric
laws such as Illinois BIPA) and by voice/likeness laws (e.g. the Tennessee ELVIS
Act). **If you intend to clone or synthesize a voice, you are responsible for
obtaining that person's informed consent.** The EU AI Act requires synthetic
audio to be disclosed/labelled (from 2 Aug 2026). Multi-source datasets typically
mix speakers — unsuitable for single-voice TTS without diarization (see hearsay's
`--diarize`).

- **consent_status:** unknown — user must verify
- **intended_use:** TTS/STT training (user-defined)

## Licensing information

- **source_license:** unknown — user must verify (each source may differ)

hearsay is a local, user-side tool: it processed this media on your machine and
wrote these files locally. It does not host, distribute, or download content on
your behalf, and ships no datasets. **You are solely responsible** for ensuring you
have the rights to this audio and for complying with each source platform's terms.
Audio and transcripts are typically copyrighted; building a personal/research
dataset differs from redistributing one, which generally needs the rights holder's
permission. *This note is informational, not legal advice.*
"""
    (out_dir / "dataset_card.md").write_text(card, encoding="utf-8")
    return "dataset_card.md"
