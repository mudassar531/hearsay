# IDEAS

Nice ideas that appeared mid-build, parked here per SPEC.md non-goals. None are scheduled.

## Still parked from SPEC Phase 6 (stretch — do not start without explicit approval)

- `--frames` keyframe extraction
- Web URL article fallback

## Moved into the v0.2 Dataset Export Mode plan (2026-06-14)

These were parked Phase-6 items; the dataset-export work gives them a home, so they
now live in `SPEC.md` ("v0.2 — Dataset Export Mode") / `PROGRESS.md` rather than here:

- **Speaker diarization** — moved to **Phase D5** as an optional `hearsay[diarize]`
  extra (pyannote.audio / whisperX), the key to single-voice TTS from multi-speaker
  audio. Design: `docs/dataset-mode-design.md` §5.
- **Vector-store / export helpers** — the "export helpers" idea is realized by v0.2's
  dataset exporters (LJSpeech `metadata.csv`, NeMo `manifest.jsonl`, optional HF
  `audiofolder`). A dedicated **vector-store/embeddings exporter** (chunks → a vector
  DB) remains a possible smaller future add-on, distinct from training-dataset export,
  but is no longer the headline parked item.

## From the real-world TTS audit (2026-06-15)

A 6-lens audit of a dataset built from a real (multi-speaker, music-heavy) MrBeast
video surfaced three gaps. The first — hard clip boundaries / no edge padding — is
**fixed** (edge padding + a de-click fade; `--pad`). The other two are parked here
because each needs real design, not a quick patch:

- **Optional denoise / source-separation extra (`hearsay[denoise]`).** Diarization
  isolates the right *voice* but cannot strip a background music/crowd bed mixed under
  it (the audit measured a ~-22.8 dBFS noise floor with energy in the 60–250 Hz
  music-bass band). A pre-slice speech-enhancement / source-separation pass (e.g.
  Demucs or a denoiser) would lift in-the-wild audio toward clean TTS quality. Heavy
  (torch) → must be an opt-in extra with its own model download, like `[diarize]`.
- **Transcript-fidelity validation.** Quality filters check char-rate/duration, not
  *correctness*, so a garbled-but-plausible-rate ASR line can survive (the audit found
  one: "I would ben ab'tle to do do the good I."). A real fix is a cross-check pass
  (re-transcribe with a larger reference model, or a second engine, and flag low
  agreement) — deliberately **not** a brittle string heuristic, which would wrongly
  drop verbatim disfluencies ("And I I couldn't") and break the verbatim guarantee.
  **Partly shipped in 0.8.0 as `hearsay verify`** — a sample-based, post-build pairing
  and accuracy report. Still parked: a per-clip cross-check at *build* time that drops
  low-agreement clips (2× transcription cost; needs a reference-model policy).
