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
