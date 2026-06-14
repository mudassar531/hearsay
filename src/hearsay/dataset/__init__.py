"""Dataset-export mode: turn media into TTS/STT training datasets.

A second, additive output mode (alongside the markdown/JSON engine): word-level
transcript timings are segmented into short training clips, sliced from the
audio, and written in standard layouts (LJSpeech ``metadata.csv`` + ``wavs/``,
NeMo-style JSONL manifest). See ``docs/dataset-mode-design.md``.

This package is imported only by the dataset code path; the captions/markdown
pipeline never pays its (light, stdlib + pydantic) import cost.
"""
