#!/usr/bin/env bash
# Record the hearsay dataset-mode demo gif (demo/dataset.gif) with vhs.
#
# Requirements: vhs (https://github.com/charmbracelet/vhs — `brew install vhs`),
# ffmpeg, and `hearsay` available (run from the repo; this script puts the repo's
# `uv run` hearsay on PATH for you). Runs fully offline against the bundled test
# fixture, so it needs a cached Whisper model but no network.
#
# Usage: ./demo/record_dataset.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v vhs >/dev/null 2>&1; then
  echo "vhs is not installed. Install it with: brew install vhs" >&2
  echo "(or see https://github.com/charmbracelet/vhs for other platforms)" >&2
  exit 1
fi

# Record in an isolated temp dir so the demo's output files don't litter the repo.
HEARSAY_DEMO_DIR="$(mktemp -d)"
export HEARSAY_DEMO_DIR
trap 'rm -rf "$HEARSAY_DEMO_DIR"' EXIT

# A nicely-named, license-clean input: the bundled ~5s synthetic fixture, upsampled
# to 22050 Hz so the demo command needs no extra flags (no upsampling notice).
ffmpeg -hide_banner -loglevel error -y \
  -i tests/fixtures/sample.wav -ar 22050 "$HEARSAY_DEMO_DIR/interview.wav"

# Put the repo's hearsay on PATH and stay offline (cached model, no HF Hub notice).
export PATH="$(dirname "$(uv run which hearsay)"):$PATH"
export HF_HUB_OFFLINE=1

echo "Recording demo/dataset.gif (offline, uses a cached Whisper model)…"
vhs demo/dataset.tape
echo "Wrote demo/dataset.gif"
