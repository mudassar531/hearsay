#!/usr/bin/env bash
# Record the hearsay demo gif (demo/demo.gif) with vhs.
#
# Requirements: vhs (https://github.com/charmbracelet/vhs — `brew install vhs`),
# ffmpeg, and `hearsay` on PATH (`uv tool install hearsay`, or run from the repo
# with `uv run` after `export PATH="$(uv run which hearsay | xargs dirname):$PATH"`).
# Needs network access (it ingests a real YouTube video's captions).
#
# Usage: ./demo/record.sh
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

echo "Recording demo/demo.gif (this ingests a real video, ~10s of network)…"
vhs demo/hearsay.tape
echo "Wrote demo/demo.gif"
