#!/usr/bin/env bash
# Convert the webm recorded by demo/record_webui.py into an optimised GIF
# (two-pass palette for clean text). Reproduces demo/webui.gif.
#
# Usage: demo/make_webui_gif.sh /tmp/webui-rec/<recording>.webm demo/webui.gif
set -euo pipefail

SRC="${1:?usage: make_webui_gif.sh <input.webm> [output.gif]}"
OUT="${2:-demo/webui.gif}"
FPS="${FPS:-13}"
WIDTH="${WIDTH:-1040}"

PALETTE="$(mktemp -t hearsay-palette).png"
trap 'rm -f "$PALETTE"' EXIT

ffmpeg -y -i "$SRC" \
  -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff" "$PALETTE"
ffmpeg -y -i "$SRC" -i "$PALETTE" \
  -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
  "$OUT"

ls -lh "$OUT"
