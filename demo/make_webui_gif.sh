#!/usr/bin/env bash
# Convert the webm recorded by demo/record_webui.py into a small, GitHub-friendly
# GIF: ffmpeg two-pass palette (clean text) + an optional gifsicle lossy pass.
# Reproduces demo/webui.gif (kept well under ~1 MB so GitHub renders it inline).
#
# Usage: demo/make_webui_gif.sh /tmp/webui-rec/<recording>.webm demo/webui.gif
set -euo pipefail

SRC="${1:?usage: make_webui_gif.sh <input.webm> [output.gif]}"
OUT="${2:-demo/webui.gif}"
FPS="${FPS:-10}"
WIDTH="${WIDTH:-860}"
LOSSY="${LOSSY:-60}"
COLORS="${COLORS:-160}"

PALETTE="$(mktemp -t hearsay-palette).png"
trap 'rm -f "$PALETTE"' EXIT

ffmpeg -y -i "$SRC" \
  -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=max_colors=${COLORS}:stats_mode=diff" "$PALETTE"
ffmpeg -y -i "$SRC" -i "$PALETTE" \
  -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" \
  "$OUT"

# Optional lossy LZW optimisation — the big size win for text-heavy captures.
if command -v gifsicle >/dev/null 2>&1; then
  gifsicle -O3 --lossy="${LOSSY}" --colors "${COLORS}" "$OUT" -o "$OUT"
fi

ls -lh "$OUT"
