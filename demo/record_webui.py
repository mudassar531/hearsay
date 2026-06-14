#!/usr/bin/env python3
"""Record a GIF of the hearsay web UI in action.

Drives the local ``hearsay web`` server with Playwright (Chromium), records the
session to webm, which ``demo/make_webui_gif.sh`` then turns into an optimised
GIF. Reproduces ``demo/webui.gif``.

Setup::

    uv pip install playwright && uv run playwright install chromium
    hearsay web --port 8765 &        # start the UI in another shell

Run::

    python demo/record_webui.py http://127.0.0.1:8765/ \
        "https://www.youtube.com/watch?v=rStL7niR7gs" /tmp/webui-rec

Prints the path to the recorded .webm on stdout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
VIDEO_URL = sys.argv[2] if len(sys.argv) > 2 else "https://www.youtube.com/watch?v=rStL7niR7gs"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/webui-rec")
OUT.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1280, 820


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        ctx = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,
            record_video_dir=str(OUT),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = ctx.new_page()
        page.goto(PAGE_URL, wait_until="networkidle")
        time.sleep(1.3)  # let the hero render

        # Type a YouTube URL the way a person would, then send.
        page.locator("#input").click()
        page.locator("#input").press_sequentially(VIDEO_URL, delay=42)
        time.sleep(0.7)
        page.locator("#send").click()

        # Wait until the transcript markdown is actually rendered.
        page.wait_for_function(
            "() => { const m = document.querySelector('.md');"
            " return m && m.innerText.trim().length > 80; }",
            timeout=90_000,
        )
        time.sleep(1.6)  # let the reader take in the result

        # Scroll through the transcript so the timestamps/paragraphs show.
        page.mouse.wheel(0, 680)
        time.sleep(1.5)
        page.mouse.wheel(0, 680)
        time.sleep(1.7)

        ctx.close()  # finalises the webm
        browser.close()

    videos = sorted(OUT.glob("*.webm"))
    print(str(videos[-1]) if videos else "NO VIDEO PRODUCED")


if __name__ == "__main__":
    main()
