"""A polished, dependency-free web UI for hearsay.

One static HTML page (embedded below) talks to two JSON endpoints that wrap the
existing pipeline. The design is Swiss/International style: white, Helvetica, a
strict grid, hairline rules, generous whitespace, monochrome with one restrained
accent. A sidebar holds history, the top bar holds the model selector, and a
composer takes a YouTube URL or an uploaded audio/video file, returning clean
timestamped markdown with a live preview.

Built on the standard-library HTTP server so it adds no runtime dependencies.
File uploads are sent as the raw request body (filename in a query param),
sidestepping multipart parsing (and ``cgi``, which is gone in Python 3.13).
History lives entirely in the browser (localStorage); the server is stateless.
"""

import base64
import io
import json
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hearsay.errors import HearsayError, InvalidSourceError
from hearsay.models import Document
from hearsay.pipeline import (
    NoCaptionsError,
    ingest_file,
    ingest_youtube,
    ingest_youtube_transcribe,
)
from hearsay.render import render_markdown
from hearsay.timefmt import format_timestamp
from hearsay.transcribe import DEFAULT_MODEL
from hearsay.youtube import extract_playlist_id, extract_video_id

# Upload ceiling so a stray huge request can't exhaust memory (raw body is read
# fully into RAM). 1 GiB comfortably covers multi-hour audio.
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


# --- Pipeline glue (network/transcription happens downstream) --------------


def process_url(
    url: str,
    *,
    transcribe: bool = False,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    vad: bool = True,
) -> Document:
    """Ingest a single YouTube video URL (captions, or whisper/parakeet).

    Mirrors the CLI's single-video behaviour: captions first, falling back to
    local transcription when there are none (or when ``transcribe`` forces it).
    """
    url = url.strip()
    if not url:
        raise InvalidSourceError("No URL provided.", hint="Paste a YouTube video link.")
    if extract_playlist_id(url) is not None:
        raise InvalidSourceError(
            "Playlists and podcast feeds aren't supported in the web UI yet.",
            hint="Use the CLI for batch sources, e.g. hearsay <url> --all.",
        )
    if extract_video_id(url) is None:
        raise InvalidSourceError(
            f"Not a recognized YouTube video URL: {url}",
            hint="Paste a single YouTube video link, or upload a file instead.",
        )
    if transcribe:
        return ingest_youtube_transcribe(url, model_size=model, language=language, vad_filter=vad)
    try:
        return ingest_youtube(url, language=language or "en")
    except NoCaptionsError:
        return ingest_youtube_transcribe(url, model_size=model, language=language, vad_filter=vad)


def process_file(
    name: str,
    data: bytes,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    vad: bool = True,
) -> Document:
    """Transcribe uploaded file bytes, preserving the original name for the title."""
    safe_name = Path(name).name or "upload"
    with tempfile.TemporaryDirectory(prefix="hearsay-web-") as tmp:
        path = Path(tmp) / safe_name
        path.write_bytes(data)
        return ingest_file(path, model_size=model, language=language, vad_filter=vad)


def _document_payload(doc: Document) -> dict:
    """Shape a Document into the JSON the page renders."""
    return {
        "ok": True,
        "markdown": render_markdown(doc),
        "title": doc.meta.title,
        "method": doc.method,
        "language": doc.language,
        "duration": format_timestamp(doc.meta.duration_s),
        "sections": len(doc.sections),
    }


# --- Dataset mode (single source -> a downloadable zip) --------------------

_DEFAULT_DATASET_FORMATS = ["ljspeech", "jsonl"]


def _zip_dir(root: Path) -> bytes:
    """Zip a directory tree into in-memory bytes (relative paths)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    return buf.getvalue()


def _dataset_payload(out_dir: Path, report: object) -> dict:
    """Shape a build report + the zipped dataset into the JSON the page downloads."""
    return {
        "ok": True,
        "dataset": True,
        "clips": report.clip_count,  # type: ignore[attr-defined]
        "duration": format_timestamp(report.total_duration_s),  # type: ignore[attr-defined]
        "dropped": report.dropped_count,  # type: ignore[attr-defined]
        "warnings": report.warnings,  # type: ignore[attr-defined]
        "zip_name": f"{out_dir.name}.zip",
        "zip_b64": base64.b64encode(_zip_dir(out_dir)).decode("ascii"),
    }


def _dataset_config(out_dir: Path, opts: dict):
    from hearsay.dataset.models import DatasetConfig

    formats = opts.get("formats") or _DEFAULT_DATASET_FORMATS
    try:
        return DatasetConfig(
            out_dir=out_dir,
            formats=list(formats),
            sample_rate=int(opts.get("sample_rate") or 22050),
            segment_min_s=float(opts.get("segment_min") or 1.0),
            segment_max_s=float(opts.get("segment_max") or 15.0),
        )
    except (ValueError, TypeError) as exc:  # bad numeric option -> a 400, not an opaque 500
        raise InvalidSourceError(
            f"Invalid dataset option: {exc}",
            hint="sample-rate must be a positive integer and segment bounds positive numbers.",
        ) from exc


def build_url_dataset(url: str, *, opts: dict) -> dict:
    """Build a dataset from one YouTube video URL and return a zip payload."""
    from hearsay.dataset.build import build_dataset_from_youtube

    url = url.strip()
    if not url:
        raise InvalidSourceError("No URL provided.", hint="Paste a YouTube video link.")
    if extract_playlist_id(url) is not None:
        raise InvalidSourceError(
            "Playlists aren't supported in the web UI's dataset mode.",
            hint="Use the CLI for batch sources, e.g. hearsay dataset <playlist-url>.",
        )
    if extract_video_id(url) is None:
        raise InvalidSourceError(
            f"Not a recognized YouTube video URL: {url}",
            hint="Paste a single YouTube video link, or upload a file instead.",
        )
    with tempfile.TemporaryDirectory(prefix="hearsay-web-ds-") as tmp:
        out_dir = Path(tmp) / "hearsay-dataset"
        report = build_dataset_from_youtube(
            url,
            config=_dataset_config(out_dir, opts),
            model_size=str(opts.get("model") or DEFAULT_MODEL),
            language=(opts.get("lang") or None),
            vad_filter=bool(opts.get("vad", True)),
        )
        return _dataset_payload(out_dir, report)


def build_file_dataset(name: str, data: bytes, *, opts: dict) -> dict:
    """Build a dataset from an uploaded file and return a zip payload."""
    from hearsay.dataset.build import build_dataset_from_file

    safe_name = Path(name).name or "upload"
    with tempfile.TemporaryDirectory(prefix="hearsay-web-ds-") as tmp:
        source = Path(tmp) / safe_name
        source.write_bytes(data)
        out_dir = Path(tmp) / "hearsay-dataset"
        report = build_dataset_from_file(
            source,
            config=_dataset_config(out_dir, opts),
            model_size=str(opts.get("model") or DEFAULT_MODEL),
            language=(opts.get("lang") or None),
            vad_filter=bool(opts.get("vad", True)),
        )
        return _dataset_payload(out_dir, report)


# --- HTTP server ----------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "hearsay-webui"

    def log_message(self, *args: object) -> None:
        pass  # quiet by default — no request logging to stderr

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_html(_PAGE)
        elif path == "/health":
            self._send_json({"ok": True})
        else:
            self._send_json({"ok": False, "error": "Not found."}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/url":
                payload = _document_payload(self._handle_url())
            elif parsed.path == "/api/file":
                payload = _document_payload(self._handle_file(parse_qs(parsed.query)))
            elif parsed.path == "/api/dataset":
                payload = self._handle_dataset_url()
            elif parsed.path == "/api/dataset-file":
                payload = self._handle_dataset_file(parse_qs(parsed.query))
            else:
                self._send_json({"ok": False, "error": "Not found."}, status=404)
                return
        except HearsayError as exc:
            self._send_json({"ok": False, "error": exc.message, "hint": exc.hint}, status=400)
        except Exception as exc:  # surface anything unexpected as a 500
            self._send_json({"ok": False, "error": str(exc)}, status=500)
        else:
            self._send_json(payload)

    # -- request helpers --

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_UPLOAD_BYTES:
            raise InvalidSourceError(
                "Upload is too large.",
                hint="The web UI accepts files up to 1 GiB; use the CLI for bigger ones.",
            )
        return self.rfile.read(length) if length else b""

    def _handle_url(self) -> Document:
        body = json.loads(self._read_body() or b"{}")
        return process_url(
            str(body.get("url", "")),
            transcribe=bool(body.get("transcribe", False)),
            model=str(body.get("model") or DEFAULT_MODEL),
            language=(body.get("lang") or None),
            vad=bool(body.get("vad", True)),
        )

    def _handle_file(self, query: dict[str, list[str]]) -> Document:
        data = self._read_body()
        if not data:
            raise InvalidSourceError("No file received.", hint="Choose an audio/video file.")
        name = (query.get("name") or ["upload"])[0]
        model = (query.get("model") or [DEFAULT_MODEL])[0]
        lang = (query.get("lang") or [""])[0] or None
        vad = (query.get("vad") or ["1"])[0] != "0"
        return process_file(name, data, model=model, language=lang, vad=vad)

    def _handle_dataset_url(self) -> dict:
        body = json.loads(self._read_body() or b"{}")
        return build_url_dataset(str(body.get("url", "")), opts=body)

    def _handle_dataset_file(self, query: dict[str, list[str]]) -> dict:
        data = self._read_body()
        if not data:
            raise InvalidSourceError("No file received.", hint="Choose an audio/video file.")
        name = (query.get("name") or ["upload"])[0]
        opts: dict = {
            "model": (query.get("model") or [DEFAULT_MODEL])[0],
            "lang": (query.get("lang") or [""])[0] or None,
            "vad": (query.get("vad") or ["1"])[0] != "0",
            "sample_rate": (query.get("sample_rate") or ["22050"])[0],
            "segment_min": (query.get("segment_min") or ["1.0"])[0],
            "segment_max": (query.get("segment_max") or ["15.0"])[0],
        }
        return build_file_dataset(name, data, opts=opts)

    # -- response helpers --

    def _send_json(self, payload: dict, *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create (but do not start) the threaded hearsay web server."""
    return ThreadingHTTPServer((host, port), _Handler)


def run_server(host: str = "127.0.0.1", port: int = 8756) -> None:
    """Serve the web UI until interrupted."""
    httpd = make_server(host, port)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"hearsay web UI → http://{shown}:{port}  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


# --- The page (single file: HTML + CSS + JS, no external assets) -----------

_PAGE = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hearsay</title>
<style>
  :root{
    --bg:#ffffff; --panel:#ffffff; --fg:#111111; --fg2:#3a3a3a; --muted:#9a9a9a;
    --line:#e7e7e7; --line2:#111111; --hover:#f4f4f4; --accent:#e3000f;
    --maxw:760px;
  }
  html[data-theme="dark"]{
    --bg:#0c0c0c; --panel:#0c0c0c; --fg:#f1f1f1; --fg2:#cfcfcf; --muted:#7e7e7e;
    --line:#242424; --line2:#f1f1f1; --hover:#161616; --accent:#ff3b30;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.55 "Helvetica Neue",Helvetica,Arial,"Inter",system-ui,sans-serif;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  button{font:inherit;cursor:pointer;color:inherit;border:0;background:none}
  svg{display:block}
  .lbl{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:500}
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:var(--line);border:3px solid var(--bg);background-clip:padding-box;border-radius:0}
  ::selection{background:var(--fg);color:var(--bg)}

  .app{display:flex;height:100vh;overflow:hidden}

  /* sidebar */
  .sidebar{width:248px;flex:0 0 248px;display:flex;flex-direction:column;background:var(--panel);
    border-right:1px solid var(--line);transition:margin-left .18s ease}
  .app.collapsed .sidebar{margin-left:-249px}
  .side-top{padding:20px 18px 14px;display:flex;flex-direction:column;gap:18px}
  .brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:17px;letter-spacing:-.02em}
  .brand .mark{width:13px;height:13px;background:var(--accent)}
  .new-btn{display:flex;align-items:center;gap:9px;padding:9px 11px;border:1px solid var(--line2);
    font-size:12px;letter-spacing:.04em;font-weight:500;transition:.14s}
  .new-btn:hover{background:var(--fg);color:var(--bg)}
  .new-btn svg{width:13px;height:13px}
  .hist{flex:1;overflow:auto;padding:6px 10px 12px}
  .hist .lbl{padding:14px 8px 8px;display:block}
  .hist-item{position:relative;padding:10px 26px 10px 9px;cursor:pointer;border-top:1px solid var(--line)}
  .hist-item:hover{background:var(--hover)}
  .hist-item.active{box-shadow:inset 2px 0 0 var(--accent)}
  .hist-item .t{font-size:13px;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .hist-item .s{font-size:11px;color:var(--muted);margin-top:1px}
  .hist-item .del{position:absolute;right:6px;top:50%;transform:translateY(-50%);opacity:0;color:var(--muted);
    font-size:16px;line-height:1;padding:3px 5px;transition:.12s}
  .hist-item:hover .del{opacity:1}
  .hist-item .del:hover{color:var(--accent)}
  .hist-empty{color:var(--muted);font-size:12.5px;padding:14px 8px;line-height:1.6}
  .side-foot{padding:13px 18px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);
    display:flex;justify-content:space-between;letter-spacing:.04em}
  .side-foot a{color:var(--muted);text-decoration:none}
  .side-foot a:hover{color:var(--fg)}

  /* main */
  .main{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
  .topbar{height:58px;flex:0 0 58px;display:flex;align-items:center;gap:14px;padding:0 20px;
    border-bottom:1px solid var(--line)}
  .icon-btn{width:32px;height:32px;display:grid;place-items:center;color:var(--fg2)}
  .icon-btn:hover{color:var(--fg)}
  .icon-btn svg{width:18px;height:18px}
  .top-title{flex:1;min-width:0;font-size:13px;font-weight:500;letter-spacing:.02em;color:var(--fg);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .model{display:flex;align-items:center;gap:8px}
  .model .lbl{display:none}
  .model select{border:1px solid var(--line);background:var(--bg);color:var(--fg);font:inherit;font-size:12.5px;
    padding:7px 26px 7px 11px;outline:none;border-radius:0;cursor:pointer;
    -webkit-appearance:none;appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,var(--fg2) 50%),linear-gradient(135deg,var(--fg2) 50%,transparent 50%);
    background-position:calc(100% - 14px) 15px,calc(100% - 9px) 15px;background-size:5px 5px,5px 5px;background-repeat:no-repeat}
  .model select:hover{border-color:var(--fg2)}

  .scroll{flex:1;overflow:auto}
  .stream{max-width:var(--maxw);margin:0 auto;padding:34px 28px 160px}

  /* hero */
  .hero{position:absolute;inset:58px 0 0;display:flex;flex-direction:column;align-items:center;justify-content:center;
    text-align:center;padding:24px}
  .hero .mark{width:30px;height:30px;background:var(--accent);margin-bottom:26px}
  .hero .eyebrow{margin-bottom:14px}
  .hero h1{margin:0 0 14px;font-size:46px;line-height:1;font-weight:700;letter-spacing:-.035em}
  .hero p{margin:0 0 30px;color:var(--muted);max-width:380px;font-size:14.5px}
  .chips{display:flex;gap:0;border:1px solid var(--line2)}
  .chip{padding:11px 18px;font-size:12.5px;color:var(--fg);transition:.14s;letter-spacing:.02em}
  .chip+.chip{border-left:1px solid var(--line2)}
  .chip:hover{background:var(--fg);color:var(--bg)}

  /* messages */
  .msg{display:grid;grid-template-columns:64px 1fr;gap:0;padding:26px 0;border-top:1px solid var(--line)}
  .msg:first-child{border-top:0}
  .msg .role{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);padding-top:3px}
  .msg.bot .role{color:var(--accent)}
  .msg .body{min-width:0}
  .msg.user .src{font-size:14px;color:var(--fg);word-break:break-all}
  .doc{border:1px solid var(--line)}
  .doc-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:11px 14px;border-bottom:1px solid var(--line)}
  .meta{display:flex;flex-wrap:wrap;gap:5px 16px;font-size:11.5px;color:var(--muted);flex:1;min-width:0;align-items:center}
  .meta b{color:var(--fg2);font-weight:600}
  .badge{font-size:10.5px;letter-spacing:.04em;padding:3px 7px;border:1px solid var(--line2);color:var(--fg);white-space:nowrap}
  .seg{display:flex;border:1px solid var(--line)}
  .seg button{padding:5px 12px;color:var(--muted);font-size:11.5px;letter-spacing:.03em}
  .seg button+button{border-left:1px solid var(--line)}
  .seg button.on{background:var(--fg);color:var(--bg)}
  .doc-actions{display:flex;gap:0;border:1px solid var(--line)}
  .doc-actions button{padding:5px 12px;color:var(--fg2);font-size:11.5px;letter-spacing:.03em}
  .doc-actions button+button{border-left:1px solid var(--line)}
  .doc-actions button:hover{background:var(--fg);color:var(--bg)}
  .doc-body{padding:20px 22px;max-height:58vh;overflow:auto}
  .raw{white-space:pre-wrap;font:12.5px/1.7 "SF Mono",ui-monospace,Menlo,Consolas,monospace;color:var(--fg2)}
  .md h1{font-size:21px;margin:.1em 0 .6em;font-weight:700;letter-spacing:-.02em}
  .md h2{font-size:13px;margin:1.8em 0 .7em;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
    padding-top:.7em;border-top:1px solid var(--line)}
  .md p{margin:.6em 0;color:var(--fg2);font-size:14px}
  .md hr{border:0;border-top:1px solid var(--line);margin:1.4em 0}
  .md .ts{color:var(--muted);font-weight:600;font-variant-numeric:tabular-nums}
  .md .fm{color:var(--muted);font-size:11.5px;border-left:2px solid var(--line);padding-left:12px;margin-bottom:18px;
    white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;line-height:1.7}
  .think{display:flex;align-items:center;gap:11px;color:var(--muted);font-size:13px;padding:2px 0}
  .spin{width:13px;height:13px;border:1.5px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .err{color:var(--accent);font-size:13.5px;white-space:pre-wrap;border-left:2px solid var(--accent);padding-left:12px}

  /* composer */
  .composer-wrap{position:absolute;left:0;right:0;bottom:0;padding:0 28px 26px;
    background:linear-gradient(to top,var(--bg) 62%,transparent)}
  .composer{max-width:var(--maxw);margin:0 auto;border:1px solid var(--line2);background:var(--bg)}
  .composer.focus{box-shadow:0 0 0 1px var(--line2)}
  .composer.drag{background:var(--hover)}
  .file-chip{display:none;align-items:center;gap:10px;padding:9px 12px;border-bottom:1px solid var(--line);
    font-size:12.5px;color:var(--fg2)}
  .file-chip.show{display:flex}
  .file-chip .x{color:var(--muted);font-size:15px;line-height:1}
  .file-chip .x:hover{color:var(--accent)}
  .crow{display:flex;align-items:flex-end;gap:4px;padding:6px 6px 6px 10px}
  .attach{width:34px;height:34px;flex:0 0 34px;display:grid;place-items:center;color:var(--fg2)}
  .attach:hover{color:var(--fg)}
  .attach svg{width:17px;height:17px}
  .composer textarea{flex:1;border:0;background:none;color:var(--fg);resize:none;outline:none;font:inherit;font-size:14px;
    max-height:168px;padding:9px 4px;line-height:1.5}
  .composer textarea::placeholder{color:var(--muted)}
  .send{width:36px;height:36px;flex:0 0 36px;display:grid;place-items:center;background:var(--fg);color:var(--bg);transition:.12s}
  .send svg{width:17px;height:17px}
  .send:disabled{background:var(--line);color:var(--muted)}
  .send:not(:disabled):hover{background:var(--accent)}
  .copts{display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:9px 12px;border-top:1px solid var(--line);
    color:var(--muted);font-size:11.5px;letter-spacing:.02em}
  .copts label{display:flex;align-items:center;gap:7px;cursor:pointer;text-transform:uppercase;letter-spacing:.06em;font-size:10.5px}
  .copts input[type=text]{width:96px;background:var(--bg);border:1px solid var(--line);color:var(--fg);padding:4px 8px;
    font:inherit;font-size:12px;outline:none;border-radius:0;text-transform:none;letter-spacing:0}
  .copts input[type=text]:focus{border-color:var(--fg2)}
  .copts input[type=checkbox]{accent-color:var(--accent)}
  .hint{text-align:center;color:var(--muted);font-size:10.5px;letter-spacing:.04em;margin-top:11px}
  @media(max-width:720px){
    .sidebar{position:absolute;z-index:5;height:100%}
    .app.collapsed .sidebar{margin-left:-260px}
    .app:not(.collapsed) .main{filter:brightness(.45)}
    .msg{grid-template-columns:54px 1fr}
  }
</style>
</head>
<body>
<div class="app" id="app">
  <aside class="sidebar">
    <div class="side-top">
      <div class="brand"><span class="mark"></span>hearsay</div>
      <button class="new-btn" id="new">
        <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M7 1.5v11M1.5 7h11"/></svg>
        NEW TRANSCRIPT</button>
    </div>
    <div class="hist" id="hist"></div>
    <div class="side-foot"><span>LOCAL · PRIVATE</span>
      <a href="https://github.com/mudassar531/hearsay" target="_blank" rel="noopener">GITHUB</a></div>
  </aside>

  <div class="main">
    <div class="topbar">
      <button class="icon-btn" id="toggle" title="Toggle sidebar">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="2.5" y="3.5" width="13" height="11"/><line x1="7" y1="3.5" x2="7" y2="14.5"/></svg>
      </button>
      <div class="top-title" id="topTitle">New transcript</div>
      <div class="model" title="Transcription model">
        <span class="lbl">Model</span>
        <select id="model">
          <option value="auto">auto</option>
          <option value="parakeet">parakeet</option>
          <option value="parakeet-en">parakeet-en</option>
          <option value="tiny">whisper tiny</option>
          <option value="base">whisper base</option>
          <option value="small">whisper small</option>
          <option value="medium">whisper medium</option>
          <option value="large-v3">whisper large-v3</option>
        </select>
      </div>
      <button class="icon-btn" id="theme" title="Toggle theme">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="9" cy="9" r="6.5"/><path d="M9 2.5a6.5 6.5 0 0 1 0 13z" fill="currentColor" stroke="none"/></svg>
      </button>
    </div>

    <div class="scroll" id="scroll">
      <div class="stream" id="stream"></div>
    </div>

    <div class="composer-wrap">
      <div class="composer" id="composer">
        <div class="file-chip" id="fileChip"><span id="fileName"></span><button class="x" id="fileX">✕</button></div>
        <div class="crow">
          <button class="attach" id="attach" title="Attach audio/video file">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M14 8.4l-5.3 5.3a3 3 0 0 1-4.3-4.3l6.1-6.1a2 2 0 0 1 2.9 2.9l-5.7 5.7a1 1 0 0 1-1.5-1.5l5.1-5.1"/></svg>
          </button>
          <textarea id="input" rows="1" placeholder="Paste a YouTube URL, or attach a file…"></textarea>
          <button class="send" id="send" title="Ingest">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 14.5V4M4.5 8.5 9 3.8l4.5 4.7"/></svg>
          </button>
        </div>
        <div class="copts">
          <label id="wrapTr"><input type="checkbox" id="transcribe"> Force transcription</label>
          <label><input type="checkbox" id="vad" checked> VAD</label>
          <label style="text-transform:none;letter-spacing:0">Lang <input type="text" id="lang" placeholder="auto"></label>
          <label title="Build a TTS/STT training dataset (downloads a .zip)"><input type="checkbox" id="dataset"> Dataset</label>
        </div>
        <div class="copts" id="dsOpts" style="display:none">
          <label style="text-transform:none;letter-spacing:0">Seg s <input type="text" id="segMin" value="1" style="width:42px"> – <input type="text" id="segMax" value="15" style="width:42px"></label>
          <label style="text-transform:none;letter-spacing:0">Rate <input type="text" id="sr" value="22050" style="width:64px"></label>
          <span style="color:var(--muted);font-size:10px">Playlists/feeds &amp; long sources → use the CLI</span>
        </div>
      </div>
      <div class="hint">LOCAL TRANSCRIPTION DOWNLOADS THE MODEL ONCE, THEN RUNS OFFLINE</div>
    </div>
    <input type="file" id="file" hidden accept="audio/*,video/*,.mp3,.m4a,.wav,.mp4,.webm,.mkv,.mov,.flac,.ogg,.opus,.aac">
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const stream = $('#stream'), scroll = $('#scroll');
let pickedFile = null, current = null;

/* ---- theme + sidebar ---- */
const root = document.documentElement;
root.dataset.theme = localStorage.getItem('hs-theme') || 'light';
$('#theme').onclick = () => { root.dataset.theme = root.dataset.theme==='light'?'dark':'light';
  localStorage.setItem('hs-theme', root.dataset.theme); };
$('#toggle').onclick = () => $('#app').classList.toggle('collapsed');

/* ---- model persistence ---- */
$('#model').value = localStorage.getItem('hs-model') || 'auto';
$('#model').onchange = e => localStorage.setItem('hs-model', e.target.value);

/* ---- history (localStorage) ---- */
const HKEY = 'hs-history';
const load = () => { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } };
const save = h => localStorage.setItem(HKEY, JSON.stringify(h.slice(0,100)));
function renderHist() {
  const h = load(); const box = $('#hist');
  if (!h.length) { box.innerHTML = '<div class="hist-empty">No transcripts yet.</div>'; return; }
  box.innerHTML = '<span class="lbl">Recent</span>';
  h.forEach(item => {
    const el = document.createElement('div');
    el.className = 'hist-item' + (current===item.id?' active':'');
    el.innerHTML = `<div class="t"></div><div class="s">${esc(item.method||'')} · ${esc(item.duration||'')}</div><button class="del" title="Delete">✕</button>`;
    el.querySelector('.t').textContent = item.title || 'Untitled';
    el.onclick = e => { if (e.target.classList.contains('del')) return; openItem(item); };
    el.querySelector('.del').onclick = e => { e.stopPropagation();
      save(load().filter(x=>x.id!==item.id)); if(current===item.id) newChat(); renderHist(); };
    box.appendChild(el);
  });
}

/* ---- markdown render (safe, minimal) ---- */
function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function renderMd(md){
  let html='',i=0; const lines=md.split('\n');
  if (lines[0]==='---'){ let j=1; const fm=[]; while(j<lines.length&&lines[j]!=='---')fm.push(lines[j++]);
    if(j<lines.length){ html+='<div class="fm">'+esc(fm.join('\n'))+'</div>'; i=j+1; } }
  let para=[]; const flush=()=>{ if(para.length){ html+='<p>'+inl(para.join(' '))+'</p>'; para=[]; } };
  const inl=s=>esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')
    .replace(/(\[\d\d:\d\d:\d\d(?:\s*[–-]\s*\d\d:\d\d:\d\d)?\])/g,'<span class="ts">$1</span>');
  for(;i<lines.length;i++){ const ln=lines[i];
    if(ln.startsWith('## ')){flush();html+='<h2>'+inl(ln.slice(3))+'</h2>';}
    else if(ln.startsWith('# ')){flush();html+='<h1>'+inl(ln.slice(2))+'</h1>';}
    else if(ln.trim()==='---'){flush();html+='<hr>';}
    else if(ln.trim()===''){flush();}
    else para.push(ln); }
  flush(); return html;
}

/* ---- view building ---- */
function clearStream(){ stream.innerHTML=''; }
function showHero(){
  current=null; $('#topTitle').textContent='New transcript'; renderHist();
  clearStream();
  const h=document.createElement('div'); h.className='hero';
  h.innerHTML=`<div class="mark"></div><div class="eyebrow lbl">Video &amp; audio → markdown</div>
    <h1>hearsay</h1>
    <p>Turn any YouTube video, podcast, or recording into clean, timestamped, LLM-ready markdown.</p>
    <div class="chips">
      <button class="chip" data-ex="https://www.youtube.com/watch?v=9GLYsrMpprs">Try a YouTube video</button>
      <button class="chip" data-attach>Upload an audio file</button>
    </div>`;
  stream.appendChild(h);
  h.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{ if(c.dataset.attach)$('#file').click();
    else{ $('#input').value=c.dataset.ex; autoGrow(); $('#input').focus(); } });
}
function msgShell(role, isBot){
  const m=document.createElement('div'); m.className='msg '+(isBot?'bot':'user');
  m.innerHTML=`<div class="role">${role}</div><div class="body"></div>`;
  return m;
}
function userMsg(text){
  const m=msgShell('You',false);
  const s=document.createElement('div'); s.className='src'; s.textContent=text;
  m.querySelector('.body').appendChild(s); stream.appendChild(m); return m;
}
function botThinking(){
  const m=msgShell('hearsay',true);
  m.querySelector('.body').innerHTML=`<div class="think"><div class="spin"></div>
    <span>Transcribing… this can take a while for long media.</span></div>`;
  stream.appendChild(m); toBottom(); return m;
}
function docNode(d){
  const wrap=document.createElement('div'); wrap.className='doc';
  wrap.innerHTML=`
    <div class="doc-head">
      <div class="meta">
        <span class="badge">${esc(d.method)}</span>
        <span>lang <b>${esc(d.language)}</b></span>
        <span>duration <b>${esc(d.duration)}</b></span>
        <span><b>${d.sections}</b> sections</span>
      </div>
      <div class="seg"><button class="on" data-v="md">Preview</button><button data-v="raw">Markdown</button></div>
      <div class="doc-actions"><button data-a="copy">Copy</button><button data-a="dl">Download</button></div>
    </div>
    <div class="doc-body"><div class="md"></div><div class="raw" style="display:none"></div></div>`;
  wrap.querySelector('.md').innerHTML=renderMd(d.markdown);
  wrap.querySelector('.raw').textContent=d.markdown;
  const seg=wrap.querySelectorAll('.seg button'), md=wrap.querySelector('.md'), raw=wrap.querySelector('.raw');
  seg.forEach(b=>b.onclick=()=>{ seg.forEach(x=>x.classList.toggle('on',x===b));
    const v=b.dataset.v; md.style.display=v==='md'?'':'none'; raw.style.display=v==='raw'?'':'none'; });
  wrap.querySelector('[data-a=copy]').onclick=()=>navigator.clipboard.writeText(d.markdown);
  wrap.querySelector('[data-a=dl]').onclick=()=>{
    const slug=(d.title||'transcript').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')||'transcript';
    const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([d.markdown],{type:'text/markdown'}));
    a.download=slug+'.md'; a.click(); URL.revokeObjectURL(a.href); };
  return wrap;
}
function toBottom(){ scroll.scrollTop=scroll.scrollHeight; }

/* ---- open a saved item ---- */
function openItem(item){
  current=item.id; $('#topTitle').textContent=item.title; renderHist(); clearStream();
  userMsg(item.source);
  const m=msgShell('hearsay',true); m.querySelector('.body').appendChild(docNode(item));
  stream.appendChild(m); scroll.scrollTop=0;
}

/* ---- submit ---- */
function newChat(){ pickedFile=null; updChip(); $('#input').value=''; autoGrow(); showHero(); }
$('#new').onclick=newChat;

function downloadZip(b64, name){
  const bin=atob(b64); const arr=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([arr],{type:'application/zip'}));
  a.download=name||'hearsay-dataset.zip'; a.click(); URL.revokeObjectURL(a.href);
}
function datasetNode(d){
  const wrap=document.createElement('div'); wrap.className='doc';
  const warns=(d.warnings||[]).map(w=>'<p style="color:var(--accent)">! '+esc(w)+'</p>').join('');
  wrap.innerHTML='<div class="doc-head"><div class="meta">'+
    '<span class="badge">DATASET</span>'+
    '<span><b>'+d.clips+'</b> clips</span><span>duration <b>'+esc(d.duration)+'</b></span>'+
    '<span>dropped <b>'+d.dropped+'</b></span></div>'+
    '<div class="doc-actions"><button data-a="dl">Download .zip</button></div></div>'+
    '<div class="doc-body"><div class="md"><p>Dataset ready — the .zip download should have started.</p>'+warns+'</div></div>';
  wrap.querySelector('[data-a=dl]').onclick=()=>downloadZip(d.zip_b64,d.zip_name);
  return wrap;
}

async function submit(){
  const text=$('#input').value.trim();
  if(!pickedFile && !text) return;
  const model=$('#model').value, lang=$('#lang').value.trim(),
        vad=$('#vad').checked, transcribe=$('#transcribe').checked, dataset=$('#dataset').checked;
  const sourceLabel = pickedFile ? ('File · '+pickedFile.name) : text;

  if(stream.querySelector('.hero')) clearStream();
  current=null; $('#topTitle').textContent = pickedFile ? pickedFile.name : (text.slice(0,60)||'Transcript');
  userMsg(sourceLabel);
  const thinking=botThinking();
  $('#send').disabled=true;
  const fileToSend=pickedFile; pickedFile=null; updChip(); $('#input').value=''; autoGrow();

  try{
    let res;
    if(dataset){
      const ds={model,lang,vad,segment_min:$('#segMin').value,segment_max:$('#segMax').value,sample_rate:$('#sr').value};
      if(fileToSend){
        const q=new URLSearchParams({name:fileToSend.name,model,lang,vad:vad?'1':'0',
          segment_min:ds.segment_min,segment_max:ds.segment_max,sample_rate:ds.sample_rate});
        res=await fetch('/api/dataset-file?'+q,{method:'POST',body:fileToSend});
      }else{
        res=await fetch('/api/dataset',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({url:text,...ds})});
      }
      const d=await res.json();
      if(!d.ok) throw new Error((d.error||'Failed')+(d.hint?'\nTry: '+d.hint:''));
      downloadZip(d.zip_b64,d.zip_name);
      thinking.querySelector('.body').innerHTML=''; thinking.querySelector('.body').appendChild(datasetNode(d));
      $('#topTitle').textContent='Dataset · '+d.clips+' clips';
      return;  // datasets aren't added to history (not re-openable as markdown)
    }
    if(fileToSend){
      const q=new URLSearchParams({name:fileToSend.name,model,lang,vad:vad?'1':'0'});
      res=await fetch('/api/file?'+q,{method:'POST',body:fileToSend});
    }else{
      res=await fetch('/api/url',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({url:text,transcribe,model,lang,vad})});
    }
    const d=await res.json();
    if(!d.ok) throw new Error((d.error||'Failed')+(d.hint?'\nTry: '+d.hint:''));
    thinking.querySelector('.body').innerHTML=''; thinking.querySelector('.body').appendChild(docNode(d));
    $('#topTitle').textContent=d.title;
    const item={...d, id:Date.now(), source:sourceLabel};
    const h=load(); h.unshift(item); save(h); current=item.id; renderHist();
  }catch(e){
    const b=thinking.querySelector('.body'); b.innerHTML='';
    const er=document.createElement('div'); er.className='err'; er.textContent=e.message; b.appendChild(er);
  }finally{
    $('#send').disabled=false; toBottom();
  }
}
$('#send').onclick=submit;
$('#dataset').onchange=e=>{ $('#dsOpts').style.display=e.target.checked?'flex':'none'; };

/* ---- composer behaviour ---- */
const input=$('#input'), comp=$('#composer');
function autoGrow(){ input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,168)+'px'; }
input.addEventListener('input',autoGrow);
input.addEventListener('focus',()=>comp.classList.add('focus'));
input.addEventListener('blur',()=>comp.classList.remove('focus'));
input.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); submit(); } });
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=e=>{ pickedFile=e.target.files[0]||null; updChip(); e.target.value=''; };
function updChip(){ const c=$('#fileChip'); if(pickedFile){ $('#fileName').textContent='File · '+pickedFile.name; c.classList.add('show'); }
  else c.classList.remove('show'); }
$('#fileX').onclick=()=>{ pickedFile=null; updChip(); };
['dragover','dragenter'].forEach(ev=>comp.addEventListener(ev,e=>{e.preventDefault();comp.classList.add('drag');}));
['dragleave','drop'].forEach(ev=>comp.addEventListener(ev,e=>{e.preventDefault();comp.classList.remove('drag');}));
comp.addEventListener('drop',e=>{ if(e.dataTransfer.files[0]){ pickedFile=e.dataTransfer.files[0]; updChip(); } });

/* ---- boot ---- */
renderHist(); showHero();
</script>
</body>
</html>
"""
