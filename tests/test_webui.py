"""Tests for the simple web UI.

The server is started on an ephemeral loopback port (conftest allows local
connections) and exercised over real HTTP. Network-touching pipeline calls are
replaced with fakes so the suite stays offline.
"""

import http.client
import json
import threading
from collections.abc import Iterator

import pytest

from hearsay import webui
from hearsay.errors import InvalidSourceError
from hearsay.models import Document, SourceMetadata

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _doc(title: str, method: str) -> Document:
    return Document(
        meta=SourceMetadata(title=title, source="x", channel="c", duration_s=12.0, video_id="vid"),
        method=method,
        language="en",
        ingested_at="2026-01-01T00:00:00Z",
        sections=[],
    )


@pytest.fixture
def server() -> Iterator[str]:
    httpd = webui.make_server("127.0.0.1", 0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _post(addr: str, path: str, body: bytes, ctype: str = "application/json") -> tuple[int, dict]:
    conn = http.client.HTTPConnection(addr, timeout=5)
    conn.request("POST", path, body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    status = resp.status
    conn.close()
    return status, data


def test_index_serves_html(server: str) -> None:
    conn = http.client.HTTPConnection(server, timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    assert resp.status == 200
    assert "text/html" in resp.getheader("Content-Type", "")
    assert "<title>hearsay</title>" in body


def test_url_endpoint_rejects_empty(server: str) -> None:
    status, data = _post(server, "/api/url", json.dumps({"url": ""}).encode())
    assert status == 400
    assert data["ok"] is False
    assert data["hint"]


def test_url_endpoint_rejects_playlist(server: str) -> None:
    url = "https://www.youtube.com/playlist?list=PL1234567890"
    status, data = _post(server, "/api/url", json.dumps({"url": url}).encode())
    assert status == 400
    assert "playlist" in data["error"].lower() or "feed" in data["error"].lower()


def test_url_endpoint_returns_markdown(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webui, "ingest_youtube", lambda url, **k: _doc("My Video", "captions"))
    url = "https://www.youtube.com/watch?v=abcdefghijk"
    status, data = _post(server, "/api/url", json.dumps({"url": url}).encode())
    assert status == 200
    assert data["ok"] is True
    assert data["title"] == "My Video"
    assert data["method"] == "captions"
    assert data["markdown"].startswith("---")


def test_url_transcribe_uses_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_transcribe(url: str, **kwargs: object) -> Document:
        called.update(kwargs)
        return _doc("T", "parakeet-tdt-0.6b-v3")

    monkeypatch.setattr(webui, "ingest_youtube_transcribe", fake_transcribe)
    doc = webui.process_url(
        "https://youtu.be/abcdefghijk", transcribe=True, model="parakeet", language="en", vad=False
    )
    assert doc.method == "parakeet-tdt-0.6b-v3"
    assert called["model_size"] == "parakeet"
    assert called["vad_filter"] is False


def test_file_endpoint_dispatches(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_ingest_file(path: object, **kwargs: object) -> Document:
        seen["name"] = __import__("pathlib").Path(str(path)).name
        seen.update(kwargs)
        return _doc("clip", "whisper-tiny")

    monkeypatch.setattr(webui, "ingest_file", fake_ingest_file)
    status, data = _post(
        server, "/api/file?name=clip.mp3&model=tiny", b"fake-audio-bytes", ctype="audio/mpeg"
    )
    assert status == 200
    assert data["ok"] is True
    assert seen["name"] == "clip.mp3"
    assert seen["model_size"] == "tiny"


def test_process_url_rejects_non_youtube() -> None:
    with pytest.raises(InvalidSourceError):
        webui.process_url("https://example.com/not-a-video")
