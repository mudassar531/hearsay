"""Tests for the web UI's dataset mode (build -> zip download), offline.

The server runs on an ephemeral loopback port; the build_dataset_from_* functions
(lazily imported inside webui) are monkeypatched on hearsay.dataset.build to write
a tiny dataset and return a report — so no transcription/network happens.
"""

import http.client
import io
import json
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

import hearsay.dataset.build as ds_build
from hearsay import webui
from hearsay.dataset.models import BuildReport


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
    conn.close()
    return resp.status, data


def _fake_build(captured: dict):
    def fake(arg, *, config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        (config.out_dir / "wavs").mkdir(parents=True, exist_ok=True)
        (config.out_dir / "manifest.jsonl").write_text(
            '{"audio_filepath": "wavs/x_0001.wav", "duration": 1.0, "text": "hi", "offset": 0.0}\n'
        )
        return BuildReport(
            out_dir=str(config.out_dir),
            source=str(arg),
            clip_count=2,
            total_duration_s=3.0,
            oversized_count=0,
            sample_rate=config.sample_rate,
            language="en",
            formats=config.formats,
            warnings=["a heads-up"],
        )

    return fake


def _post_zip(
    addr: str, path: str, body: bytes, ctype: str = "application/json"
) -> tuple[dict, bytes]:
    """POST and return ``(report, zip_bytes)`` from a dataset response (a file, not JSON)."""
    conn = http.client.HTTPConnection(addr, timeout=5)
    conn.request("POST", path, body=body, headers={"Content-Type": ctype})
    resp = conn.getresponse()
    assert resp.status == 200, resp.read()
    assert resp.getheader("Content-Type") == "application/zip"
    assert resp.getheader("Content-Disposition", "").startswith("attachment;")
    report = json.loads(resp.getheader("X-Hearsay-Report") or "{}")
    data = resp.read()
    conn.close()
    return report, data


def _zip_names(raw: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return zf.namelist()


def test_dataset_url_builds_and_returns_zip(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(ds_build, "build_dataset_from_media_url", _fake_build(captured))
    body = json.dumps(
        {
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "sample_rate": 16000,
            "segment_min": 2,
            "segment_max": 8,
        }
    ).encode()
    data, raw = _post_zip(server, "/api/dataset", body)
    assert data["dataset"] is True
    assert data["clips"] == 2 and data["dropped"] == 0
    assert data["warnings"] == ["a heads-up"]
    assert "manifest.jsonl" in _zip_names(raw)  # a real, loadable zip, sent as a file
    # the request options reached the build config
    assert captured["config"].sample_rate == 16000
    assert captured["config"].segment_min_s == 2.0 and captured["config"].segment_max_s == 8.0


def test_dataset_url_builds_a_playlist(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Playlists used to be refused here and sent to the CLI; the browser now builds them
    # into one merged dataset, capped so a huge feed can't be pulled through one response.
    captured: dict = {}

    def fake_playlist(url: str, **kwargs: object):
        captured["url"] = url
        captured["limit"] = kwargs.get("limit")
        return _fake_build(captured)(url, **{k: v for k, v in kwargs.items() if k != "limit"})

    monkeypatch.setattr(ds_build, "build_dataset_from_playlist", fake_playlist)
    body = json.dumps({"url": "https://www.youtube.com/playlist?list=PL1234567890"}).encode()
    data, _raw = _post_zip(server, "/api/dataset", body)
    assert data["dataset"] is True
    assert captured["url"].endswith("list=PL1234567890")
    assert captured["limit"] == 5  # default browser cap


def test_dataset_url_rejects_empty(server: str) -> None:
    status, data = _post(server, "/api/dataset", json.dumps({"url": ""}).encode())
    assert status == 400 and data["hint"]


def test_dataset_file_builds_and_returns_zip(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(ds_build, "build_dataset_from_file", _fake_build(captured))
    data, raw = _post_zip(
        server,
        "/api/dataset-file?name=clip.wav&sample_rate=22050&segment_min=1&segment_max=15",
        b"fake-audio-bytes",
        ctype="audio/wav",
    )
    assert data["dataset"] is True
    assert "manifest.jsonl" in _zip_names(raw)
    assert captured["config"].sample_rate == 22050


def test_dataset_page_offers_dataset_mode(server: str) -> None:
    conn = http.client.HTTPConnection(server, timeout=5)
    conn.request("GET", "/")
    body = conn.getresponse().read().decode()
    conn.close()
    # Output is a visible mode choice, not a checkbox people miss.
    assert 'id="modeDs"' in body and 'id="modeDoc"' in body
    assert "/api/dataset" in body
    # And the language is picked from a list, so an invalid code cannot be typed.
    assert '<select id="lang">' in body and 'value="ur"' in body


def test_dataset_malformed_numeric_param_is_400(
    server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ds_build, "build_dataset_from_media_url", _fake_build({}))
    body = json.dumps(
        {"url": "https://www.youtube.com/watch?v=abcdefghijk", "sample_rate": "notanumber"}
    ).encode()
    status, data = _post(server, "/api/dataset", body)
    assert status == 400 and data["ok"] is False
    assert data["hint"]  # friendly, not an opaque 500


def test_dataset_no_temp_leak(server: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tempfile

    monkeypatch.setattr(ds_build, "build_dataset_from_media_url", _fake_build({}))
    before = set(Path(tempfile.gettempdir()).glob("hearsay-web-ds-*"))
    _post_zip(
        server,
        "/api/dataset",
        json.dumps({"url": "https://www.youtube.com/watch?v=abcdefghijk"}).encode(),
    )
    after = set(Path(tempfile.gettempdir()).glob("hearsay-web-ds-*"))
    assert after == before  # the build's TemporaryDirectory was cleaned up


def test_zip_dir_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "wavs").mkdir()
    (tmp_path / "wavs" / "a.wav").write_bytes(b"RIFF....")
    (tmp_path / "metadata.csv").write_text("id|t|t\n")
    names = _zip_names(webui._zip_dir(tmp_path))
    assert set(names) == {"wavs/a.wav", "metadata.csv"}


def test_malformed_json_body_is_a_400(server: str) -> None:
    status, data = _post(server, "/api/dataset", b"{not json")
    assert status == 400 and "JSON" in data["error"]
    status, data = _post(server, "/api/url", b"[1, 2]")
    assert status == 400 and data["hint"]
