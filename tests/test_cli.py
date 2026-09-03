"""Tests for the CLI: version/help smoke tests plus the ingest command."""

import json
import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hearsay.cli as cli
from hearsay.captions import CaptionResult, normalize_snippets
from hearsay.cli import app
from hearsay.errors import NoCaptionsError
from hearsay.feeds import Episode, Feed
from hearsay.models import Document, Paragraph, Section, SourceMetadata, Transcript
from hearsay.pipeline import build_document
from hearsay.youtube import PlaylistEntry, parse_metadata

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


# --- Smoke tests ----------------------------------------------------------


def test_version_flag_prints_real_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert re.fullmatch(r"hearsay \d+\.\d+\.\d+\S*\n?", result.output)
    assert "0.0.0.dev0" not in result.output


def test_help_shows_pitch() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM-ready" in result.output


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "Usage:" in result.output


def test_console_script_points_at_cli_app() -> None:
    (script,) = entry_points(group="console_scripts", name="hearsay")
    assert script.value == "hearsay.cli:app"
    assert script.load() is app


# --- Command-group routing (DefaultCommandGroup) --------------------------


def test_mcp_subcommand_is_registered() -> None:
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "server" in result.output.lower()


def test_options_before_source_route_to_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `hearsay -o X <source>` (options before the positional) must still work.
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    dest = tmp_path / "o.md"
    result = runner.invoke(app, ["-o", str(dest), "https://youtu.be/rStL7niR7gs"])
    assert result.exit_code == 0, result.output
    assert dest.exists()


def test_unknown_bare_token_is_treated_as_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare token that isn't a subcommand routes to ingest (as a source path).
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["notacommand"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "not found" in result.output.lower()


# --- Ingest command (offline via a patched pipeline) ----------------------


def _fixture_document(video_id: str) -> Document:
    meta = parse_metadata(json.loads((FIXTURES / f"{video_id}.meta.json").read_text()), "url")
    data = json.loads((FIXTURES / f"{video_id}.transcript.json").read_text())
    captions = CaptionResult(
        segments=normalize_snippets(data["snippets"]),
        language_code=data["language_code"],
        is_generated=data["is_generated"],
    )
    return build_document(meta, captions, ingested_at="2026-06-13T10:00:00Z")


def test_ingest_writes_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    result = runner.invoke(app, ["https://www.youtube.com/watch?v=rStL7niR7gs"])
    assert result.exit_code == 0, result.output
    out = tmp_path / "rStL7niR7gs.md"
    assert out.exists()
    text = out.read_text()
    assert text.startswith("---\n")
    assert "You Would Be a Terrible Leader" in text


def test_ingest_respects_output_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    dest = tmp_path / "custom.md"
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "--output", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.exists()


def test_non_feed_url_falls_back_to_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-feed http URL is a media page, not an error.

    Only fetching distinguishes a podcast feed from a video page, so a FeedError means
    "not a feed" — hand it to yt-dlp, which supports ~1800 sites. Before this, every
    non-YouTube video URL died with an RSS complaint.
    """
    from hearsay.errors import FeedError

    def not_a_feed(url: str):
        raise FeedError(f"No podcast episodes found at {url}.", hint="Point me at an RSS feed.")

    seen: dict = {}

    def fake_transcribe(url: str, **kwargs: object):
        seen["url"] = url
        return _whisper_doc("Sonda wyborcza")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "fetch_feed", not_a_feed)
    monkeypatch.setattr(cli, "ingest_youtube_transcribe", fake_transcribe)
    result = runner.invoke(app, ["https://www.dailymotion.com/video/x8ougt8"])
    assert result.exit_code == 0, result.output
    assert seen["url"] == "https://www.dailymotion.com/video/x8ougt8"
    assert (tmp_path / "x8ougt8.md").exists()


def test_unreachable_url_still_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hearsay.errors import FeedError, MetadataError

    def not_a_feed(url: str):
        raise FeedError("no entries", hint="Point me at an RSS feed.")

    def no_such_video(url: str, **kwargs: object):
        raise MetadataError(f"yt-dlp failed for {url}: Unsupported URL", hint="Check the link.")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "fetch_feed", not_a_feed)
    monkeypatch.setattr(cli, "ingest_youtube_transcribe", no_such_video)
    result = runner.invoke(app, ["https://example.com/not-a-video"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Unsupported URL" in result.output
    assert not list(tmp_path.glob("*.md"))


def test_ingest_bad_output_path_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing parent directory must produce a hint, not a traceback.
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    dest = tmp_path / "missing_dir" / "out.md"
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "-o", str(dest)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "directory" in result.output.lower()


def test_ingest_output_is_directory_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "directory" in result.output.lower()


# --- Whisper paths (offline; ingestion functions are patched) -------------


def _whisper_doc(title: str = "clip") -> Document:
    meta = SourceMetadata(
        title=title, source="x", channel="Local file", duration_s=10.0, video_id=title
    )
    return Document(
        meta=meta,
        method="whisper-tiny",
        language="en",
        ingested_at="2026-06-13T10:00:00Z",
        sections=[
            Section(
                title="[00:00:00 – 00:00:10]",
                start_s=0,
                paragraphs=[Paragraph(text="hello world from whisper", start_s=0, end_s=10)],
            )
        ],
    )


def test_local_file_is_transcribed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "talk.wav"
    clip.write_bytes(b"x")
    captured: dict[str, object] = {}

    def fake_ingest_file(path, **kwargs):
        captured["path"] = path
        captured["model_size"] = kwargs.get("model_size")
        return _whisper_doc("talk")

    monkeypatch.setattr(cli, "ingest_file", fake_ingest_file)
    out = tmp_path / "talk.md"
    result = runner.invoke(app, [str(clip), "--model", "tiny", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert captured["model_size"] == "tiny"
    assert out.read_text().startswith("---\n")
    assert "whisper-tiny" in out.read_text()


def test_transcribe_flag_forces_whisper_on_youtube(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"captions": False, "transcribe": False}

    def fake_captions(*a, **k):
        called["captions"] = True
        return _fixture_document("rStL7niR7gs")

    def fake_transcribe(*a, **k):
        called["transcribe"] = True
        return _whisper_doc("forced")

    monkeypatch.setattr(cli, "ingest_youtube", fake_captions)
    monkeypatch.setattr(cli, "ingest_youtube_transcribe", fake_transcribe)
    result = runner.invoke(
        app, ["https://youtu.be/rStL7niR7gs", "--transcribe", "-o", str(tmp_path / "o.md")]
    )
    assert result.exit_code == 0, result.output
    assert called["transcribe"] is True
    assert called["captions"] is False  # --transcribe skips captions entirely


def test_no_vad_flag_threads_to_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_transcribe(*a, **k):
        captured["vad_filter"] = k.get("vad_filter")
        return _whisper_doc("song")

    monkeypatch.setattr(cli, "ingest_youtube_transcribe", fake_transcribe)
    url = "https://youtu.be/rStL7niR7gs"
    # default → VAD on
    runner.invoke(app, [url, "--transcribe", "-o", str(tmp_path / "a.md")])
    assert captured["vad_filter"] is True
    # --no-vad → VAD off
    runner.invoke(app, [url, "--transcribe", "--no-vad", "-o", str(tmp_path / "b.md")])
    assert captured["vad_filter"] is False


def test_auto_fallback_to_whisper_when_no_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_captions(*a, **k):
        raise NoCaptionsError("This video has no captions in any language: x", hint="...")

    fell_back = {"yes": False}

    def fake_transcribe(*a, **k):
        fell_back["yes"] = True
        return _whisper_doc("fallback")

    monkeypatch.setattr(cli, "ingest_youtube", no_captions)
    monkeypatch.setattr(cli, "ingest_youtube_transcribe", fake_transcribe)
    out = tmp_path / "o.md"
    result = runner.invoke(app, ["https://www.youtube.com/watch?v=abcdefghijk", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert fell_back["yes"] is True
    assert "No captions found" in result.output
    assert out.exists()


def test_unsupported_local_file_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [str(notes)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "unsupported" in result.output.lower()


def test_invalid_model_choice_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # --model now accepts a CTranslate2 repo id or local path as well as a size, so the
    # check moved from Typer's enum to hearsay, which can say what would have worked.
    monkeypatch.chdir(tmp_path)
    clip = tmp_path / "a.wav"
    clip.write_bytes((FIXTURES / "sample.wav").read_bytes())
    result = runner.invoke(app, [str(clip), "--model", "huge"])
    assert result.exit_code == 1
    assert "huge" in result.output
    assert "Traceback" not in result.output


def test_existing_file_wins_over_youtube_substring_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real file under a "youtu.be" folder must be transcribed, not sent to yt-dlp.
    weird = tmp_path / "youtu.be" / "abcdefghijk.wav"
    weird.parent.mkdir()
    weird.write_bytes(b"x")
    routed = {"file": False, "youtube": False}

    def fake_file(path, **kwargs):
        routed["file"] = True
        return _whisper_doc(path.stem)

    def fake_youtube(*a, **k):
        routed["youtube"] = True
        return _fixture_document("rStL7niR7gs")

    monkeypatch.setattr(cli, "ingest_file", fake_file)
    monkeypatch.setattr(cli, "ingest_youtube", fake_youtube)
    result = runner.invoke(app, [str(weird), "-o", str(tmp_path / "o.md")])
    assert result.exit_code == 0, result.output
    assert routed["file"] is True
    assert routed["youtube"] is False


# --- JSON sidecar ---------------------------------------------------------


def test_json_sidecar_written_and_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    md = tmp_path / "out.md"
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "--json", "-o", str(md)])
    assert result.exit_code == 0, result.output
    sidecar = tmp_path / "out.json"
    assert md.exists() and sidecar.exists()
    transcript = Transcript.model_validate_json(sidecar.read_text())
    assert transcript.title == "You Would Be a Terrible Leader"
    assert transcript.chunks
    assert transcript.chunks[0].section  # every chunk is tagged with its section


def test_json_sidecar_does_not_clobber_md_when_output_ends_in_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    dest = tmp_path / "report.json"
    result = runner.invoke(app, ["https://youtu.be/rStL7niR7gs", "--json", "-o", str(dest)])
    assert result.exit_code == 0, result.output
    # Markdown went to the -o target; the JSON sidecar did not overwrite it.
    assert dest.read_text().startswith("---\n")
    assert (tmp_path / "report.json.json").exists()


# --- Batch: playlists and feeds -------------------------------------------


def _entries(n: int) -> list[PlaylistEntry]:
    return [
        PlaylistEntry(
            video_id=f"vid{i:08d}xyz"[:11], title=f"Video {i}", url=f"https://youtu.be/v{i}"
        )
        for i in range(1, n + 1)
    ]


def test_playlist_no_selection_lists_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "fetch_playlist", lambda url: ("My Playlist", _entries(3)))
    result = runner.invoke(app, ["https://www.youtube.com/playlist?list=PLabc"])
    assert result.exit_code == 0, result.output
    assert "My Playlist" in result.output
    assert "Video 1" in result.output and "Video 3" in result.output


def test_playlist_latest_ingests_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "fetch_playlist", lambda url: ("PL", _entries(3)))
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["https://www.youtube.com/playlist?list=PLabc", "--latest", "--output-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.md"))) == 1


def test_playlist_batch_continues_past_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "fetch_playlist", lambda url: ("PL", _entries(3)))

    def flaky(url, **kwargs):  # second video fails; others succeed
        if url == "https://youtu.be/v2":
            from hearsay.errors import VideoUnavailableError

            raise VideoUnavailableError("v2 is private", hint="...")
        return _fixture_document("rStL7niR7gs")

    monkeypatch.setattr(cli, "ingest_youtube", flaky)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["https://www.youtube.com/playlist?list=PLabc", "--all", "--output-dir", str(out)]
    )
    # A partial batch still writes everything that worked, but exits 1 so a script or
    # CI job can see that work was lost (2 would mean nothing succeeded at all).
    assert result.exit_code == 1, result.output
    assert len(list(out.glob("*.md"))) == 2  # two succeeded, one failed but did not abort
    assert "1 failed" in result.output or "private" in result.output


def _feed(n: int) -> Feed:
    episodes = [
        Episode(
            title=f"Episode {i}", audio_url=f"http://x/{i}.mp3", duration_s=60.0 * i, guid=f"g{i}"
        )
        for i in range(1, n + 1)
    ]
    return Feed(title="My Show", episodes=episodes)


def test_feed_no_selection_lists_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "fetch_feed", lambda url: _feed(2))
    result = runner.invoke(app, ["https://example.com/feed.xml"])
    assert result.exit_code == 0, result.output
    assert "My Show" in result.output
    assert "Episode 1" in result.output


def test_feed_episode_n_ingests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "fetch_feed", lambda url: _feed(3))
    captured: dict[str, object] = {}

    def fake_ingest_episode(episode, show, **kwargs):
        captured["title"] = episode.title
        return _whisper_doc(episode.title)

    monkeypatch.setattr(cli, "ingest_episode", fake_ingest_episode)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["https://example.com/feed.xml", "--episode", "2", "--output-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert captured["title"] == "Episode 2"
    assert len(list(out.glob("*.md"))) == 1


def test_feed_colliding_titles_do_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two distinct titles that slugify to the same stem must yield two files.
    episodes = [
        Episode(title="Ep #1: Hi!", audio_url="http://x/1.mp3", duration_s=60.0, guid="a"),
        Episode(title="Ep 1 Hi", audio_url="http://x/2.mp3", duration_s=60.0, guid="b"),
    ]
    monkeypatch.setattr(cli, "fetch_feed", lambda url: Feed(title="Show", episodes=episodes))
    monkeypatch.setattr(cli, "ingest_episode", lambda ep, show, **k: _whisper_doc(ep.title))
    out = tmp_path / "out"
    result = runner.invoke(app, ["https://example.com/feed.xml", "--all", "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.md"))) == 2  # no silent overwrite


def test_batch_exit_code_is_2_when_every_item_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch where nothing worked used to exit 0 — indistinguishable from success."""
    monkeypatch.setattr(cli, "fetch_playlist", lambda url: ("PL", _entries(2)))

    def always_fails(url, **kwargs):
        from hearsay.errors import VideoUnavailableError

        raise VideoUnavailableError("private", hint="...")

    monkeypatch.setattr(cli, "ingest_youtube", always_fails)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["https://www.youtube.com/playlist?list=PLabc", "--all", "--output-dir", str(out)]
    )
    assert result.exit_code == 2, result.output
    assert list(out.glob("*.md")) == []


def test_batch_exit_code_is_0_when_everything_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "fetch_playlist", lambda url: ("PL", _entries(2)))
    monkeypatch.setattr(cli, "ingest_youtube", lambda url, **kw: _fixture_document("rStL7niR7gs"))
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["https://www.youtube.com/playlist?list=PLabc", "--all", "--output-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.md"))) == 2


# --- Runtime flags reach yt-dlp and Whisper through the shared env vars ----------


def test_cookies_and_device_flags_set_the_shared_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HEARSAY_YTDLP_ARGS", raising=False)
    monkeypatch.delenv("HEARSAY_DEVICE", raising=False)
    monkeypatch.setattr(cli, "ingest_file", lambda *a, **k: _whisper_doc())
    result = runner.invoke(
        app,
        [
            str(FIXTURES / "sample.wav"),
            "--cookies-from-browser",
            "chrome",
            "--device",
            "cpu",
            "-o",
            str(tmp_path / "x.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert cli.os.environ["HEARSAY_YTDLP_ARGS"] == "--cookies-from-browser chrome"
    assert cli.os.environ["HEARSAY_DEVICE"] == "cpu"


# --- --lang on the captions path ---------------------------------------------------


def test_lang_is_validated_before_captions_are_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--lang urdu` used to reach the captions path, match nothing, and quietly hand back
    whatever track existed."""
    called = []
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: called.append(1))
    result = runner.invoke(
        app,
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--lang",
            "urdu",
            "-o",
            str(tmp_path / "x"),
        ],
    )
    assert result.exit_code == 1 and not called
    assert "Did you mean 'ur'" in result.output


def test_caption_language_fallback_is_announced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Captions fall back to any available track; a Spanish request that quietly came
    back English read as a Spanish transcript."""
    monkeypatch.setattr(cli, "ingest_youtube", lambda *a, **k: _fixture_document("rStL7niR7gs"))
    result = runner.invoke(
        app,
        [
            "https://www.youtube.com/watch?v=rStL7niR7gs",
            "--lang",
            "es",
            "-o",
            str(tmp_path / "x.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No 'es' captions" in result.output and "'en' track" in result.output
    # and nothing is said when the requested track was the one used
    result = runner.invoke(
        app,
        [
            "https://www.youtube.com/watch?v=rStL7niR7gs",
            "--lang",
            "en-GB",
            "-o",
            str(tmp_path / "y.md"),
        ],
    )
    assert result.exit_code == 0 and "captions on this video" not in result.output
