"""Local speech-to-text with a pluggable backend.

Two engines are supported and selected by model name:

* **Parakeet** (``parakeet`` / ``parakeet-en``) — NVIDIA Parakeet TDT running on
  Apple's MLX. On Apple Silicon it transcribes ~3x faster than ``whisper-small``
  on CPU (~24x vs ~7x realtime on an M1 Pro), with comparable accuracy.
  Multilingual (v3) or English-only (v2).
* **Whisper** (``tiny``…``large-v3``) — faster-whisper / ctranslate2 on CPU,
  int8. Portable everywhere; the fallback when Parakeet is unavailable.

``auto`` (the default) picks Parakeet when it is installed and runnable, else
``whisper-small`` — so every platform gets the fastest engine it has.

Both backends are imported lazily inside the functions here: the captions-only
path, and the engine you are not using, never pay the (heavy, slow) import cost.
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from faster_whisper.tokenizer import _LANGUAGE_CODES

from hearsay.errors import TranscriptionError
from hearsay.models import Segment, Word

# Whisper sizes, smallest (fastest) to largest (most accurate).
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")
# Back-compat alias (older imports referenced MODEL_SIZES).
MODEL_SIZES = WHISPER_SIZES

# Parakeet model aliases → MLX-community repo ids. ``parakeet`` is multilingual
# (25 European languages); ``parakeet-en`` is English-only. Override the repo
# for either alias with HEARSAY_PARAKEET_MODEL.
PARAKEET_MODELS = {
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
    "parakeet-en": "mlx-community/parakeet-tdt-0.6b-v2",
}

# Whisper size used when ``auto`` falls back off Parakeet.
DEFAULT_WHISPER = "small"
# Languages where the small checkpoints do not merely lose accuracy but produce the
# wrong thing entirely. Measured on Uzbek: `small` returns romanised approximations,
# `medium` collapses into Khmer and Georgian glyphs, and only `large-v3` returns
# readable Uzbek. Whisper's own per-language WER puts all of these in its worst tier,
# so `auto` opens the large checkpoint for them instead of shipping confident nonsense.
LOW_RESOURCE_WHISPER = "large-v3"
LOW_RESOURCE_LANGUAGES = frozenset(
    [
        "am",
        "az",
        "ba",
        "be",
        "bo",
        "br",
        "cy",
        "fo",
        "gl",
        "gu",
        "ha",
        "haw",
        "hy",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "la",
        "lb",
        "ln",
        "lo",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "mt",
        "my",
        "ne",
        "nn",
        "oc",
        "pa",
        "ps",
        "sa",
        "sd",
        "si",
        "sn",
        "so",
        "sq",
        "su",
        "sw",
        "ta",
        "te",
        "tg",
        "tk",
        "tl",
        "tt",
        "uk",
        "ur",
        "uz",
        "yi",
        "yo",
        "yue",
    ]
)
# Parakeet TDT 0.6b v3 is multilingual over exactly these 25 European languages. It does
# not refuse anything else — it transliterates it into confident nonsense, so an Urdu or
# Arabic recording comes back as fluent-looking Latin gibberish with no error anywhere.
# `auto` therefore must not hand it audio in a language it cannot read.
PARAKEET_LANGUAGES = frozenset(
    [
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    ]
)
# Whisper size used purely to identify the language before picking an engine: the
# smallest checkpoint is enough to tell Urdu from German, and it looks at one window.
_DETECT_WHISPER = "tiny"
# Below this confidence the guess is worth less than the default, so it is discarded.
_DETECT_MIN_PROBABILITY = 0.35

# The codes Whisper accepts, and the handful of English names people reach for instead
# (a full name table would be a new dependency for a typo hint).
_WHISPER_LANGUAGE_CODES = frozenset(_LANGUAGE_CODES)
_NAMED_LANGUAGES = {
    "urdu": "ur", "arabic": "ar", "hindi": "hi", "english": "en", "spanish": "es",
    "french": "fr", "german": "de", "chinese": "zh", "mandarin": "zh", "japanese": "ja",
    "korean": "ko", "russian": "ru", "portuguese": "pt", "italian": "it", "dutch": "nl",
    "turkish": "tr", "persian": "fa", "farsi": "fa", "punjabi": "pa", "bengali": "bn",
    "uzbek": "uz", "pashto": "ps", "sindhi": "sd", "tamil": "ta", "telugu": "te",
    "indonesian": "id", "vietnamese": "vi", "thai": "th", "polish": "pl", "ukrainian": "uk",
}  # fmt: skip
# Default model: resolve the fastest available engine at transcription time.
DEFAULT_MODEL = "auto"
# ISO 639-2 "undetermined": the engine transcribed the audio but cannot name its
# language. Downstream, an unknown language disables the language-specific filters
# rather than guessing at them.
UNKNOWN_LANGUAGE = "und"

# Long files are transcribed in overlapping windows so progress can be reported
# and memory stays bounded; shorter files run in a single pass.
_PARAKEET_CHUNK_S = 120.0

# Reports (processed_seconds, total_seconds) as transcription advances.
ProgressCallback = Callable[[float, float], None]


class TranscriptionResult(NamedTuple):
    """Engine output mapped onto hearsay's segment model.

    ``method`` is the document-facing label (e.g. ``whisper-small`` or
    ``parakeet-tdt-0.6b-v3``); ``model_size`` is the requested model identifier.
    ``words`` is the engine-agnostic word-level timing, populated only when
    ``word_timestamps=True`` (dataset mode); ``None`` otherwise, so the markdown
    path and its JSON schema are unaffected.
    """

    segments: list[Segment]
    language: str
    duration_s: float
    model_size: str
    method: str
    words: list[Word] | None = None


def transcribe_audio(
    path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    local_files_only: bool = False,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    prompt: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Transcribe an audio/video file into timed segments.

    Args:
        path: Path to a local media file (any container ffmpeg can decode).
        model_size: ``auto`` (default), a Parakeet alias (``parakeet`` /
            ``parakeet-en``), or a Whisper size (``tiny``…``large-v3``).
        language: Force a language code, or ``None`` to auto-detect (Whisper).
            Parakeet auto-detects internally; this only labels the output.
        local_files_only: If True, never reach the network — load the model
            from the local cache or fail. Used by the offline test suite.
        vad_filter: Voice-activity detection (Whisper only). ``True`` (default)
            skips silence/music beds — right for speech. Set ``False`` for
            music/songs. Parakeet ignores this (it has no VAD pre-filter).
        prompt: Text prefixed to the decoder as context (Whisper only). Steers spelling
            and, for a language written in more than one script, which one comes back —
            Uzbek returned 38% Latin / 61% Cyrillic unprompted, and 100% of either with
            a one-line sample of that script. Mixed orthography ruins a TTS dataset.
        word_timestamps: When True, also return engine-agnostic word-level
            timings on ``result.words`` (for dataset mode). Default False, so the
            markdown/JSON path is unchanged and pays no extra cost.
        on_progress: Optional callback invoked with (processed_s, total_s).

    Raises:
        TranscriptionError: the model could not be loaded (e.g. not cached and
            offline, or Parakeet requested but not installed) or the file could
            not be decoded.
    """
    # With no stated language, `auto` cannot know whether Parakeet can read this audio.
    # Identify it first with the smallest Whisper checkpoint (one window, ~40 MB) —
    # otherwise a Naat, a Quran recitation, or any Hindi/Chinese/Japanese recording comes
    # back as confident Latin nonsense that no downstream check can spot.
    language = normalize_language(language)
    detected: str | None = None
    if model_size == "auto" and language is None and _parakeet_available():
        detected = detect_language(path, local_files_only=local_files_only)
        # Parakeet cannot detect at all, so there it *is* the label; Whisper re-detects.
    engine, identifier = _resolve_engine(model_size, language or detected)
    if engine == "parakeet":
        try:
            return _transcribe_parakeet(
                path,
                repo_id=identifier,
                language=language,
                local_files_only=local_files_only,
                word_timestamps=word_timestamps,
                on_progress=on_progress,
            )
        except TranscriptionError:
            # `auto` promises "the fastest engine that works here", so a Parakeet that
            # is importable but cannot actually run (no network for the ~2.5 GB weights,
            # a corrupt cache, an MLX/OS mismatch) must fall through to Whisper rather
            # than leave the machine with no transcription at all. An explicitly named
            # engine still fails loudly — a deliberate choice is never downgraded.
            if model_size != "auto":
                raise
            identifier = DEFAULT_WHISPER
    return _transcribe_whisper(
        path,
        model_size=identifier,
        prompt=prompt,
        # Deliberately NOT `language or detected`: the probe runs on the *tiny*
        # checkpoint and only has to be right enough to pick an engine. Whisper detects
        # again during the real decode with the model actually being used, which is
        # strictly better — forcing the probe's guess turned a Cyrillic Uzbek news
        # bulletin (mis-guessed as Persian) into Arabic-script nonsense.
        language=language,
        local_files_only=local_files_only,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        on_progress=on_progress,
    )


def detect_language(path: Path, *, local_files_only: bool = False) -> str | None:
    """Identify the spoken language of ``path``, or None if it cannot be determined.

    Uses the smallest Whisper checkpoint over a single window — enough to tell Urdu from
    German, which is all the engine picker needs. Detection failing is never fatal: the
    caller falls back to its normal default.
    """
    try:
        from faster_whisper.audio import decode_audio

        model = _load_whisper(_DETECT_WHISPER, local_files_only=local_files_only)
        language, probability, _ = model.detect_language(
            audio=decode_audio(str(path), sampling_rate=16000)
        )
    except Exception:  # a probe is an optimisation, never a reason to fail the job
        return None
    return language if probability >= _DETECT_MIN_PROBABILITY else None


# A language written in more than one script comes back in whichever the decoder
# prefers: unprompted Uzbek measured 38% Latin / 61% Cyrillic within a single clip.
# Seeding the decoder with a line of the target script does pin it to 100% — but on
# audio the model cannot read, Whisper emits that seed *as the transcript*: large-v3 on
# a 20s Uzbek news clip returned the prompt verbatim and nothing else. For a training
# set that is a clip paired with words nobody said, so hearsay never sets a prompt on
# its own. The real answer to a language Whisper cannot read is a model that can —
# pass one to --model. `prompt=` stays available for callers who want it knowingly.
_SCRIPT_SAMPLES = {
    ("uz", "latn"): "Assalomu alaykum. O'zbekiston Respublikasi haqida gaplashamiz.",
    ("uz", "cyrl"): "Ассалому алайкум. Ўзбекистон Республикаси ҳақида гаплашамиз.",
    ("sr", "latn"): "Dobar dan. Razgovaramo o Srbiji i njenoj istoriji.",
    ("sr", "cyrl"): "Добар дан. Разговарамо о Србији и њеној историји.",  # noqa: RUF001
}
_DEFAULT_SCRIPTS = {"uz": "latn", "sr": "cyrl"}


def script_prompt(language: str | None) -> str | None:
    """A decoder seed that pins the output script, for a caller that opts in.

    ``uz`` gives Latin, ``uz-Cyrl`` Cyrillic. NOT applied automatically — see the note
    above: a seed the model cannot ground in the audio comes back as the transcript.
    """
    if not language:
        return None
    parts = language.strip().lower().replace("_", "-").split("-")
    base = parts[0]
    script = parts[1] if len(parts) > 1 else _DEFAULT_SCRIPTS.get(base)
    return _SCRIPT_SAMPLES.get((base, script or ""))


def normalize_language(language: str | None) -> str | None:
    """Validate a language code, or raise with the codes that would have worked.

    Whisper takes ISO-639-1 codes, not names: ``--lang urdu`` used to reach the decoder
    and come back as a wall of 100 codes under "check the file is valid audio", which
    blames the file for a typo in a flag.
    """
    if language is None:
        return None
    code = language.strip().lower().replace("_", "-")
    if not code:
        return None
    if code in _WHISPER_LANGUAGE_CODES:
        return code
    base = code.split("-", 1)[0]  # "en-GB" -> "en"
    if base in _WHISPER_LANGUAGE_CODES:
        return base
    guess = _NAMED_LANGUAGES.get(code)
    raise TranscriptionError(
        f"Unknown language code: {language!r}." + (f" Did you mean '{guess}'?" if guess else ""),
        hint="Use an ISO-639-1 code such as en, es, ur, ar, hi, zh, ru, uz — "
        "or leave --lang off to detect it automatically.",
    )


def _resolve_engine(model_size: str, language: str | None = None) -> tuple[str, str]:
    """Map a model name to an ``(engine, identifier)`` pair.

    ``auto`` resolves to Parakeet when it is importable *and* can read ``language``,
    else Whisper — Parakeet covers 25 European languages and silently transliterates
    anything else. An explicit Parakeet alias stays Parakeet (and fails loudly later if
    it cannot load), so an explicit choice is never silently downgraded.
    """
    if model_size == "auto":
        if _parakeet_available() and (language is None or language in PARAKEET_LANGUAGES):
            return "parakeet", PARAKEET_MODELS["parakeet"]
        if language in LOW_RESOURCE_LANGUAGES:
            return "whisper", LOW_RESOURCE_WHISPER
        return "whisper", DEFAULT_WHISPER
    if model_size in PARAKEET_MODELS:
        repo = os.environ.get("HEARSAY_PARAKEET_MODEL") or PARAKEET_MODELS[model_size]
        return "parakeet", repo
    if model_size in WHISPER_SIZES:
        return "whisper", model_size
    if _is_custom_whisper(model_size):
        # faster-whisper accepts a CTranslate2 repo id or a local directory wherever it
        # accepts a size. Passing those through is what makes low-resource languages
        # usable at all: stock Whisper saw 0.3 hours of Uzbek and scores ~90% WER on
        # FLEURS uz, while community fine-tunes trained on hundreds of hours reach single
        # digits. hearsay ships no such model and cannot vouch for one — it just stops
        # standing in the way of using it.
        return "whisper", model_size
    raise TranscriptionError(
        f"Unknown transcription model: {model_size!r}",
        hint=f"Choose one of: auto, {', '.join(PARAKEET_MODELS)}, {', '.join(WHISPER_SIZES)} "
        "— or pass a CTranslate2 Whisper model: a Hugging Face id like "
        "'org/model-ct2', or a path to a converted local directory.",
    )


def _is_custom_whisper(model_size: str) -> bool:
    """True when ``model_size`` names a CTranslate2 model rather than a built-in size.

    A Hugging Face id (``org/name``) or an existing local directory. Anything else is a
    typo and should be reported as one rather than handed to the loader.
    """
    if "/" in model_size or "\\" in model_size:
        return True
    return Path(model_size).is_dir()


def resolve_method(model_size: str) -> str:
    """The ``method`` label a transcription with ``model_size`` will report.

    Resolves ``auto`` the same way the engine picker does, so a caller can reason
    about the engine actually about to run (e.g. to warn that a checkpoint is too
    small for reliable word alignment) before paying for a transcription.
    """
    engine, identifier = _resolve_engine(model_size)
    # Parakeet labels itself with the bare repo name, Whisper with its size.
    return identifier.split("/")[-1] if engine == "parakeet" else f"whisper-{identifier}"


def _parakeet_available() -> bool:
    """True when parakeet-mlx can be imported (Apple Silicon + extra installed)."""
    from importlib.util import find_spec

    try:
        return find_spec("parakeet_mlx") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


# --- Parakeet (MLX) backend ----------------------------------------------


def _transcribe_parakeet(
    path: Path,
    *,
    repo_id: str,
    language: str | None,
    local_files_only: bool,
    word_timestamps: bool,
    on_progress: ProgressCallback | None,
) -> TranscriptionResult:
    """Transcribe via NVIDIA Parakeet on MLX (Apple Silicon GPU/Neural Engine)."""
    model = _load_parakeet(repo_id, local_files_only=local_files_only)
    sample_rate = model.preprocessor_config.sample_rate
    audio_total_s = 0.0  # true media length, learned from the chunk callback

    def _chunk_cb(end_samples: float, total_samples: float) -> None:
        nonlocal audio_total_s
        if total_samples > 0:
            audio_total_s = total_samples / sample_rate
        if on_progress is not None and total_samples > 0:
            on_progress(min(end_samples, total_samples) / sample_rate, total_samples / sample_rate)

    try:
        result = model.transcribe(
            str(path),
            chunk_duration=_PARAKEET_CHUNK_S,
            chunk_callback=_chunk_cb,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Could not transcribe {path.name}: {exc}",
            hint="Check the file is a valid audio/video file and ffmpeg is installed.",
        ) from exc

    segments: list[Segment] = []
    for sent in result.sentences:
        text = sent.text.strip()
        if not text:
            continue
        start_s = max(0.0, float(sent.start))
        end_s = max(float(sent.end), start_s)
        segments.append(Segment(text=text, start_s=start_s, end_s=end_s))

    # Prefer the true media length (from the chunk callback) so trailing silence
    # or a speechless file still reports the real duration — matching the Whisper
    # path. The callback doesn't fire for clips <= one chunk (120s), so fall back
    # to the last spoken segment's end there.
    last_seg_end = segments[-1].end_s if segments else 0.0
    duration_s = max(audio_total_s, last_seg_end)
    if on_progress is not None:
        on_progress(duration_s, duration_s)
    label = repo_id.split("/")[-1]
    words: list[Word] | None = None
    if word_timestamps:
        from hearsay.dataset.words import words_from_parakeet

        words = words_from_parakeet(result.tokens)
    return TranscriptionResult(
        segments=segments,
        # Parakeet's result carries no detected language, and the multilingual model
        # covers 25 European languages — several non-Latin. Reporting "en" on an
        # unlabelled Russian or Greek recording would be a confident lie, and the
        # dataset filters trust this value to pick a target script: they would then
        # drop every clip as wrong-script. "und" is the honest answer, and it makes
        # the script filter stand down rather than mis-fire. Pass --lang to label it.
        language=language or UNKNOWN_LANGUAGE,
        duration_s=duration_s,
        model_size=label,
        method=label,
        words=words,
    )


def _load_parakeet(repo_id: str, *, local_files_only: bool):
    """Load a Parakeet MLX model, mapping failures to friendly errors."""
    try:
        from parakeet_mlx import from_pretrained
    except ImportError as exc:
        raise TranscriptionError(
            "parakeet-mlx is not installed.",
            hint=(
                "Install the fast Apple-Silicon engine with "
                'uv tool install "hearsay[parakeet]" (macOS arm64), '
                "or use --model small for CPU Whisper."
            ),
        ) from exc
    # parakeet-mlx has no offline-only switch; force Hugging Face into offline
    # mode so a cached model loads without any network round-trip. The env var
    # alone is read at import time, so we also flip the live module constant.
    _ensure_download_timeout()
    with _hf_offline(local_files_only):
        try:
            return from_pretrained(repo_id)
        except Exception as exc:
            raise TranscriptionError(
                f"Could not load the Parakeet model '{repo_id}': {exc}",
                hint=(
                    "The model downloads once (~2.5GB). Check your network on first "
                    "use, then it is cached for offline runs."
                ),
            ) from exc


@contextmanager
def _hf_offline(enabled: bool):
    """Temporarily force ``huggingface_hub`` offline when ``enabled``.

    The ``HF_HUB_OFFLINE`` env var is read into a module constant at import
    time, so setting it now is too late if hf_hub is already imported — we flip
    the live constant as well, then restore both on exit.
    """
    if not enabled:
        yield
        return
    prior_env = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from huggingface_hub import constants

        prior_const = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = True
    except Exception:  # pragma: no cover - hf_hub always present with parakeet
        constants = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if prior_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior_env
        if constants is not None:
            constants.HF_HUB_OFFLINE = prior_const


# Slow links need more than huggingface_hub's 10s default read-timeout for the
# multi-GB Parakeet/Whisper weights, or a single chunk read raises "The read
# operation timed out" mid-download. Raise the floor to 60s while honoring any
# value the user already set (via env or the live constant).
_DOWNLOAD_TIMEOUT_FLOOR = "60"


def _ensure_download_timeout() -> None:
    """Bump huggingface_hub's download read-timeout floor for large weights.

    Sets the env var (covers fresh/forked subprocesses and the not-yet-imported
    case) and, since the constant is frozen from the env at import time but read
    live at download time, also bumps the live constant — but only when it is
    still at the default, so an explicit user value is never lowered. Touches
    nothing offline-related, so the ``local_files_only`` path stays network-free.
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", _DOWNLOAD_TIMEOUT_FLOOR)
    try:
        from huggingface_hub import constants

        default = getattr(constants, "DEFAULT_DOWNLOAD_TIMEOUT", 10)
        if default >= constants.HF_HUB_DOWNLOAD_TIMEOUT:
            constants.HF_HUB_DOWNLOAD_TIMEOUT = int(_DOWNLOAD_TIMEOUT_FLOOR)
    except Exception:  # pragma: no cover - defensive; hf_hub ships with both backends
        return


# --- Whisper (faster-whisper) backend ------------------------------------


def _transcribe_whisper(
    path: Path,
    *,
    model_size: str,
    language: str | None,
    local_files_only: bool,
    vad_filter: bool,
    word_timestamps: bool,
    on_progress: ProgressCallback | None,
    prompt: str | None = None,
) -> TranscriptionResult:
    """Transcribe via faster-whisper on CPU (int8)."""
    model = _load_whisper(model_size, local_files_only=local_files_only)
    # whisper's segment iterator is lazy — the actual decode happens as we
    # consume it, so live progress is reported here. Only this loop can fail on
    # a bad file; Segment construction is done afterward so a model invariant
    # slip (e.g. end < start) surfaces honestly, not as a bogus "bad file".
    raw: list[tuple[str, float, float]] = []
    raw_words: list = []  # faster-whisper Word objects, only when word_timestamps
    try:
        segment_iter, info = model.transcribe(
            str(path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            initial_prompt=prompt,
        )
        for seg in segment_iter:
            raw.append((seg.text, seg.start, seg.end))
            if word_timestamps and seg.words:
                raw_words.extend(seg.words)
            if on_progress is not None:
                on_progress(min(seg.end, info.duration), info.duration)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not transcribe {path.name}: {exc}",
            hint="Check the file is a valid audio/video file and ffmpeg is installed.",
        ) from exc

    segments: list[Segment] = []
    for text, start, end in raw:
        text = text.strip()
        if not text:
            continue
        start_s = max(0.0, start)
        end_s = max(end, start_s)  # whisper occasionally yields end < start
        segments.append(Segment(text=text, start_s=start_s, end_s=end_s))
    if on_progress is not None:
        on_progress(info.duration, info.duration)
    words: list[Word] | None = None
    if word_timestamps:
        from hearsay.dataset.words import words_from_whisper

        words = words_from_whisper(raw_words)
    return TranscriptionResult(
        segments=segments,
        language=info.language or "en",
        duration_s=float(info.duration),
        model_size=model_size,
        method=f"whisper-{model_size}",
        words=words,
    )


def _load_whisper(model_size: str, *, local_files_only: bool):
    """Load a faster-whisper model on CPU, mapping failures to friendly errors."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TranscriptionError(
            "faster-whisper is not installed.",
            hint="Reinstall hearsay so transcription support is available.",
        ) from exc
    _ensure_download_timeout()
    try:
        return WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load the '{model_size}' whisper model: {exc}",
            hint=(
                "The model downloads once (~tens of MB to ~1.5GB). Check your "
                "network on first use, then it is cached for offline runs."
            ),
        ) from exc
