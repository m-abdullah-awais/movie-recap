"""Stage 1, dialogue acquisition.

The AI reads the movie rather than watching it, so this stage produces the text
that all story understanding is built from. Three sources are tried in order of
cost: an embedded text subtitle track, a sidecar subtitle file, and finally
speech recognition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .. import ffmpeg, models, probe, srt
from ..cache import Cache, StageOutcome, run_stage, write_json
from ..config import MODELS_DIR, SIDECAR_SUB_EXTS, Settings

STAGE = "ingest"
VERSION = 1

TRANSCRIPT_FILE = "transcript.json"
SUBS_FILE = "dialogue.srt"

PARAM_NAMES = ("preferred_langs", "asr_model", "asr_compute", "min_coverage_ratio")


class NeedsAudio(RuntimeError):
    """Raised when the only remaining source is speech recognition and the proxy
    audio has not been produced yet.

    Speech recognition needs the 16 kHz wav that stage 2 writes, so ingest cannot
    always finish before proxy. The caller resolves this by running the proxy
    stage and calling ingest again, which costs nothing when the proxy is already
    cached.
    """


class ModelUnavailable(RuntimeError):
    """The speech model could not be loaded, usually a download failure.

    Kept distinct from a generic error so the caller can report a clear cause and
    still carry on with the stages that do not depend on dialogue.
    """


def find_sidecars(source: Path) -> list[Path]:
    """Subtitle files sitting next to the movie.

    Matches both ``movie.srt`` and language tagged variants such as
    ``movie.en.srt``, which is how most subtitle downloads are named.
    """
    found: list[Path] = []
    for candidate in sorted(source.parent.glob(f"{source.stem}*")):
        if candidate.suffix.lower() in SIDECAR_SUB_EXTS and candidate.is_file():
            found.append(candidate)

    def rank(path: Path) -> tuple[int, int]:
        lowered = path.name.lower()
        lang_bonus = 0 if any(t in lowered for t in (".en.", ".eng.", ".english.")) else 1
        return (lang_bonus, len(path.name))

    return sorted(found, key=rank)


def _extract_embedded(source: Path, stream_index: int, target: Path) -> str:
    ffmpeg.run([
        ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source.resolve()),
        "-map", f"0:{stream_index}",
        "-c:s", "srt",
        str(target),
    ], timeout=900)
    return target.read_text(encoding="utf-8", errors="replace")


def _convert_sidecar(sidecar: Path, target: Path) -> str:
    """Normalize any sidecar format to SRT so one parser covers everything."""
    if sidecar.suffix.lower() == ".srt":
        return sidecar.read_text(encoding="utf-8", errors="replace")
    ffmpeg.run([
        ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(sidecar.resolve()),
        "-c:s", "srt",
        str(target),
    ], timeout=300)
    return target.read_text(encoding="utf-8", errors="replace")


def _transcribe(wav: Path, runtime_s: float, settings: Settings, quiet: bool) -> list[srt.Cue]:
    """Speech recognition fallback.

    Greedy decoding is the right trade here because the text feeds story
    understanding rather than on-screen subtitles, and it is markedly faster on a
    low power CPU. Conditioning on previous text is disabled because it is the
    main cause of the repetition loops that ruin long transcriptions.
    """
    try:
        from faster_whisper import WhisperModel  # imported late so the CLI works without it
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed, so films without subtitles cannot be read. "
            "Run scripts\\setup.ps1 to install it into the project venv."
        ) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"  loading whisper {settings.asr_model} ({settings.asr_compute}) from .models")

    try:
        model = WhisperModel(
            settings.asr_model,
            device="cpu",
            compute_type=settings.asr_compute,
            download_root=str(MODELS_DIR),
            # ctranslate2 scales with physical cores, not hyperthreads, so half
            # the logical count is the right figure on this 4 core part.
            cpu_threads=max(1, (os.cpu_count() or 4) // 2),
        )
    except Exception as exc:  # noqa: BLE001
        # The first use of recognition downloads roughly 250 MB. A rate limit or
        # a dropped connection surfaces here as a Hugging Face error, which is
        # unreadable on its own, so it is translated into something actionable.
        raise ModelUnavailable(
            f"the {settings.asr_model} speech model could not be loaded or downloaded "
            f"into {MODELS_DIR.name}. Either retry when the connection is better, or "
            f"place a matching .srt file next to the movie to skip recognition entirely. "
            f"Underlying error: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        ) from exc

    segments, _info = model.transcribe(
        str(wav),
        language="en",
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    cues: list[srt.Cue] = []
    for segment in segments:
        text = srt.clean_text(segment.text or "")
        if text:
            cues.append(srt.Cue(float(segment.start), float(segment.end), text))
        if not quiet and runtime_s > 0 and len(cues) % 25 == 0:
            pct = min(100.0, float(segment.end) / runtime_s * 100.0)
            sys.stdout.write(f"\r  transcribing {pct:5.1f}%  ({len(cues)} cues)")
            sys.stdout.flush()
    if not quiet:
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()

    return srt.dedupe(sorted(cues, key=lambda c: c.start))


def run(
    cache: Cache,
    source: Path,
    probe_data: dict,
    settings: Settings,
    *,
    wav: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    runtime = ffmpeg.duration_seconds(probe_data)
    candidates = probe.classify_subtitles(probe_data, settings)
    params = settings.params(*PARAM_NAMES)

    def work() -> dict:
        target = cache.path(SUBS_FILE)
        attempts: list[dict] = []

        # 1. Embedded text tracks, best ranked first. A track that extracts but
        #    covers almost none of the runtime is rejected and the next is tried,
        #    which catches mislabelled forced and commentary tracks.
        for candidate in [c for c in candidates if c.usable]:
            try:
                content = _extract_embedded(source, candidate.stream.index, target)
                cues = srt.parse(content)
            except Exception as exc:  # noqa: BLE001 - fall through to the next source
                attempts.append({"source": f"embedded {candidate.stream.label}",
                                 "ok": False, "error": str(exc).split("\n")[0]})
                continue

            stats = srt.stats(cues, runtime)
            if cues and stats["coverage_ratio"] >= settings.min_coverage_ratio:
                return _payload(cache, cues, runtime, "embedded", candidates,
                                stream=candidate.describe(), attempts=attempts, quiet=quiet)
            attempts.append({
                "source": f"embedded {candidate.stream.label}",
                "ok": False,
                "error": f"coverage {stats['coverage_ratio']:.3f} below "
                         f"{settings.min_coverage_ratio}, {len(cues)} cues",
            })

        # 2. Sidecar subtitle files next to the movie.
        for sidecar in find_sidecars(source):
            try:
                cues = srt.parse(_convert_sidecar(sidecar, target))
            except Exception as exc:  # noqa: BLE001
                attempts.append({"source": f"sidecar {sidecar.name}",
                                 "ok": False, "error": str(exc).split("\n")[0]})
                continue
            stats = srt.stats(cues, runtime)
            if cues and stats["coverage_ratio"] >= settings.min_coverage_ratio:
                return _payload(cache, cues, runtime, "sidecar", candidates,
                                stream={"file": sidecar.name}, attempts=attempts, quiet=quiet)
            attempts.append({"source": f"sidecar {sidecar.name}", "ok": False,
                             "error": f"coverage {stats['coverage_ratio']:.3f} too low"})

        # 3. Speech recognition, which needs the proxy audio.
        if wav is None or not wav.is_file():
            # The model is checked before the film is read, not after. Reading a
            # film takes minutes, and a download that was going to fail fails
            # just as well beforehand. A recipient of this tool waited six
            # minutes for the read and only then saw the model could not be
            # fetched, which is six minutes spent to learn nothing.
            if not models.whisper_available(settings.asr_model):
                if not quiet:
                    print("  this film has no usable subtitles, so speech "
                          "recognition is needed")
                if not models.ensure_whisper(settings.asr_model, quiet=quiet,
                                             attempts=2):
                    raise ModelUnavailable(
                        f"the {settings.asr_model} speech model is not in "
                        f"{MODELS_DIR.name} and could not be downloaded, and this "
                        "film has no usable subtitles. Either run fetch-models "
                        "when the connection is better, or place a matching .srt "
                        "file next to the movie to skip recognition entirely."
                    )
            raise NeedsAudio("speech recognition requires the proxy audio")

        cues = _transcribe(wav, runtime, settings, quiet)
        if not cues:
            raise RuntimeError(
                "no dialogue could be obtained from subtitles or speech recognition"
            )
        return _payload(cache, cues, runtime, "asr", candidates,
                        stream={"model": settings.asr_model}, attempts=attempts, quiet=quiet)

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('origin')}",
            f"{meta.get('cue_count', 0)} cues",
            f"{meta.get('word_count', 0)} words",
            f"coverage {meta.get('coverage_ratio', 0):.2f}",
        ]
        if meta.get("low_coverage"):
            bits.append("LOW COVERAGE")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, [TRANSCRIPT_FILE], work, force=force, summarize=summarize
    )


def _payload(
    cache: Cache,
    cues: list[srt.Cue],
    runtime: float,
    origin: str,
    candidates: list[probe.SubtitleCandidate],
    *,
    stream: dict,
    attempts: list[dict],
    quiet: bool,
) -> dict:
    stats = srt.stats(cues, runtime)
    low = bool(stats["coverage_ratio"] < 0.05)
    if low and not quiet:
        print(f"  warning: dialogue covers only {stats['coverage_ratio']:.1%} of the runtime")

    write_json(cache.path(TRANSCRIPT_FILE), {
        "schema": 1,
        "source_id": cache.sid,
        "source_name": cache.source.name,
        "runtime_s": round(runtime, 3),
        "origin": origin,
        "language": "en",
        "selected": stream,
        "subtitle_streams": [c.describe() for c in candidates],
        "attempts": attempts,
        "stats": stats,
        "cues": [
            {"i": i, "start": round(c.start, 3), "end": round(c.end, 3), "text": c.text}
            for i, c in enumerate(cues)
        ],
    })
    return {
        "transcript": TRANSCRIPT_FILE,
        "origin": origin,
        "selected": stream,
        "attempts": attempts,
        "low_coverage": low,
        **{k: v for k, v in stats.items()},
    }
