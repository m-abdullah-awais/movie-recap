"""Stage 7, narration audio.

Each script segment is spoken once and measured. The measured duration is the
thing that matters: it tells stage 8 exactly how much footage to pack behind that
line, which is why no forced aligner is ever needed.

Every line is cached by the hash of its own text, so re-running after editing one
segment re-speaks only that segment, and two identical lines are synthesised once.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import models
from ..cache import Cache, StageOutcome, read_json, run_stage, write_json
from ..config import Settings

STAGE = "narrate"
VERSION = 1

NARRATION_FILE = "narration.json"
AUDIO_DIR = "audio"

PARAM_NAMES = ("narrate_gap_s", "piper_length_scale", "script_words_per_minute")


class NoVoice(RuntimeError):
    """Neither Piper nor the system voice could speak."""


def text_key(text: str, length_scale: float) -> str:
    """Cache key for one spoken line.

    The speaking rate is folded in, because the same words at a different rate
    are a different audio file with a different duration.
    """
    blob = f"{length_scale}:{text}".encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def wav_seconds(path: Path) -> float:
    """Exact duration from the wav header, no decoding needed."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except (wave.Error, OSError):
        return 0.0


class PiperSpeaker:
    """Piper text to speech, as specified for this project."""

    def __init__(self, voice: models.PiperVoice, length_scale: float):
        try:
            from piper import PiperVoice as Loader
        except ImportError as exc:
            raise NoVoice("the piper-tts package is not installed") from exc

        try:
            self._voice = Loader.load(str(voice.model), config_path=str(voice.config))
        except TypeError:
            # Older releases take the config positionally.
            self._voice = Loader.load(str(voice.model), str(voice.config))
        except Exception as exc:  # noqa: BLE001
            raise NoVoice(f"the Piper voice could not be loaded: {exc}") from exc

        self._length_scale = length_scale
        self.name = voice.name

    def speak(self, text: str, target: Path) -> bool:
        """Write one spoken line to a wav file.

        The synthesis entry point was renamed between Piper releases, so both
        spellings are tried rather than pinning to one version.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(target), "wb") as handle:
                if hasattr(self._voice, "synthesize_wav"):
                    self._voice.synthesize_wav(text, handle)
                else:
                    self._voice.synthesize(text, handle)
        except Exception:  # noqa: BLE001 - a single bad line must not stop the run
            target.unlink(missing_ok=True)
            return False
        return target.is_file() and target.stat().st_size > 0


class SystemSpeaker:
    """Windows' built-in speech synthesiser.

    A documented fallback, not a preference. It is markedly more robotic than
    Piper, but it needs no download at all, so a film can still be finished on a
    machine where the voice model cannot be fetched.
    """

    name = "windows-sapi"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise NoVoice("the system speech fallback is Windows only")

    def speak(self, text: str, target: Path) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        escaped = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{target}'); "
            f"$s.Speak('{escaped}'); $s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=120, check=True,
            )
        except (subprocess.SubprocessError, OSError):
            target.unlink(missing_ok=True)
            return False
        return target.is_file() and target.stat().st_size > 0


def make_speaker(settings: Settings, quiet: bool):
    """Piper if its voice is available, otherwise the system voice."""
    voice = models.ensure_piper_voice(quiet=quiet, attempts=2)
    if voice is not None:
        try:
            speaker = PiperSpeaker(voice, settings.piper_length_scale)
            if not quiet:
                print(f"  speaking with Piper, voice {speaker.name}")
            return speaker, None
        except NoVoice as exc:
            if not quiet:
                print(f"  Piper unavailable: {exc}")

    try:
        speaker = SystemSpeaker()
    except NoVoice as exc:
        raise NoVoice(
            "no speech synthesiser is available. Install piper-tts and let the "
            f"voice download succeed. {exc}"
        ) from exc
    if not quiet:
        print("  Piper voice unavailable, falling back to the system voice")
    return speaker, "system_voice"


def run(
    cache: Cache,
    settings: Settings,
    *,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    script_path = cache.path("script.json")
    if not script_path.is_file():
        raise RuntimeError("no script.json found. Run the script stage first.")

    script = read_json(script_path)
    segments = script.get("segments") or []
    if not segments:
        raise RuntimeError("the script has no segments to speak")

    params = settings.params(*PARAM_NAMES)
    params["segment_count"] = len(segments)
    params["text_digest"] = hashlib.blake2b(
        "\n".join(s.get("narration", "") for s in segments).encode("utf-8"),
        digest_size=8,
    ).hexdigest()

    def work() -> dict:
        speaker, degraded = make_speaker(settings, quiet)
        audio_dir = cache.dir / AUDIO_DIR
        audio_dir.mkdir(parents=True, exist_ok=True)

        planned = []
        for segment in segments:
            text = str(segment.get("narration") or "").strip()
            key = text_key(text, settings.piper_length_scale)
            planned.append({
                "segment": segment,
                "text": text,
                "wav": audio_dir / f"seg_{key}.wav",
            })

        # Identical lines share one file, so each distinct text is spoken once.
        todo = {}
        for item in planned:
            if item["text"] and not item["wav"].is_file():
                todo[item["wav"]] = item["text"]

        if not quiet:
            print(f"  speaking {len(todo)} new lines of {len(planned)} segments")

        failures: list[str] = []
        if todo:
            def say(pair) -> bool:
                target, text = pair
                return speaker.speak(text, target)

            with ThreadPoolExecutor(max_workers=max(1, settings.narrate_workers)) as pool:
                results = list(pool.map(say, todo.items()))
            for (target, _text), ok in zip(todo.items(), results):
                if not ok:
                    failures.append(target.name)

        entries = []
        cursor = 0.0
        for position, item in enumerate(planned):
            seconds = wav_seconds(item["wav"]) if item["wav"].is_file() else 0.0
            if seconds <= 0.0:
                # A line that could not be spoken is dropped rather than left as
                # a silent gap, which would desynchronise everything after it.
                continue
            segment = item["segment"]
            entries.append({
                "i": len(entries),
                "script_i": segment.get("i", position),
                "wav": f"{AUDIO_DIR}/{item['wav'].name}",
                "seconds": round(seconds, 3),
                "start_s": round(cursor, 3),
                "end_s": round(cursor + seconds, 3),
                "story_time": segment.get("story_time"),
                "spoiler_ceiling": segment.get("spoiler_ceiling"),
                "visual_query": segment.get("visual_query"),
                "narration": item["text"],
            })
            cursor += seconds + settings.narrate_gap_s

        if not entries:
            raise RuntimeError("no narration audio was produced")

        # The trailing gap after the final line is not part of the video.
        total = round(max(0.0, cursor - settings.narrate_gap_s), 3)
        spoken = sum(e["seconds"] for e in entries)
        words = sum(len(e["narration"].split()) for e in entries)

        payload = {
            "schema": 1,
            "source_id": cache.sid,
            "source_name": cache.source.name,
            "voice": getattr(speaker, "name", "unknown"),
            "gap_s": settings.narrate_gap_s,
            "total_seconds": total,
            "segments": entries,
            "stats": {
                "segment_count": len(entries),
                "dropped": len(planned) - len(entries),
                "spoken_seconds": round(spoken, 2),
                "total_seconds": total,
                "total_minutes": round(total / 60.0, 2),
                "words": words,
                "words_per_minute": round(words / max(1e-6, spoken / 60.0), 1),
                "failures": failures,
                "degraded": degraded or ("some_lines_failed" if failures else None),
            },
        }
        write_json(cache.path(NARRATION_FILE), payload)
        return {"narration": NARRATION_FILE, "voice": payload["voice"], **payload["stats"]}

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('segment_count', 0)} lines",
            f"{meta.get('total_minutes', 0)} min",
            f"{meta.get('words_per_minute', 0)} wpm",
            str(meta.get("voice", "")),
        ]
        if meta.get("dropped"):
            bits.append(f"{meta['dropped']} dropped")
        if meta.get("degraded"):
            bits.append(f"DEGRADED {meta['degraded']}")
        return ", ".join(b for b in bits if b)

    return run_stage(
        cache, STAGE, VERSION, params, [NARRATION_FILE], work,
        force=force, summarize=summarize,
    )
