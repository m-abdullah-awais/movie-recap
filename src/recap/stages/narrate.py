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
VERSION = 5  # bumped: Kokoro only, Piper removed

NARRATION_FILE = "narration.json"
AUDIO_DIR = "audio"

PARAM_NAMES = (
    "voice", "narrate_gap_s", "kokoro_speed", "script_words_per_minute",
)


class NoVoice(RuntimeError):
    """Neither Kokoro nor the system voice could speak."""


def text_key(text: str, voice: str, speed: float) -> str:
    """Cache key for one spoken line.

    The voice and the speaking rate are folded in, because the same words in a
    different voice or at a different rate are a different audio file with a
    different duration.
    """
    blob = f"{voice}:{speed}:{text}".encode("utf-8")
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


class KokoroSpeaker:
    """Kokoro text to speech.

    One model holds every voice, selected by name, so switching narrator costs
    nothing once the model is loaded. Output is float samples rather than a wav
    file, so it is written here rather than handed to the library.
    """

    def __init__(self, voice: str, assets: models.KokoroAssets, speed: float = 1.0):
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise NoVoice("the kokoro-onnx package is not installed") from exc
        try:
            self._kokoro = Kokoro(str(assets.model), str(assets.voices))
        except Exception as exc:  # noqa: BLE001
            raise NoVoice(f"the Kokoro model could not be loaded: {exc}") from exc

        self._voice = voice
        # Larger is faster here. Measured on am_michael: 0.9 gives 143 words per
        # minute, 1.0 gives 154, and 1.3 gives 190.
        self._speed = speed if speed > 0 else 1.0
        self.name = voice

    def voices(self) -> list[str]:
        try:
            return sorted(self._kokoro.get_voices())
        except Exception:  # noqa: BLE001
            return []

    def speak(self, text: str, target: Path) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            samples, rate = self._kokoro.create(
                text, voice=self._voice, speed=self._speed, lang="en-us"
            )
        except Exception:  # noqa: BLE001 - one bad line must not stop the run
            return False

        try:
            import numpy as np

            audio = np.asarray(samples, dtype=np.float32)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1.0:
                audio = audio / peak
            pcm = (audio * 32767.0).astype(np.int16)
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(rate))
                handle.writeframes(pcm.tobytes())
        except Exception:  # noqa: BLE001
            target.unlink(missing_ok=True)
            return False
        return target.is_file() and target.stat().st_size > 0


class SystemSpeaker:
    """Windows' built-in speech synthesiser.

    A documented fallback, not a preference. It is markedly more robotic than
    Kokoro, but it needs no download at all, so a film can still be finished on
    a machine where the voice model cannot be fetched.
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
    """Kokoro, falling back to the system voice.

    It falls back rather than failing, because a missing narrator should not
    throw away the stages that already succeeded.
    """
    if models.kokoro_available():
        try:
            speaker = KokoroSpeaker(
                models.voice_name(settings.voice),
                models.kokoro_paths(),
                settings.kokoro_speed,
            )
            if not quiet:
                print(f"  speaking with Kokoro, voice {speaker.name}")
            return speaker, None
        except NoVoice as exc:
            if not quiet:
                print(f"  Kokoro unavailable: {exc}")
    elif not quiet:
        print("  the Kokoro model is not in .models, falling back")

    try:
        speaker = SystemSpeaker()
    except NoVoice as exc:
        raise NoVoice(
            "no speech synthesiser is available. Put the Kokoro model in "
            f".models, or run fetch-models. {exc}"
        ) from exc
    if not quiet:
        print("  Kokoro unavailable, falling back to the system voice")
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
            key = text_key(
                text, models.voice_name(settings.voice), settings.kokoro_speed
            )
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
                # Stage 8 shows the beat this line narrates, so it needs to know
                # which beat that is. Carried through rather than looked up.
                "beat_id": segment.get("beat_id"),
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
