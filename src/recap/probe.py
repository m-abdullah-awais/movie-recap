"""Stream selection and encoder capability probing.

Inputs are arbitrary containers and codecs, so nothing here assumes a layout.
Both the subtitle track and the working ffmpeg command are discovered per file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import ffmpeg
from .config import BITMAP_SUB_CODECS, TEXT_SUB_CODECS, Settings

# Track titles that indicate something other than the main dialogue.
_BAD_TITLE_HINTS = (
    "commentary", "director", "cast", "lyrics", "song", "karaoke",
    "signs", "forced", "descriptive", "description",
)


def _title_penalty(title: str | None) -> int:
    if not title:
        return 0
    lowered = title.lower()
    return -80 if any(hint in lowered for hint in _BAD_TITLE_HINTS) else 0


@dataclass
class SubtitleCandidate:
    stream: ffmpeg.StreamInfo
    score: int
    usable: bool
    kind: str  # "text", "bitmap", or "unknown"
    reason: str

    def describe(self) -> dict:
        return {
            "index": self.stream.index,
            "codec": self.stream.codec,
            "language": self.stream.language,
            "title": self.stream.title,
            "default": self.stream.default,
            "forced": self.stream.forced,
            "kind": self.kind,
            "usable": self.usable,
            "reason": self.reason,
            "score": self.score,
        }


def classify_subtitles(probe_data: dict, settings: Settings) -> list[SubtitleCandidate]:
    """Rank every subtitle stream, best first.

    Bitmap formats are recorded but never marked usable. Extracting them yields
    images rather than text, so a film whose only subtitles are Presentation
    Graphic Stream must fall through to speech recognition instead of producing a
    transcript full of nothing.
    """
    candidates: list[SubtitleCandidate] = []
    for stream in ffmpeg.streams(probe_data, "subtitle"):
        codec = stream.codec.lower()
        if codec in TEXT_SUB_CODECS:
            kind, usable, reason = "text", True, "text subtitle codec"
        elif codec in BITMAP_SUB_CODECS:
            kind, usable, reason = "bitmap", False, "bitmap subtitles carry no text"
        else:
            kind, usable, reason = "unknown", False, f"unrecognised subtitle codec {codec}"

        score = 0
        if stream.language in settings.preferred_langs:
            score += 100
        elif stream.language is None:
            score += 20  # untagged tracks are usually the main one
        if stream.default:
            score += 15
        if stream.forced:
            score -= 90  # forced tracks hold only foreign-language lines
        score += _title_penalty(stream.title)

        frames = (stream.raw.get("tags") or {}).get("NUMBER_OF_FRAMES")
        try:
            score += min(int(frames) // 200, 25)
        except (TypeError, ValueError):
            pass

        candidates.append(SubtitleCandidate(stream, score, usable, kind, reason))

    candidates.sort(key=lambda c: (c.usable, c.score), reverse=True)
    return candidates


def select_audio(probe_data: dict, settings: Settings) -> ffmpeg.StreamInfo | None:
    """Pick the audio track most likely to carry the main dialogue.

    More channels scores higher because a surround mix keeps dialogue on the
    centre channel, and a mono downmix of it is clearer than a stereo music and
    effects track.
    """
    best: tuple[int, ffmpeg.StreamInfo] | None = None
    for stream in ffmpeg.streams(probe_data, "audio"):
        score = 0
        if stream.language in settings.preferred_langs:
            score += 100
        elif stream.language is None:
            score += 20
        if stream.default:
            score += 15
        score += min(stream.channels or 0, 8) * 3
        score += _title_penalty(stream.title)
        if best is None or score > best[0]:
            best = (score, stream)
    return best[1] if best else None


@dataclass
class ProxyPlan:
    """One rung of the encoder ladder."""

    name: str
    input_args: list[str] = field(default_factory=list)
    video_args: list[str] = field(default_factory=list)
    note: str = ""


def encoder_ladder(settings: Settings) -> list[ProxyPlan]:
    """Candidate commands in preference order.

    Hardware decode is requested with an ``nv12`` output format so frames are
    downloaded to system memory automatically and the filter graph stays in
    software. Mixing Quick Sync surfaces with ``hwdownload`` mid graph is fragile
    across driver versions, and filtering at 480p costs almost nothing anyway.
    The expensive part, decoding full resolution video, is what gets accelerated.
    """
    qsv_decode = ["-hwaccel", "qsv", "-hwaccel_output_format", "nv12"]
    x264 = [
        "-c:v", "libx264",
        "-preset", settings.x264_preset,
        "-crf", str(settings.x264_crf),
        "-pix_fmt", "yuv420p",
    ]
    qsv_encode = [
        "-c:v", "h264_qsv",
        "-global_quality", str(settings.qsv_quality),
        "-look_ahead", "0",
        "-pix_fmt", "nv12",
    ]

    plans: list[ProxyPlan] = []
    if settings.allow_qsv:
        plans.append(ProxyPlan("qsv-decode+qsv-encode", qsv_decode, qsv_encode,
                               "hardware decode and hardware encode"))
        plans.append(ProxyPlan("qsv-decode+x264", qsv_decode, x264,
                               "hardware decode, software encode"))
    plans.append(ProxyPlan("sw-decode+x264", [], x264, "software decode and encode"))
    return plans


def pick_plan(
    source: Path,
    plans: list[ProxyPlan],
    build: Callable[[ProxyPlan, int], list[str]],
    workdir: Path,
    *,
    probe_seconds: int = 20,
) -> tuple[ProxyPlan, list[dict]]:
    """Try each rung against the real input and return the first that works.

    Quick Sync decode support varies by codec, so capability is established by
    running a short real encode rather than by inspecting the encoder list. The
    cost is a few seconds and it removes an entire class of mid-run failure on a
    stage that otherwise takes minutes.
    """
    attempts: list[dict] = []
    for plan in plans:
        args = build(plan, probe_seconds)
        try:
            ffmpeg.run(args, cwd=workdir, timeout=300)
        except Exception as exc:  # noqa: BLE001 - any failure just moves to the next rung
            attempts.append({"plan": plan.name, "ok": False, "error": str(exc).split("\n")[0]})
            continue
        attempts.append({"plan": plan.name, "ok": True})
        return plan, attempts

    raise RuntimeError(
        "no working ffmpeg configuration was found for this file. Attempts: "
        + "; ".join(f"{a['plan']}: {a.get('error', 'ok')}" for a in attempts)
    )
