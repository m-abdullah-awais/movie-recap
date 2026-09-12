"""Thin subprocess wrappers around ffmpeg and ffprobe.

MoviePy is deliberately not used anywhere in this project. It is slow and cannot
stream-copy, which the render stage depends on.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class FfmpegError(RuntimeError):
    """A ffmpeg or ffprobe invocation failed. Carries the tail of stderr, since
    ffmpeg puts the actual reason in the last few lines."""

    def __init__(self, message: str, args: list[str], stderr: str, returncode: int):
        self.cmd_args = args
        self.stderr = stderr
        self.returncode = returncode
        tail = "\n".join(stderr.strip().splitlines()[-6:])
        super().__init__(f"{message} (exit {returncode})\n{tail}")


class MissingBinary(RuntimeError):
    pass


def _resolve(name: str, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        if not Path(override).is_file():
            raise MissingBinary(f"{env_var} points at {override}, which is not a file")
        return override
    found = shutil.which(name)
    if not found:
        raise MissingBinary(
            f"{name} was not found on PATH. Install it or set {env_var} to its full path."
        )
    return found


def ffmpeg_bin() -> str:
    return _resolve("ffmpeg", "FFMPEG")


def ffprobe_bin() -> str:
    return _resolve("ffprobe", "FFPROBE")


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing both streams as text.

    ``cwd`` matters for ffmpeg calls that use ``metadata=print:file=``. Filter
    graph arguments treat ``:`` as an option separator and ``\\`` as an escape,
    so a Windows absolute path inside a filter would need double escaping.
    Running with ``cwd`` set to the destination folder lets the filter use a bare
    relative filename and sidesteps that entirely.
    """
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise FfmpegError(
            f"{Path(args[0]).name} failed", args, proc.stderr or "", proc.returncode
        )
    return proc


def run_with_progress(
    args: list[str],
    *,
    cwd: Path | None = None,
    total_seconds: float = 0.0,
    on_progress: "Callable[[float, float], None] | None" = None,
) -> None:
    """Run ffmpeg while reporting how far through the media it has reached.

    ``-progress pipe:1`` makes ffmpeg emit machine readable key/value lines on
    stdout, which is far more reliable to parse than the human ``-stats`` output.
    Worth the extra plumbing because the proxy pass takes minutes on this CPU and
    a silent terminal is indistinguishable from a hang.
    """
    full = [args[0], "-progress", "pipe:1", "-nostats", *args[1:]]
    proc = subprocess.Popen(
        full,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None

    # stderr must be drained while stdout is being read, not afterwards. Some
    # configurations emit a warning per frame, which is well over a hundred
    # thousand lines for a feature film. That fills the stderr pipe buffer,
    # ffmpeg blocks writing to it, stops producing progress on stdout, and the
    # read loop below waits forever. Only the tail is kept, since that is all an
    # error message needs.
    tail: deque[str] = deque(maxlen=40)

    def drain() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            tail.append(line.rstrip())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_ms" and on_progress:
                try:
                    done = int(value) / 1_000_000.0
                except ValueError:
                    continue
                on_progress(done, total_seconds)
    finally:
        proc.stdout.close()
        code = proc.wait()
        reader.join(timeout=5)
        proc.stderr.close()

    if code != 0:
        raise FfmpegError("ffmpeg failed", full, "\n".join(tail), code)


def probe(path: Path) -> dict:
    """Full ffprobe JSON for a media file. Container agnostic: the format is
    detected from content, never from the file extension."""
    args = [
        ffprobe_bin(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]
    proc = run(args)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FfmpegError("ffprobe returned unparseable JSON", args, proc.stdout, 0) from exc


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec: str
    codec_type: str
    language: str | None
    title: str | None
    default: bool
    forced: bool
    channels: int | None
    width: int | None
    height: int | None
    fps: float | None
    duration_s: float | None
    raw: dict

    @property
    def label(self) -> str:
        bits = [f"#{self.index}", self.codec]
        if self.language:
            bits.append(self.language)
        if self.title:
            bits.append(f'"{self.title}"')
        if self.forced:
            bits.append("forced")
        if self.default:
            bits.append("default")
        return " ".join(bits)


def _fraction(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    num, _, den = value.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    return n / d if d else None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def streams(probe_data: dict, codec_type: str | None = None) -> list[StreamInfo]:
    out: list[StreamInfo] = []
    for s in probe_data.get("streams", []):
        if codec_type and s.get("codec_type") != codec_type:
            continue
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        lang = tags.get("language") or tags.get("LANGUAGE")
        out.append(
            StreamInfo(
                index=int(s.get("index", -1)),
                codec=s.get("codec_name") or "unknown",
                codec_type=s.get("codec_type") or "unknown",
                language=lang.lower() if isinstance(lang, str) else None,
                title=tags.get("title") or tags.get("TITLE"),
                default=bool(disp.get("default")),
                forced=bool(disp.get("forced")),
                channels=s.get("channels"),
                width=s.get("width"),
                height=s.get("height"),
                fps=_fraction(s.get("avg_frame_rate")) or _fraction(s.get("r_frame_rate")),
                duration_s=_float_or_none(s.get("duration")),
                raw=s,
            )
        )
    return out


def duration_seconds(probe_data: dict) -> float:
    """Runtime in seconds, preferring the container value and falling back to the
    longest stream, since some containers omit format duration."""
    fmt = _float_or_none((probe_data.get("format") or {}).get("duration"))
    if fmt and fmt > 0:
        return fmt
    best = 0.0
    for s in probe_data.get("streams", []):
        d = _float_or_none(s.get("duration")) or 0.0
        best = max(best, d)
    return best


def start_time_seconds(probe_data: dict) -> float:
    """Container start offset. Transport streams frequently start at a non-zero
    timestamp, and later stages must subtract this to keep proxy time and source
    time aligned."""
    return _float_or_none((probe_data.get("format") or {}).get("start_time")) or 0.0


def qsv_available() -> bool:
    """True when the Quick Sync encoder actually initializes on this machine.

    The encoder being compiled into ffmpeg says nothing about whether the
    hardware and driver will accept it, so this runs a real two second encode to
    a null output rather than trusting the encoder list.
    """
    try:
        run(
            [
                ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-init_hw_device", "qsv=hw",
                "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=2",
                "-vf", "hwupload=extra_hw_frames=16,format=qsv",
                "-c:v", "h264_qsv", "-f", "null", "-",
            ],
            timeout=90,
        )
    except (FfmpegError, MissingBinary, subprocess.TimeoutExpired):
        return False
    return True
