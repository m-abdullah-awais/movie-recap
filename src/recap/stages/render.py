"""Stage 9, the final render.

One ffmpeg invocation turns the edit decision list into a finished video. Clips
are read straight from the original film through the concat demuxer using an
in point and an out point per clip, so no intermediate files are written.

The narration sits over the film's own audio, which is ducked underneath it by a
sidechain compressor keyed on the narration itself. Subtitles are written as a
sidecar file rather than burned in, so they stay optional.

A note on stream copying. Copying the video stream is much faster, but a copy can
only begin on a keyframe, and the clip boundaries here come from shot detection
and narration timing, which fall wherever they fall. Copying would therefore
either shift every clip to the nearest earlier keyframe or produce corrupt
leading frames. The default is a hardware re-encode, which is frame accurate.
``--copy-video`` is available for when speed matters more than exact cuts.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import ffmpeg, probe
from ..cache import Cache, StageOutcome, atomic_path, read_json, run_stage, write_json
from ..config import Settings

STAGE = "render"
VERSION = 2  # bumped: subtitles timed to speech, not to footage

FINAL_FILE = "final.mp4"
SUBTITLE_FILE = "subtitle.srt"
CLIPLIST_FILE = "clips.concat.txt"
NARRATION_WAV = "narration_track.wav"
SILENCE_WAV = "gap.wav"

PARAM_NAMES = (
    "render_height", "render_crf", "qsv_quality", "allow_qsv", "copy_video",
    "duck_threshold", "duck_ratio", "narration_gain", "source_gain",
)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_subtitles(target: Path, timeline: list[dict], gap_s: float) -> int:
    """Sidecar subtitles carrying the narration, not the film's dialogue.

    Timed to the spoken duration, not to the footage allotted to the line. The
    footage also covers the silence that follows, so using it would hold each
    caption on screen through the gap and butt it against the next one.
    """
    blocks = []
    for position, entry in enumerate(timeline, 1):
        start = _num(entry.get("start_s"))
        spoken = _num(entry.get("spoken_s")) or _num(entry.get("seconds"))
        end = start + spoken
        text = str(entry.get("narration") or "").strip()
        if not text:
            continue
        blocks.append(
            f"{position}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n"
        )
    with atomic_path(target) as tmp:
        tmp.write_text("\n".join(blocks), encoding="utf-8")
    return len(blocks)


def write_clip_list(target: Path, source: Path, timeline: list[dict]) -> int:
    """Concat demuxer script reading every clip out of the original film.

    Single quotes in the path are escaped the way the demuxer expects, and the
    path is written once per clip because the demuxer requires it.
    """
    absolute = str(source.resolve()).replace("\\", "/").replace("'", "'\\''")
    lines = ["ffconcat version 1.0"]
    count = 0
    for entry in timeline:
        for clip in entry.get("clips") or []:
            start = _num(clip.get("src_start"))
            end = _num(clip.get("src_end"))
            if end <= start:
                continue
            lines.append(f"file '{absolute}'")
            lines.append(f"inpoint {start:.3f}")
            lines.append(f"outpoint {end:.3f}")
            count += 1
    with atomic_path(target) as tmp:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def build_narration_track(
    cache: Cache, timeline: list[dict], gap_s: float, settings: Settings, quiet: bool
) -> Path:
    """Join the spoken lines into one track, with a fixed gap between them.

    Built with the concat demuxer over the per-line wav files and a generated
    silence file, rather than a filter graph with one input per line. A feature
    length recap has enough lines that the filter graph approach becomes
    unwieldy, and this keeps the command the same size regardless.
    """
    silence = cache.path(SILENCE_WAV)
    if not silence.is_file() or silence.stat().st_size == 0:
        ffmpeg.run([
            ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"anullsrc=r={settings.audio_rate}:cl=mono",
            "-t", f"{max(0.01, gap_s):.3f}",
            "-c:a", "pcm_s16le",
            str(silence),
        ], timeout=120)

    listing = cache.dir / "narration.concat.txt"
    lines = ["ffconcat version 1.0"]
    for position, entry in enumerate(timeline):
        wav = cache.dir / str(entry.get("wav") or "")
        if not wav.is_file():
            continue
        lines.append(f"file '{wav.name if wav.parent == cache.dir else wav.as_posix()}'")
        if position < len(timeline) - 1:
            lines.append(f"file '{silence.name}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    target = cache.path(NARRATION_WAV)
    # Run from the cache directory so the relative names in the list resolve.
    ffmpeg.run([
        ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", listing.name,
        "-c:a", "pcm_s16le", "-ac", "1", "-ar", str(settings.audio_rate),
        target.name,
    ], cwd=cache.dir, timeout=900)
    return target


def video_args(settings: Settings, use_qsv: bool) -> list[str]:
    if settings.copy_video:
        return ["-c:v", "copy"]
    scale = []
    if settings.render_height:
        scale = ["-vf", f"scale=-2:{settings.render_height}:flags=bicubic"]
    if use_qsv:
        return scale + [
            "-c:v", "h264_qsv",
            "-global_quality", str(settings.qsv_quality),
            "-look_ahead", "0",
            "-pix_fmt", "nv12",
        ]
    return scale + [
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", str(settings.render_crf), "-pix_fmt", "yuv420p",
    ]


def audio_ordinal(probe_data: dict, settings: Settings) -> int:
    """Which audio stream to take, counted among audio streams only.

    A dual-audio film is common, and this one has Hindi first and English
    second. Referring to the first audio stream would put the wrong language
    under the narration. The concat demuxer renumbers streams, so the selection
    has to be expressed as an ordinal among audio streams rather than as the
    original stream index.
    """
    audio_streams = ffmpeg.streams(probe_data, "audio")
    chosen = probe.select_audio(probe_data, settings)
    if chosen is None:
        return 0
    for ordinal, stream in enumerate(audio_streams):
        if stream.index == chosen.index:
            return ordinal
    return 0


def audio_filter(settings: Settings, ordinal: int) -> str:
    """Narration over the film's audio, with the film ducked underneath.

    The narration is split: one copy is mixed in, the other keys the compressor.
    Without the split the sidechain input would be consumed and the mix would
    lose the narration entirely.
    """
    return (
        f"[1:a]volume={settings.narration_gain}[nar];"
        f"[nar]asplit=2[narmix][narkey];"
        f"[0:a:{ordinal}]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"volume={settings.source_gain}[src];"
        f"[src][narkey]sidechaincompress="
        f"threshold={settings.duck_threshold}:ratio={settings.duck_ratio}"
        f":attack=5:release=300[ducked];"
        f"[ducked][narmix]amix=inputs=2:duration=longest:normalize=0[mix]"
    )


def run(
    cache: Cache,
    source: Path,
    probe_data: dict,
    settings: Settings,
    *,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    edl_path = cache.path("edl.json")
    if not edl_path.is_file():
        raise RuntimeError("no edl.json found. Run the select stage first.")

    edl = read_json(edl_path)
    timeline = edl.get("timeline") or []
    if not timeline:
        raise RuntimeError("the edit decision list is empty")

    gap_s = _num(edl.get("gap_s"), 0.35)
    params = settings.params(*PARAM_NAMES)
    params["edl_lines"] = len(timeline)
    params["edl_clips"] = sum(len(t.get("clips") or []) for t in timeline)

    def work() -> dict:
        clip_count = write_clip_list(cache.path(CLIPLIST_FILE), source, timeline)
        if clip_count == 0:
            raise RuntimeError("the edit decision list produced no usable clips")

        subtitle_count = write_subtitles(cache.path(SUBTITLE_FILE), timeline, gap_s)
        if not quiet:
            print(f"  joining {len(timeline)} spoken lines into one track")
        narration_wav = build_narration_track(cache, timeline, gap_s, settings, quiet)

        use_qsv = False
        if not settings.copy_video and settings.allow_qsv:
            use_qsv = ffmpeg.qsv_available()

        ordinal = audio_ordinal(probe_data, settings)
        chosen = probe.select_audio(probe_data, settings)
        if not quiet and chosen is not None:
            print(f"  film audio: {chosen.label} (audio stream {ordinal})")

        expected = _num(edl.get("total_seconds"))
        if not quiet:
            plan = "stream copy" if settings.copy_video else (
                "hardware encode" if use_qsv else "software encode")
            print(f"  rendering {clip_count} clips, about "
                  f"{expected / 60:.1f} min, with a {plan}")

        args = [
            ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", cache.path(CLIPLIST_FILE).name,
            "-i", narration_wav.name,
            "-filter_complex", audio_filter(settings, ordinal),
            "-map", "0:v:0", "-map", "[mix]",
            *video_args(settings, use_qsv),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            "-shortest",
            FINAL_FILE,
        ]

        def report(done: float, total: float) -> None:
            if quiet:
                return
            pct = min(100.0, done / total * 100.0) if total else 0.0
            print(f"\r  rendering {pct:5.1f}%  ({done / 60:.1f} of {total / 60:.1f} min)",
                  end="", flush=True)

        try:
            ffmpeg.run_with_progress(
                args, cwd=cache.dir, total_seconds=expected,
                on_progress=None if quiet else report,
            )
        except ffmpeg.FfmpegError:
            if settings.copy_video or not use_qsv:
                raise
            # Hardware encoding can fail on a stream the probe accepted, so the
            # software path gets one chance before the stage is called a failure.
            if not quiet:
                print("\n  hardware encode failed, retrying in software")
            args = [a for a in args]
            index = args.index("-filter_complex")
            head, tail = args[: index + 2], args[index + 2 :]
            rebuilt = head + [
                "-map", "0:v:0", "-map", "[mix]",
                *video_args(settings, use_qsv=False),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", "-shortest", FINAL_FILE,
            ]
            del tail
            ffmpeg.run_with_progress(
                rebuilt, cwd=cache.dir, total_seconds=expected,
                on_progress=None if quiet else report,
            )
            use_qsv = False

        if not quiet:
            print("\r" + " " * 70 + "\r", end="")

        final = cache.path(FINAL_FILE)
        final_probe = ffmpeg.probe(final)
        video = (ffmpeg.streams(final_probe, "video") or [None])[0]
        duration = ffmpeg.duration_seconds(final_probe)

        return {
            "final": FINAL_FILE,
            "subtitles": SUBTITLE_FILE,
            "subtitle_count": subtitle_count,
            "clip_count": clip_count,
            "line_count": len(timeline),
            "duration_s": round(duration, 2),
            "duration_min": round(duration / 60.0, 2),
            "expected_s": round(expected, 2),
            "drift_s": round(duration - expected, 2),
            "width": video.width if video else None,
            "height": video.height if video else None,
            "size_mb": round(final.stat().st_size / 1e6, 1),
            "encoder": "copy" if settings.copy_video else ("h264_qsv" if use_qsv else "libx264"),
            "degraded": None,
        }

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('duration_min', 0)} min",
            f"{meta.get('width')}x{meta.get('height')}",
            f"{meta.get('size_mb', 0)}MB",
            str(meta.get("encoder")),
            f"{meta.get('clip_count', 0)} clips",
        ]
        drift = _num(meta.get("drift_s"))
        if abs(drift) > 1.0:
            bits.append(f"drift {drift:+.1f}s")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, [FINAL_FILE, SUBTITLE_FILE], work,
        force=force, summarize=summarize,
    )
