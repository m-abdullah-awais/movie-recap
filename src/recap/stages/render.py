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
import re
import shutil
import time
import wave
from pathlib import Path

from .. import ffmpeg, probe
from ..cache import Cache, StageOutcome, atomic_path, read_json, run_stage, write_json
from ..config import OUTPUT_DIR, Settings

STAGE = "render"
VERSION = 9  # bumped: audio muted to narration only, and level normalised

FINAL_FILE = "final.mp4"
# Written here first, then moved into place. See the note in run(). The .mp4
# extension is kept because ffmpeg picks the output container from it, and a
# name ending in .part makes it refuse to choose a muxer at all.
FINAL_TMP = "final.partial.mp4"
SUBTITLE_FILE = "subtitle.srt"
CLIPLIST_FILE = "clips.concat.txt"
NARRATION_WAV = "narration_track.wav"
# Named by sample rate. The gap must match the narration exactly, and a file
# cached from a run with a different voice would silently be the wrong length.
SILENCE_WAV_FMT = "gap_{rate}hz.wav"

PARAM_NAMES = (
    "render_height", "render_crf", "qsv_quality", "allow_qsv", "copy_video",
    "mute_source_audio", "loudness_lufs", "duck_threshold", "duck_ratio",
    "narration_gain", "source_gain", "publish_subtitles",
)


_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9 ._-]+")


def safe_name(text: str, fallback: str = "recap") -> str:
    """A filename-safe version of a film title."""
    cleaned = _UNSAFE_NAME_RE.sub("", str(text or "")).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:90] or fallback


def publish(
    final: Path, subtitles: Path, title: str, with_subtitles: bool = False
) -> tuple[Path, Path | None]:
    """Place the finished video in output/ under a timestamped name.

    A hard link is used where the filesystem allows it, so a 300 MB video is not
    copied twice onto a disk that is already near full. The link is a real second
    directory entry, so clearing the cache later leaves the published file
    intact, which is the behaviour worth having.

    The name carries a timestamp so a re-run never overwrites a render worth
    keeping. That only holds because the render writes a new file and moves it
    into place rather than overwriting the previous one: hard links to a file
    that is truncated and rewritten all change together, which would make these
    timestamps look like history while being a single mutable file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{safe_name(title)} - {stamp}"

    video = OUTPUT_DIR / f"{stem}.mp4"
    caption = OUTPUT_DIR / f"{stem}.srt" if with_subtitles else None

    # A subtitle file sharing the video's name is picked up and displayed
    # automatically by most players, which is subtitles on screen whether or not
    # anything was burned in. It is left in the cache instead unless asked for.
    pairs = [(final, video)]
    if caption is not None:
        pairs.append((subtitles, caption))

    for source, destination in pairs:
        if not source.is_file():
            continue
        destination.unlink(missing_ok=True)
        try:
            os.link(source, destination)
        except (OSError, NotImplementedError):
            # Different volume, or a filesystem without hard links.
            shutil.copy2(source, destination)

    return video, caption


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


def narration_format(cache: Cache, timeline: list[dict]) -> tuple[int, int]:
    """Sample rate and channel count of the spoken lines.

    Read from the audio itself rather than taken from a setting. The concat
    demuxer adopts the parameters of the first file it opens and does not
    resample the rest, so a silence gap generated at a different rate plays at
    the wrong length. That happened here: the gap was produced at the 16 kHz rate
    used for speech recognition input while the speech engine writes its own, so
    every 0.35 second gap ran for 0.254 seconds and the track finished 7.3
    seconds short of what every other stage believed.
    """
    for entry in timeline:
        wav = cache.dir / str(entry.get("wav") or "")
        if not wav.is_file():
            continue
        try:
            with wave.open(str(wav), "rb") as handle:
                return handle.getframerate(), handle.getnchannels()
        except (wave.Error, OSError):
            continue
    return 22050, 1


def build_narration_track(
    cache: Cache, timeline: list[dict], gap_s: float, settings: Settings, quiet: bool
) -> Path:
    """Join the spoken lines into one track, with a fixed gap between them.

    Built with the concat demuxer over the per-line wav files and a generated
    silence file, rather than a filter graph with one input per line. A feature
    length recap has enough lines that the filter graph approach becomes
    unwieldy, and this keeps the command the same size regardless.
    """
    rate, channels = narration_format(cache, timeline)
    layout = "mono" if channels < 2 else "stereo"
    silence = cache.path(SILENCE_WAV_FMT.format(rate=rate))
    if not silence.is_file() or silence.stat().st_size == 0:
        ffmpeg.run([
            ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"anullsrc=r={rate}:cl={layout}",
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
        # ffmpeg runs with its working directory set to the cache folder, so the
        # path is written relative to that where possible and fully resolved
        # otherwise. Writing whatever as_posix happens to give only works while
        # the cache root is absolute, which is not something to rely on.
        try:
            listed = wav.resolve().relative_to(cache.dir.resolve()).as_posix()
        except ValueError:
            listed = wav.resolve().as_posix()
        lines.append(f"file '{listed}'")
        if position < len(timeline) - 1:
            lines.append(f"file '{silence.name}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    target = cache.path(NARRATION_WAV)
    # Run from the cache directory so the relative names in the list resolve.
    ffmpeg.run([
        ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", listing.name,
        # Output at the narration's own rate, so nothing is resampled and the
        # joined track is exactly as long as the pieces that went into it.
        "-c:a", "pcm_s16le", "-ac", str(channels), "-ar", str(rate),
        target.name,
    ], cwd=cache.dir, timeout=900)

    produced = _num(
        ffmpeg.probe(target).get("format", {}).get("duration")
    )
    expected = sum(_num(e.get("spoken_s")) or _num(e.get("seconds")) for e in timeline)
    expected += gap_s * max(0, len(timeline) - 1)
    if abs(produced - expected) > 0.5 and not quiet:
        # Worth saying out loud. A mismatch here silently shortens the finished
        # video, because the render trims to whichever stream is shorter.
        print(f"  warning: narration track is {produced:.2f}s but the timeline "
              f"expects {expected:.2f}s")
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
    """The finished audio track.

    By default the film's own sound is dropped and only the narrator is heard.
    Ducking the original underneath left its dialogue and music audible, which
    competes with the narration rather than supporting it.

    With ducking enabled instead, the narration is split: one copy is mixed in,
    the other keys the compressor. Without that split the sidechain input would
    be consumed and the mix would lose the narration entirely.
    """
    # Even out the level to a broadcast style target. Applied to whichever path
    # is taken, because narration alone is quiet and a mix is uneven.
    loudness = (f"loudnorm=I={settings.loudness_lufs}:TP=-1.5:LRA=11,"
                if settings.loudness_lufs else "")

    if settings.mute_source_audio:
        # The film's audio stream is simply never referenced, so nothing from it
        # reaches the output. Resampled here because the narration is mono at the
        # voice's own rate and the output is 48 kHz stereo.
        return (
            f"[1:a]volume={settings.narration_gain},{loudness}"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[mix]"
        )

    return (
        f"[1:a]volume={settings.narration_gain}[nar];"
        f"[nar]asplit=2[narmix][narkey];"
        f"[0:a:{ordinal}]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"volume={settings.source_gain}[src];"
        f"[src][narkey]sidechaincompress="
        f"threshold={settings.duck_threshold}:ratio={settings.duck_ratio}"
        f":attack=5:release=300[ducked];"
        f"[ducked][narmix]amix=inputs=2:duration=longest:normalize=0,"
        f"{loudness}aformat=sample_fmts=fltp:sample_rates=48000[mix]"
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
    # Same reasoning as stage 8: chain the upstream key so a new edit list always
    # produces a new render rather than serving the previous video.
    select_meta = cache.path("select.meta.json")
    if select_meta.is_file():
        try:
            params["select_key"] = read_json(select_meta).get("key")
        except Exception:  # noqa: BLE001
            params["select_key"] = None
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
        if not quiet:
            if settings.mute_source_audio:
                print("  film audio muted, narration only")
            else:
                chosen = probe.select_audio(probe_data, settings)
                if chosen is not None:
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
            FINAL_TMP,
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
                "-movflags", "+faststart", "-shortest", FINAL_TMP,
            ]
            del tail
            ffmpeg.run_with_progress(
                rebuilt, cwd=cache.dir, total_seconds=expected,
                on_progress=None if quiet else report,
            )
            use_qsv = False

        if not quiet:
            print("\r" + " " * 70 + "\r", end="")

        produced = cache.path(FINAL_TMP)
        if not produced.is_file() or produced.stat().st_size == 0:
            raise RuntimeError("ffmpeg reported success but wrote no video")

        # Replacing the file gives it a new inode. ffmpeg writing the final
        # name directly would truncate and reuse the existing one, and since
        # published copies are hard links to it, every previously published
        # render would silently change content along with it. That makes the
        # timestamped names look like history while being one mutable file.
        final = cache.path(FINAL_FILE)
        os.replace(produced, final)
        final_probe = ffmpeg.probe(final)
        video = (ffmpeg.streams(final_probe, "video") or [None])[0]
        duration = ffmpeg.duration_seconds(final_probe)

        script_path = cache.path("script.json")
        title = cache.source.stem
        if script_path.is_file():
            try:
                title = read_json(script_path).get("title") or title
            except Exception:  # noqa: BLE001 - a missing title is not a failure
                pass
        published, published_srt = publish(
            final, cache.path(SUBTITLE_FILE), title, settings.publish_subtitles
        )
        if not quiet:
            print(f"  published to output\\{published.name}")
            if published_srt is None:
                print("  no subtitle file alongside it, so nothing appears on screen")

        return {
            "final": FINAL_FILE,
            "subtitles": SUBTITLE_FILE,
            "published": str(published.relative_to(published.parents[1])),
            "published_srt": (
                str(published_srt.relative_to(published_srt.parents[1]))
                if published_srt is not None else None
            ),
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
