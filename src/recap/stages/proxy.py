"""Stage 2, the single full-film read.

One decode of the source produces the raw scene change timestamps and the 16 kHz
mono wav. Sharing that decode matters because reading a 2 hour film is the single
most expensive operation in the pipeline on this hardware, measured at about 12
minutes, and doing it twice would double the largest cost in the budget.

No full-film video proxy is written by default. Encoding one was measured to add
roughly 6 minutes on a 94 minute film while serving nothing downstream, because
stage 6 indexes only the narrowed regions the script references and stage 9
stream-copies the original. Pass ``--with-proxy`` to write one anyway, which is
useful for seeing what the pipeline saw.

The original file is otherwise untouched until the render stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .. import ffmpeg, probe
from ..cache import Cache, StageOutcome, run_stage
from ..config import Settings

STAGE = "proxy"
VERSION = 2  # bumped: bit depth normalisation and the reworked decoder ladder

PROXY_FILE = "proxy.mp4"
WAV_FILE = "proxy.wav"
SCDET_FILE = "scdet.raw.txt"

PARAM_NAMES = (
    "write_proxy_video", "proxy_height", "proxy_fps", "proxy_fps_threshold",
    "allow_qsv", "x264_preset", "x264_crf", "qsv_quality", "audio_rate",
    "detect_width", "scene_floor", "preferred_langs",
)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def build_filter_graph(settings: Settings, target_height: int, source_fps: float | None) -> str:
    """Scale once, then split into an encode branch and a detection branch.

    Detection runs on a hard downscale to ``detect_width`` because scene change
    scores are a frame difference measure that survives aggressive downsampling,
    which makes the second branch close to free.

    Detection uses a permissive floor rather than the user's threshold, and
    records each candidate cut's score. Stage 3 then filters by score. Without
    this, the threshold would be baked into the proxy pass and retuning it would
    force a full re-encode of a two hour film, which is exactly the iteration the
    cache layer exists to avoid.

    The metadata chain selects only frames that scdet flagged and prints their
    score, so the file holds one record per candidate cut instead of one per
    frame. Printing every frame would write roughly ten megabytes of text for a
    feature length film.

    The branch ends in ``nullsink`` rather than being mapped to a null output.
    Because the select filter discards every frame that is not a cut, a stretch
    of film with no cuts leaves the branch empty, and ffmpeg treats an output
    stream that received no packets as a failure. That would have wrongly failed
    the twenty second encoder probe on any film that opens on a static shot.
    Terminating the branch inside the graph removes the output stream entirely.
    """
    # format=yuv420p normalises the bit depth. A 10 bit source such as HEVC Main
    # 10 arrives as yuv420p10le, and without it the proxy would be encoded at 10
    # bit, which is slower to produce and pointless for retrieval work. Chroma
    # is deliberately kept rather than converting to gray, because it lets scdet
    # separate two scenes of similar brightness but different colour.
    detect = (
        f"scdet=t={settings.scene_floor},"
        f"metadata=mode=select:key=lavfi.scd.time,"
        f"metadata=mode=print:key=lavfi.scd.score:file={SCDET_FILE}:direct=1,"
        f"nullsink"
    )

    if not settings.write_proxy_video:
        # Lean pass. Scaling straight to the detection width is one operation
        # instead of two, and no video is encoded at all.
        #
        # The kept branch exists only to give the graph a mapped output. ffmpeg
        # rejects a filter_complex in which every branch ends in a sink, and the
        # detection branch cannot be the mapped output because its select filter
        # discards all non-cut frames, which would leave the output empty on any
        # stretch of film without a cut.
        #
        # Only the first frame is passed. Feeding every frame to the null muxer
        # made it complain about duplicate timestamps once per frame, which is
        # over a hundred thousand lines of stderr on a feature film. One frame is
        # enough to keep the output non-empty and silences it completely.
        return (
            f"[0:v]scale={settings.detect_width}:-2:flags=neighbor,format=yuv420p,"
            f"split=2[kept][pdet];"
            f"[kept]trim=end_frame=1[keep];"
            f"[pdet]{detect}"
        )

    chain = [f"scale=-2:{target_height}:flags=fast_bilinear", "format=yuv420p"]
    # Leave 23.976 and 24 fps films untouched. Only reduce genuinely high frame
    # rate sources, where the extra frames are wasted encode work. Applied only
    # in this branch, since without an encode there is nothing to save.
    if source_fps and source_fps > settings.proxy_fps_threshold:
        chain.append(f"fps={settings.proxy_fps}")
    prepared = ",".join(chain)

    return (
        f"[0:v]{prepared},split=2[penc][pdet];"
        f"[pdet]scale={settings.detect_width}:-2:flags=neighbor,{detect}"
    )


def build_command(
    source: Path,
    plan: probe.ProxyPlan,
    settings: Settings,
    *,
    target_height: int,
    source_fps: float | None,
    audio_index: int | None,
    proxy_name: str,
    wav_name: str,
    limit_seconds: int | None = None,
) -> list[str]:
    """Assemble the single fused ffmpeg invocation.

    Output filenames are deliberately bare and relative: the command is run with
    its working directory set to the cache folder so that the filter graph's
    ``file=`` argument contains no colon or backslash. A Windows absolute path
    inside a filter graph would need both characters escaped, which is a
    well known source of breakage.
    """
    args = [ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    args += plan.input_args
    if limit_seconds:
        args += ["-t", str(limit_seconds)]
    args += ["-i", str(source.resolve())]
    args += ["-filter_complex", build_filter_graph(settings, target_height, source_fps)]

    # Output 1. Either the 480p proxy, or a discarded stream that exists purely
    # to give the filter graph a mapped output.
    if settings.write_proxy_video:
        args += ["-map", "[penc]", *plan.video_args, "-an", "-sn", "-dn", proxy_name]
    else:
        # The destination is the platform null device, not "-". A dash means
        # stdout, which is already carrying the machine readable progress
        # stream, and the two collide and hang the run.
        args += ["-map", "[keep]", "-an", "-sn", "-dn", "-f", "null", os.devnull]
    # Output 2, mono speech audio for the recognition fallback.
    if audio_index is not None:
        args += [
            "-map", f"0:{audio_index}",
            "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(settings.audio_rate),
            "-c:a", "pcm_s16le",
            wav_name,
        ]
    return args


def _progress_printer(label: str):
    """In-place progress line.

    Written to stdout rather than stderr so it stays in order with the stage
    headings. Mixing the two streams interleaves wrongly as soon as output is
    piped to a file, because each is buffered independently.
    """

    def report(done: float, total: float) -> None:
        if total > 0:
            pct = min(100.0, done / total * 100.0)
            sys.stdout.write(f"\r  {label} {pct:5.1f}%  ({done/60:.1f} of {total/60:.1f} min)")
        else:
            sys.stdout.write(f"\r  {label} {done/60:.1f} min")
        sys.stdout.flush()

    return report


def run(
    cache: Cache,
    source: Path,
    probe_data: dict,
    settings: Settings,
    *,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    duration = ffmpeg.duration_seconds(probe_data)
    video_streams = ffmpeg.streams(probe_data, "video")
    if not video_streams:
        raise RuntimeError(f"{source.name} contains no video stream")

    video = video_streams[0]
    source_height = video.height or settings.proxy_height
    # Never upscale. A source already below the proxy height is left at its own
    # size, since inventing pixels costs encode time and adds no detail.
    target_height = _even(min(settings.proxy_height, source_height))
    audio = probe.select_audio(probe_data, settings)

    params = settings.params(*PARAM_NAMES)
    params["audio_index"] = audio.index if audio else None
    params["target_height"] = target_height

    artifacts = [SCDET_FILE]
    if settings.write_proxy_video:
        artifacts.append(PROXY_FILE)
    if audio:
        artifacts.append(WAV_FILE)

    def work() -> dict:
        def build(plan: probe.ProxyPlan, seconds: int) -> list[str]:
            return build_command(
                source, plan, settings,
                target_height=target_height,
                source_fps=video.fps,
                audio_index=audio.index if audio else None,
                proxy_name="probe.proxy.mp4",
                wav_name="probe.proxy.wav",
                limit_seconds=seconds,
            )

        plans = probe.encoder_ladder(settings, video.codec)
        if not quiet:
            print(f"  probing {len(plans)} encoder configuration(s) against this file")
        plan, attempts = probe.pick_plan(source, plans, build, cache.dir)
        for leftover in ("probe.proxy.mp4", "probe.proxy.wav", SCDET_FILE):
            cache.path(leftover).unlink(missing_ok=True)
        if not quiet:
            print(f"  using {plan.name} ({plan.note})")

        args = build_command(
            source, plan, settings,
            target_height=target_height,
            source_fps=video.fps,
            audio_index=audio.index if audio else None,
            proxy_name=PROXY_FILE,
            wav_name=WAV_FILE,
        )
        label = "encoding proxy" if settings.write_proxy_video else "reading film"
        ffmpeg.run_with_progress(
            args,
            cwd=cache.dir,
            total_seconds=duration,
            on_progress=None if quiet else _progress_printer(label),
        )
        if not quiet:
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()

        proxy_video = None
        if settings.write_proxy_video:
            proxy_probe = ffmpeg.probe(cache.path(PROXY_FILE))
            proxy_video = (ffmpeg.streams(proxy_probe, "video") or [None])[0]

        return {
            "plan": plan.name,
            "plan_attempts": attempts,
            "proxy": PROXY_FILE if settings.write_proxy_video else None,
            "wav": WAV_FILE if audio else None,
            "scdet_raw": SCDET_FILE,
            "width": proxy_video.width if proxy_video else None,
            "height": proxy_video.height if proxy_video else None,
            "fps": round(proxy_video.fps, 3) if proxy_video and proxy_video.fps else None,
            # Always the source runtime, so stage 3 has a timeline to work from
            # whether or not a proxy video was written.
            "duration_s": round(duration, 3),
            "source_duration_s": round(duration, 3),
            # Transport streams often start at a non-zero timestamp. Later stages
            # subtract this so proxy time and source time stay aligned, without
            # which every selected clip in the final render would be offset.
            "timeline_offset_s": round(ffmpeg.start_time_seconds(probe_data), 3),
            "audio_stream": audio.label if audio else None,
            "audio_index": audio.index if audio else None,
            "source_video": video.label,
        }

    def summarize(meta: dict) -> str:
        bits = [str(meta.get("plan"))]
        if meta.get("proxy"):
            proxy_file = cache.path(PROXY_FILE)
            size_mb = proxy_file.stat().st_size / 1e6 if proxy_file.is_file() else 0
            detail = f"proxy {meta.get('width')}x{meta.get('height')}"
            if meta.get("fps"):
                detail += f"@{meta['fps']:g}"
            bits.append(f"{detail}, {size_mb:.0f}MB")
        else:
            bits.append("no proxy video")
        bits.append("wav" if meta.get("wav") else "no audio")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, artifacts, work,
        force=force, summarize=summarize,
        # A film with no cut clearing the detection floor legitimately produces
        # an empty file, which must not be mistaken for a truncated one.
        may_be_empty=[SCDET_FILE],
    )
