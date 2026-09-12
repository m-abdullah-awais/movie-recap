"""Stage 2, proxy generation, with scene detection fused into the same pass.

One decode of the source produces three things: the 480p analysis proxy, the raw
scene change timestamps, and the 16 kHz mono wav. Sharing the decode matters
because decoding a 2 hour film is the single most expensive operation in the
pipeline on this hardware, and doing it twice would cost several minutes for
nothing.

The original file is never touched again until the render stage, which
stream-copies from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .. import ffmpeg, probe
from ..cache import Cache, StageOutcome, run_stage
from ..config import Settings

STAGE = "proxy"
VERSION = 1

PROXY_FILE = "proxy.mp4"
WAV_FILE = "proxy.wav"
SCDET_FILE = "scdet.raw.txt"

PARAM_NAMES = (
    "proxy_height", "proxy_fps", "proxy_fps_threshold", "allow_qsv",
    "x264_preset", "x264_crf", "qsv_quality", "audio_rate",
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
    chain = [f"scale=-2:{target_height}:flags=fast_bilinear"]
    # Leave 23.976 and 24 fps films untouched. Only reduce genuinely high frame
    # rate sources, where the extra frames are wasted encode work.
    if source_fps and source_fps > settings.proxy_fps_threshold:
        chain.append(f"fps={settings.proxy_fps}")
    prepared = ",".join(chain)

    return (
        f"[0:v]{prepared},split=2[penc][pdet];"
        f"[pdet]scale={settings.detect_width}:-2:flags=neighbor,"
        f"scdet=t={settings.scene_floor},"
        f"metadata=mode=select:key=lavfi.scd.time,"
        f"metadata=mode=print:key=lavfi.scd.score:file={SCDET_FILE}:direct=1,"
        f"nullsink"
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

    # Output 1, the analysis proxy. The detection branch needs no output of its
    # own because it ends in nullsink inside the filter graph.
    args += ["-map", "[penc]", *plan.video_args, "-an", "-sn", "-dn", proxy_name]
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

    artifacts = [PROXY_FILE, SCDET_FILE] + ([WAV_FILE] if audio else [])

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

        plans = probe.encoder_ladder(settings)
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
        ffmpeg.run_with_progress(
            args,
            cwd=cache.dir,
            total_seconds=duration,
            on_progress=None if quiet else _progress_printer("encoding proxy"),
        )
        if not quiet:
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()

        proxy_probe = ffmpeg.probe(cache.path(PROXY_FILE))
        proxy_video = (ffmpeg.streams(proxy_probe, "video") or [None])[0]

        return {
            "plan": plan.name,
            "plan_attempts": attempts,
            "proxy": PROXY_FILE,
            "wav": WAV_FILE if audio else None,
            "scdet_raw": SCDET_FILE,
            "width": proxy_video.width if proxy_video else None,
            "height": proxy_video.height if proxy_video else None,
            "fps": round(proxy_video.fps, 3) if proxy_video and proxy_video.fps else None,
            "duration_s": round(ffmpeg.duration_seconds(proxy_probe), 3),
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
        size_mb = cache.path(PROXY_FILE).stat().st_size / 1e6 if cache.path(PROXY_FILE).is_file() else 0
        bits = [f"{meta.get('width')}x{meta.get('height')}"]
        if meta.get("fps"):
            bits.append(f"{meta['fps']:g}fps")
        bits.append(str(meta.get("plan")))
        bits.append(f"{size_mb:.0f}MB")
        if not meta.get("wav"):
            bits.append("no audio")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, artifacts, work,
        force=force, summarize=summarize,
        # A film with no cut clearing the detection floor legitimately produces
        # an empty file, which must not be mistaken for a truncated one.
        may_be_empty=[SCDET_FILE],
    )
