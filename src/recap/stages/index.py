"""Stage 6, the shot index.

One keyframe per shot, embedded with CLIP alongside the script's visual queries.
Saving the query embeddings here means stage 8 is pure numpy and needs no model.

Every shot between the opening titles and the end credits is a candidate. An
earlier version indexed only narrow windows around each narration anchor, which
was a speed optimisation that quietly capped retrieval: it left about three
candidates per line, so the timestamp chose the footage and CLIP merely broke
ties. It was also brittle, because those anchors come from Claude reading
subtitle timings, and one off by half a minute put every candidate in the wrong
scene. Time is now a preference applied in stage 8 rather than a gate applied
here. Set index_whole_film to false to restore the old behaviour.

Boundaries come from stage 3 by default. PySceneDetect can refine them, but
measured on a 94 minute film that cost 6 minutes 54 seconds to find 24 extra
shots out of 243, so it is opt in rather than automatic.

Vision is used only for retrieval. It never reads the story.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import clip as clipmod
from .. import ffmpeg, models
from ..cache import Cache, StageOutcome, read_json, run_stage, write_json
from ..config import Settings

STAGE = "index"
VERSION = 6  # bumped: whole-film search, and keyframes padded not cropped

SHOTS_FILE = "shots.json"
INDEX_FILE = "clip_index.npy"
QUERY_FILE = "query_index.npy"
KEYFRAME_DIR = "keyframes"

PARAM_NAMES = (
    "index_whole_film", "keyframe_fit",
    "index_window_s", "index_target_coverage", "index_min_window_s",
    "index_max_shots", "refine_shots", "refine_threshold", "clip_batch",
    "credits_lead_in_s", "credits_lead_out_s",
)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def narrow_regions(
    anchors: list[float], window_s: float, runtime_s: float
) -> list[tuple[float, float]]:
    """Union of windows around each narration anchor.

    Overlapping windows are merged so a region is decoded once rather than once
    per anchor that happens to fall inside it.
    """
    spans = sorted(
        (max(0.0, a - window_s), min(runtime_s, a + window_s))
        for a in anchors
        if 0.0 <= a <= runtime_s
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(s, 3), round(e, 3)) for s, e in merged if e > s]


def narrow_adaptively(
    anchors: list[float],
    runtime_s: float,
    max_window_s: float,
    target_coverage: float,
    min_window_s: float,
) -> tuple[list[tuple[float, float]], float]:
    """Shrink the window until the regions cover at most the target fraction.

    A fixed window does not narrow anything when the narration is dense. Measured
    on a 94 minute film with 77 segments a median 62 seconds apart, a window of
    plus or minus 90 seconds merged into four regions covering 99 percent of the
    film, so the whole point of narrowing was lost and shot detection scanned
    everything.

    Coverage falls monotonically as the window shrinks, so a halving search finds
    a window that meets the target. The floor stops it collapsing to nothing on a
    film narrated end to end.
    """
    window = max(min_window_s, max_window_s)
    regions = narrow_regions(anchors, window, runtime_s)
    budget = max(1.0, runtime_s) * target_coverage

    while sum(e - s for s, e in regions) > budget and window > min_window_s:
        window = max(min_window_s, window / 2.0)
        regions = narrow_regions(anchors, window, runtime_s)

    return regions, window


def content_span(
    transcript: dict, runtime_s: float, settings: Settings
) -> tuple[float, float]:
    """The stretch of film that is actually the film.

    Studio logos and opening titles sit before the first spoken line, and end
    credits sit after the last one, so the dialogue timing locates both without
    any extra analysis. Measured on a 94 minute film, that excluded 34 seconds of
    opening titles and 8.5 minutes of end credits, none of which is usable
    footage for a recap.

    A lead in and lead out are kept, because a film usually opens on an
    establishing shot before anyone speaks and closes on a beat after the last
    line. Falling back to the whole runtime is safe: the credits tail flag from
    stage 3 still applies on top of this.
    """
    stats = transcript.get("stats") or {}
    first = _num(stats.get("first_cue_s"), -1.0)
    last = _num(stats.get("last_cue_s"), -1.0)
    if first < 0 or last <= first:
        return 0.0, runtime_s

    start = max(0.0, first - max(0.0, settings.credits_lead_in_s))
    end = min(runtime_s, last + max(0.0, settings.credits_lead_out_s))
    if end <= start:
        return 0.0, runtime_s
    return round(start, 3), round(end, 3)


def clamp_regions(
    regions: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """Trim narrowed regions to the film's content, dropping any that fall outside."""
    out: list[tuple[float, float]] = []
    for a, b in regions:
        a2, b2 = max(a, start), min(b, end)
        if b2 > a2:
            out.append((round(a2, 3), round(b2, 3)))
    return out


def shots_in_regions(shots: list[dict], regions: list[tuple[float, float]]) -> list[dict]:
    """Coarse shots that overlap any narrowed region, excluding the credits tail."""
    kept: list[dict] = []
    for shot in shots:
        if shot.get("tail"):
            continue
        start, end = _num(shot.get("start")), _num(shot.get("end"))
        if any(start < region_end and end > region_start
               for region_start, region_end in regions):
            kept.append(shot)
    return kept


def refine_with_scenedetect(
    source: Path,
    regions: list[tuple[float, float]],
    settings: Settings,
    quiet: bool,
) -> list[dict] | None:
    """Re-detect shot boundaries inside the narrowed regions.

    Returns None when PySceneDetect is unavailable, so the caller keeps the
    coarse boundaries rather than failing. This decodes the regions again, which
    is the expensive part, and it is why the whole film is never refined.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError:
        if not quiet:
            print("  PySceneDetect is not installed, keeping the coarse boundaries")
        return None

    refined: list[dict] = []
    try:
        for number, (start, end) in enumerate(regions, 1):
            if not quiet:
                print(f"    refining region {number}/{len(regions)}  "
                      f"{start:.0f}s to {end:.0f}s")
            video = open_video(str(source))
            video.seek(start)
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=settings.refine_threshold))
            manager.detect_scenes(video, end_time=end)
            for scene_start, scene_end in manager.get_scene_list():
                a, b = scene_start.get_seconds(), scene_end.get_seconds()
                if b > a:
                    refined.append({"start": round(a, 3), "end": round(b, 3),
                                    "duration": round(b - a, 3), "tail": False,
                                    "source": "scenedetect"})
    except Exception as exc:  # noqa: BLE001 - any backend problem falls back
        if not quiet:
            print(f"  refinement failed, keeping the coarse boundaries: {exc}")
        return None

    return refined or None


def fit_filter(mode: str) -> str:
    """How a source frame is fitted into CLIP's square input.

    Cropping matches CLIP's own preprocessing, but on a 1920x1080 frame it fits
    the short side to 224 and then keeps only the middle 224 of 398 pixels,
    discarding 44 percent of the width. In widescreen film the subject is often
    outside that middle. Padding fits the whole frame and fills the remainder
    with black, trading some detail for keeping everything the shot contains.
    """
    size = clipmod.IMAGE_SIZE
    if mode == "crop":
        return (f"scale={size}:{size}:force_original_aspect_ratio=increase,"
                f"crop={size}:{size}")
    return (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black")


def _extract_keyframe(source: Path, at_s: float, target: Path, fit: str) -> bool:
    """Pull one frame, fitted to CLIP's input size.

    ``-ss`` before the input seeks on keyframes first, which is what makes a few
    hundred independent extractions affordable.
    """
    if target.is_file() and target.stat().st_size > 0:
        return True
    try:
        ffmpeg.run(
            [
                ffmpeg.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                "-ss", f"{max(0.0, at_s):.3f}",
                "-i", str(source.resolve()),
                "-frames:v", "1",
                "-vf", fit_filter(fit),
                "-q:v", "3",
                str(target),
            ],
            timeout=120,
        )
    except (ffmpeg.FfmpegError, subprocess.TimeoutExpired, ffmpeg.MissingBinary):
        return False
    return target.is_file() and target.stat().st_size > 0


def run(
    cache: Cache,
    source: Path,
    settings: Settings,
    *,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    script_path = cache.path("script.json")
    scenes_path = cache.path("scenes.json")
    if not script_path.is_file():
        raise RuntimeError("no script.json found. Run the script stage first.")
    if not scenes_path.is_file():
        raise RuntimeError("no scenes.json found. Run the scenemap stage first.")

    script = read_json(script_path)
    scenes = read_json(scenes_path)
    transcript_path = cache.path("transcript.json")
    transcript = read_json(transcript_path) if transcript_path.is_file() else {}
    segments = script.get("segments") or []
    coarse = scenes.get("shots") or []
    runtime_s = _num(scenes.get("duration_s")) or _num(script.get("runtime_s"))
    if not segments:
        raise RuntimeError("the script has no segments, so there is nothing to index")

    params = settings.params(*PARAM_NAMES)
    # Whether CLIP is present changes the artifacts, so it belongs in the key.
    # If the model appears later, this stage re-runs and adds the embeddings.
    have_clip = models.clip_available()
    params["clip"] = have_clip
    params["segment_count"] = len(segments)

    artifacts = [SHOTS_FILE] + ([INDEX_FILE, QUERY_FILE] if have_clip else [])

    video_streams = ffmpeg.streams(ffmpeg.probe(source), "video")
    source_fps = round(video_streams[0].fps, 6) if video_streams and video_streams[0].fps else None

    def work() -> dict:
        content_start, content_end = content_span(transcript, runtime_s, settings)
        if not quiet and (content_start > 1.0 or content_end < runtime_s - 1.0):
            print(f"  film content runs {content_start:.0f}s to {content_end:.0f}s, "
                  f"discarding {content_start / 60:.1f} min of opening titles and "
                  f"{(runtime_s - content_end) / 60:.1f} min of end credits")

        if settings.index_whole_film:
            # Everything between the titles and the credits is a candidate, so a
            # good visual match can win from anywhere in the film. Time still
            # matters, but as a score in stage 8 rather than a gate here.
            regions = [(content_start, content_end)]
            window = None
        else:
            anchors = [_num(s.get("story_time")) for s in segments]
            regions, window = narrow_adaptively(
                anchors, runtime_s,
                settings.index_window_s,
                settings.index_target_coverage,
                settings.index_min_window_s,
            )
            regions = clamp_regions(regions, content_start, content_end)
            if not regions:
                raise RuntimeError(
                    "every narrowed region fell outside the film's content span"
                )

        covered = sum(e - s for s, e in regions)
        if not quiet:
            if window is None:
                print(f"  indexing the whole film, {covered / 60:.1f} min "
                      f"of {runtime_s / 60:.1f} min after titles and credits")
            else:
                print(f"  narrowed to {len(regions)} regions using a "
                      f"{window:.0f}s window, {covered / 60:.1f} min of "
                      f"{runtime_s / 60:.1f} min "
                      f"({covered / max(1.0, runtime_s) * 100:.0f}% of the film)")

        shots = None
        if settings.refine_shots:
            shots = refine_with_scenedetect(source, regions, settings, quiet)
        refined = shots is not None
        if shots is None:
            shots = shots_in_regions(coarse, regions)

        # Longest first, so the cap keeps the shots most likely to cover a
        # segment's duration rather than a run of fragments.
        # Boundaries can come from either stage 3 or PySceneDetect, so the
        # content span is enforced here rather than trusted from upstream.
        before = len(shots)
        shots = [
            sh for sh in shots
            if _num(sh.get("start")) >= content_start - 0.001
            and _num(sh.get("end")) <= content_end + 0.001
        ]
        if not quiet and before != len(shots):
            print(f"  dropped {before - len(shots)} shots in the titles or credits")

        if len(shots) > settings.index_max_shots:
            shots = sorted(shots, key=lambda s: -_num(s.get("duration")))
            shots = sorted(shots[: settings.index_max_shots],
                           key=lambda s: _num(s.get("start")))
        if not shots:
            raise RuntimeError("no shots fall inside the narrowed regions")

        keyframe_dir = cache.dir / KEYFRAME_DIR
        keyframe_dir.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        for position, shot in enumerate(shots):
            start, end = _num(shot.get("start")), _num(shot.get("end"))
            middle = start + (end - start) / 2.0
            entries.append({
                "i": position,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "keyframe_s": round(middle, 3),
                # Named by timestamp, not by position. A positional name is
                # reused by a later run whose shot boundaries differ, which
                # silently pairs a shot with a frame from somewhere else
                # entirely. The timestamp identifies the frame's actual content.
                "keyframe": f"kf_{int(round(middle * 1000)):09d}.jpg",
                "source": shot.get("source", "scenemap"),
            })

        if not quiet:
            print(f"  extracting {len(entries)} keyframes with "
                  f"{settings.keyframe_workers} workers")

        def grab(entry: dict) -> bool:
            return _extract_keyframe(
                source, entry["keyframe_s"], keyframe_dir / entry["keyframe"],
                settings.keyframe_fit,
            )

        with ThreadPoolExecutor(max_workers=max(1, settings.keyframe_workers)) as pool:
            got = list(pool.map(grab, entries))
        for entry, ok in zip(entries, got):
            entry["has_keyframe"] = bool(ok)

        usable = [e for e in entries if e["has_keyframe"]]
        if not usable:
            raise RuntimeError("no keyframes could be extracted from the narrowed regions")

        # Mean brightness is recorded for every shot whether or not CLIP loads.
        # Stage 8 uses it to avoid backing narration with a black frame from a
        # fade, which a similarity score alone would not catch.
        for entry in entries:
            entry["luma"] = None
            if not entry["has_keyframe"]:
                continue
            frame = clipmod.read_frame(keyframe_dir / entry["keyframe"])
            if frame is None:
                entry["has_keyframe"] = False
                continue
            entry["luma"] = round(float(frame.mean()), 2)

        usable = [e for e in entries if e["has_keyframe"]]
        if not usable:
            raise RuntimeError("no keyframes could be read back for indexing")

        degraded = None
        embedded = 0
        # One quick attempt only. Patiently retrying a rate limited host
        # belongs in the fetch-models command, not in the middle of a stage.
        assets = models.clip_paths() if have_clip else models.ensure_clip(
            quiet=quiet, attempts=1
        )
        clip = clipmod.load(assets)
        if clip is None:
            # Documented fallback. Stage 8 scores on time proximity alone, which
            # still produces a finished video, just with less apt footage.
            degraded = "no_clip"
            if not quiet:
                print("  CLIP is unavailable, selection will fall back to time proximity")
            for entry in entries:
                entry["clip_row"] = None
        else:
            if not quiet:
                print(f"  embedding {len(usable)} keyframes and "
                      f"{len(segments)} visual queries")
            vectors: list[np.ndarray] = []
            batch: list[np.ndarray] = []
            pending: list[dict] = []

            def flush() -> None:
                nonlocal embedded
                if not batch:
                    return
                stacked = np.stack(batch).astype(np.float32)
                result = clip.embed_images(stacked)
                for row, entry in zip(result, pending):
                    entry["clip_row"] = len(vectors)
                    vectors.append(row)
                    embedded += 1
                batch.clear()
                pending.clear()

            for entry in entries:
                entry["clip_row"] = None
                if not entry["has_keyframe"]:
                    continue
                frame = clipmod.read_frame(keyframe_dir / entry["keyframe"])
                if frame is None:
                    entry["has_keyframe"] = False
                    continue
                batch.append(clipmod.preprocess(frame))
                pending.append(entry)
                if len(batch) >= settings.clip_batch:
                    flush()
            flush()

            image_matrix = (np.stack(vectors) if vectors
                            else np.zeros((0, clipmod.EMBED_DIM), dtype=np.float32))
            query_matrix = clip.embed_texts([s.get("visual_query", "") for s in segments])
            np.save(cache.path(INDEX_FILE), image_matrix.astype(np.float32))
            np.save(cache.path(QUERY_FILE), query_matrix.astype(np.float32))

        payload = {
            "schema": 1,
            "source_id": cache.sid,
            "source_name": cache.source.name,
            "runtime_s": round(runtime_s, 2),
            # Recorded for stage 8, which must express clip lengths in whole
            # frames. The concat demuxer quantises to frames and rounds down, so
            # asking for an arbitrary duration loses up to one frame per clip.
            "source_fps": source_fps,
            "regions": [{"start": s, "end": e} for s, e in regions],
            "content_start_s": content_start,
            "content_end_s": content_end,
            "shots": entries,
            "stats": {
                "whole_film": settings.index_whole_film,
                "keyframe_fit": settings.keyframe_fit,
                "region_count": len(regions),
                "content_start_s": content_start,
                "content_end_s": content_end,
                "titles_discarded_s": round(content_start, 1),
                "credits_discarded_s": round(runtime_s - content_end, 1),
                "covered_seconds": round(covered, 1),
                "covered_fraction": round(covered / max(1.0, runtime_s), 4),
                "shot_count": len(entries),
                "keyframes_ok": len(usable),
                "refined": refined,
                "embedded": embedded,
                "embedding_dim": clipmod.EMBED_DIM if embedded else 0,
                "degraded": degraded,
            },
        }
        write_json(cache.path(SHOTS_FILE), payload)
        return {"shots": SHOTS_FILE, **payload["stats"]}

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('shot_count', 0)} shots",
            f"{meta.get('covered_seconds', 0) / 60:.0f} min indexed",
            f"{meta.get('keyframes_ok', 0)} keyframes",
        ]
        bits.append("whole film" if meta.get("whole_film") else "narrowed")
        bits.append("refined" if meta.get("refined") else "coarse boundaries")
        if meta.get("embedded"):
            bits.append(f"{meta['embedded']} embedded")
        if meta.get("degraded"):
            bits.append(f"DEGRADED {meta['degraded']}")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, artifacts, work,
        force=force, summarize=summarize,
    )
