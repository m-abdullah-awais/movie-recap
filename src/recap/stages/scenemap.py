"""Stage 3, coarse shot map for the whole film.

Reads the scene change timestamps that stage 2 emitted during its single decode
and turns them into a usable shot list. PySceneDetect is deliberately not used
here: it is accurate but far too slow to run across a full feature. It arrives
later, in stage 6, restricted to the roughly 25 minutes of film the script
actually references.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ..cache import Cache, StageOutcome, run_stage, write_json
from ..config import Settings

STAGE = "scenemap"
VERSION = 1

SCENES_FILE = "scenes.json"
SCDET_FILE = "scdet.raw.txt"

PARAM_NAMES = ("min_shot_s", "max_shot_s", "tail_fraction", "fallback_shot_s")

# metadata=print emits a frame header followed by the requested key:
#   frame:0    pts:49152   pts_time:4
#   lavfi.scd.score=27.535
# The frame number is renumbered by the preceding select filter, so only the
# pts_time in the header is a usable timestamp.
_SCORE_RE = re.compile(r"lavfi\.scd\.score=([0-9.]+)")
_TIME_RE = re.compile(r"lavfi\.scd\.time=([0-9.]+)")
_PTS_RE = re.compile(r"\bpts_time:([0-9.]+)")


def parse_cuts(raw: Path) -> list[tuple[float, float]]:
    """Extract ``(time, score)`` pairs for every candidate cut.

    The detection file is written unbuffered, so a run interrupted part way still
    leaves valid records. A trailing header with no score line is discarded
    rather than treated as an error.

    Files produced before scores were recorded carry ``lavfi.scd.time`` instead.
    Those are still read, with an unknown score represented as infinity so that
    they survive any threshold filter.
    """
    if not raw.is_file():
        return []

    cuts: list[tuple[float, float]] = []
    pending: float | None = None

    for line in raw.read_text(encoding="utf-8", errors="replace").splitlines():
        pts_match = _PTS_RE.search(line)
        if pts_match:
            try:
                pending = float(pts_match.group(1))
            except ValueError:
                pending = None
            continue

        score_match = _SCORE_RE.search(line)
        if score_match and pending is not None:
            try:
                cuts.append((pending, float(score_match.group(1))))
            except ValueError:
                pass
            pending = None
            continue

        legacy = _TIME_RE.search(line)
        if legacy:
            try:
                cuts.append((float(legacy.group(1)), float("inf")))
            except ValueError:
                pass
            pending = None

    deduped: dict[float, float] = {}
    for time, score in cuts:
        if time < 0:
            continue
        key = round(time, 3)
        deduped[key] = max(deduped.get(key, 0.0), score)
    return sorted(deduped.items())


def build_shots(cuts: list[float], duration: float, settings: Settings) -> list[dict]:
    """Turn cut points into shots, cleaning up both extremes.

    Very short shots are absorbed into their neighbour, because flash frames and
    single frame transitions are detected as cuts but are useless as footage.
    Very long shots are subdivided so that clip selection has candidates inside a
    long static dialogue scene, instead of one unusable ninety second block.
    """
    boundaries = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    boundaries = sorted(set(boundaries))

    spans: list[list[float]] = [
        [boundaries[i], boundaries[i + 1]] for i in range(len(boundaries) - 1)
    ]
    spans = [s for s in spans if s[1] > s[0]]

    merged: list[list[float]] = []
    for span in spans:
        if span[1] - span[0] < settings.min_shot_s and merged:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    # A short opening shot has no previous neighbour, so it folds forward instead.
    if len(merged) > 1 and merged[0][1] - merged[0][0] < settings.min_shot_s:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    shots: list[dict] = []
    tail_starts_at = duration * (1.0 - settings.tail_fraction)
    for start, end in merged:
        length = end - start
        pieces = max(1, math.ceil(length / settings.max_shot_s))
        step = length / pieces
        for p in range(pieces):
            s = start + p * step
            e = start + (p + 1) * step if p < pieces - 1 else end
            shots.append(
                {
                    "i": len(shots),
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "duration": round(e - s, 3),
                    "split": pieces > 1,
                    # First pass at excluding end credits. Refined in stage 8.
                    "tail": s >= tail_starts_at,
                }
            )
    return shots


def uniform_shots(duration: float, settings: Settings) -> list[dict]:
    """Fallback shot list when detection produced nothing."""
    step = settings.fallback_shot_s
    shots: list[dict] = []
    tail_starts_at = duration * (1.0 - settings.tail_fraction)
    t = 0.0
    while t < duration:
        end = min(duration, t + step)
        shots.append(
            {
                "i": len(shots),
                "start": round(t, 3),
                "end": round(end, 3),
                "duration": round(end - t, 3),
                "split": False,
                "tail": t >= tail_starts_at,
            }
        )
        t = end
    return shots


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def run(
    cache: Cache,
    proxy_meta: dict,
    settings: Settings,
    *,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    duration = float(proxy_meta.get("duration_s") or proxy_meta.get("source_duration_s") or 0.0)
    if duration <= 0:
        raise RuntimeError("proxy metadata has no usable duration, re-run the proxy stage")

    params = settings.params(*PARAM_NAMES)
    params["scene_threshold"] = settings.scene_threshold
    # Chaining the proxy's key means any change to the detection pass itself,
    # such as the downscale width or the floor, invalidates this stage too.
    params["proxy_key"] = proxy_meta.get("key")

    def work() -> dict:
        candidates = parse_cuts(cache.path(SCDET_FILE))
        # Stage 2 detected at a permissive floor. The user's threshold is applied
        # here, so retuning it costs a few seconds instead of a full re-encode.
        cuts = [t for t, score in candidates if score >= settings.scene_threshold]
        degraded = None
        if not cuts:
            # Soft degrade rather than fail. A shot map on a fixed grid is far
            # worse than real cuts but keeps every later stage runnable.
            degraded = "uniform_fallback"
            shots = uniform_shots(duration, settings)
            if not quiet:
                print("  no scene changes found, falling back to a uniform shot grid")
        else:
            shots = build_shots(cuts, duration, settings)

        durations = [s["duration"] for s in shots]
        payload = {
            "schema": 1,
            "source_id": cache.sid,
            "source_name": cache.source.name,
            "duration_s": round(duration, 3),
            "timeline_offset_s": proxy_meta.get("timeline_offset_s", 0.0),
            "params": {
                "scene_threshold": settings.scene_threshold,
                "detect_width": settings.detect_width,
                "min_shot_s": settings.min_shot_s,
                "max_shot_s": settings.max_shot_s,
                "tail_fraction": settings.tail_fraction,
            },
            "shots": shots,
            "stats": {
                "candidate_cut_count": len(candidates),
                "raw_cut_count": len(cuts),
                "shot_count": len(shots),
                "cuts_per_min": round(len(cuts) / (duration / 60.0), 2) if duration else 0.0,
                "median_shot_s": round(_median(durations), 3),
                "mean_shot_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
                "shortest_s": round(min(durations), 3) if durations else 0.0,
                "longest_s": round(max(durations), 3) if durations else 0.0,
                "split_count": sum(1 for s in shots if s["split"]),
                "tail_count": sum(1 for s in shots if s["tail"]),
                "degraded": degraded,
            },
        }
        write_json(cache.path(SCENES_FILE), payload)
        return {"scenes": SCENES_FILE, **payload["stats"]}

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('shot_count', 0)} shots",
            f"{meta.get('cuts_per_min', 0)}/min",
            f"median {meta.get('median_shot_s', 0)}s",
            f"{meta.get('raw_cut_count', 0)}/{meta.get('candidate_cut_count', 0)} cuts kept",
        ]
        if meta.get("degraded"):
            bits.append(f"DEGRADED {meta['degraded']}")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, [SCENES_FILE], work, force=force, summarize=summarize
    )
