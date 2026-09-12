"""Stage 8, choosing footage for each line of narration.

Every narration line now has a measured duration, so the job is to fill exactly
that many seconds with shots that suit what is being said. Scores combine how
well a shot matches the line's visual query, how close it sits to the moment in
the film being described, whether its length falls in a comfortable band, and
penalties for reuse, darkness, and anything past the line's spoiler ceiling.

The output is an edit decision list. No video is touched here, which makes the
whole stage cheap enough to re-run while tuning the weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cache import Cache, StageOutcome, read_json, run_stage, write_json
from ..config import Settings

STAGE = "select"
VERSION = 1

EDL_FILE = "edl.json"

PARAM_NAMES = (
    "clip_weight", "proximity_weight", "band_weight",
    "reuse_penalty", "dark_penalty", "min_clip_s", "max_clip_s",
    "proximity_sigma_s", "dark_luma", "max_shot_uses",
)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def band_score(duration: float, low: float, high: float) -> float:
    """How comfortably a shot's length sits in the preferred band.

    Full marks inside the band, tapering outside it. A shot much shorter than the
    band is a fragment, and one much longer will be trimmed anyway, so neither is
    disqualified outright.
    """
    if duration <= 0:
        return 0.0
    if low <= duration <= high:
        return 1.0
    if duration < low:
        return max(0.0, duration / low)
    return max(0.0, high / duration)


def proximity_score(shot_start: float, story_time: float, sigma_s: float) -> float:
    """Closeness of a shot to the moment being described.

    A gaussian rather than a hard window, so a good match slightly further away
    can still win over a poor match nearby.
    """
    delta = shot_start - story_time
    return float(np.exp(-(delta * delta) / (2.0 * max(1.0, sigma_s) ** 2)))


def run(
    cache: Cache,
    settings: Settings,
    *,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    narration_path = cache.path("narration.json")
    shots_path = cache.path("shots.json")
    if not narration_path.is_file():
        raise RuntimeError("no narration.json found. Run the narrate stage first.")
    if not shots_path.is_file():
        raise RuntimeError("no shots.json found. Run the index stage first.")

    narration = read_json(narration_path)
    index = read_json(shots_path)
    runtime_s = _num(index.get("runtime_s"))
    lines = narration.get("segments") or []
    shots = [s for s in (index.get("shots") or []) if s.get("has_keyframe")]
    if not lines:
        raise RuntimeError("there is no narration to lay footage against")
    if not shots:
        raise RuntimeError("the shot index is empty")

    params = settings.params(*PARAM_NAMES)
    params["line_count"] = len(lines)
    params["shot_count"] = len(shots)

    def work() -> dict:
        image_path = cache.path("clip_index.npy")
        query_path = cache.path("query_index.npy")
        image_vectors = query_vectors = None
        if image_path.is_file() and query_path.is_file():
            try:
                image_vectors = np.load(image_path)
                query_vectors = np.load(query_path)
            except (ValueError, OSError):
                image_vectors = query_vectors = None

        use_clip = (
            image_vectors is not None
            and query_vectors is not None
            and image_vectors.size > 0
            and query_vectors.shape[0] >= len(lines)
        )
        if not quiet:
            print("  scoring with CLIP similarity and time proximity" if use_clip
                  else "  CLIP index unavailable, scoring on time proximity alone")

        starts = np.array([_num(s.get("start")) for s in shots], dtype=np.float32)
        ends = np.array([_num(s.get("end")) for s in shots], dtype=np.float32)
        durations = np.maximum(0.0, ends - starts)
        lumas = np.array(
            [_num(s.get("luma"), 128.0) for s in shots], dtype=np.float32
        )
        rows = [s.get("clip_row") for s in shots]
        bands = np.array(
            [band_score(float(d), settings.min_clip_s, settings.max_clip_s)
             for d in durations],
            dtype=np.float32,
        )
        dark = (lumas < settings.dark_luma).astype(np.float32)

        uses = np.zeros(len(shots), dtype=np.int32)
        timeline: list[dict] = []
        unfilled = 0

        gap_s = _num(narration.get("gap_s"))
        for position, line in enumerate(lines):
            # Footage has to span the line plus the silence after it. The
            # narration track carries those gaps, so covering only the spoken
            # seconds would leave the video one gap per line shorter than the
            # audio, and the tail of the narration would be cut off.
            need = _num(line.get("seconds"))
            if position < len(lines) - 1:
                need += gap_s
            story_time = _num(line.get("story_time"))
            ceiling = _num(line.get("spoiler_ceiling"), story_time)

            proximity = np.array(
                [proximity_score(float(s), story_time, settings.proximity_sigma_s)
                 for s in starts],
                dtype=np.float32,
            )

            similarity = np.zeros(len(shots), dtype=np.float32)
            if use_clip:
                wanted = line.get("script_i")
                wanted = int(wanted) if wanted is not None else int(line.get("i", 0))
                query = query_vectors[min(wanted, query_vectors.shape[0] - 1)]
                # Distinct names on purpose: reusing the outer loop's variable
                # here would clobber it for the rest of the iteration.
                for shot_position, shot_row in enumerate(rows):
                    if shot_row is not None and 0 <= shot_row < image_vectors.shape[0]:
                        similarity[shot_position] = float(image_vectors[shot_row] @ query)
                # Cosine similarity for CLIP sits in a narrow positive band, so
                # it is rescaled per line. Otherwise proximity would dominate
                # simply because it already spans zero to one.
                spread = similarity.max() - similarity.min()
                if spread > 1e-6:
                    similarity = (similarity - similarity.min()) / spread

            score = (
                settings.clip_weight * similarity
                + settings.proximity_weight * proximity
                + settings.band_weight * bands
                - settings.dark_penalty * dark
            )
            # Anything at or past the ceiling would show a reveal the narration
            # has not reached. This is a hard exclusion, not a penalty.
            score = np.where(starts <= ceiling, score, -1e6)
            # The reuse penalty only discourages. The cap is what stops one
            # striking shot appearing under half the recap.
            score = np.where(uses < settings.max_shot_uses, score, -1e6)

            clips: list[dict] = []
            remaining = need
            guard = 0
            while remaining > 0.3 and guard < 64:
                guard += 1
                adjusted = score - settings.reuse_penalty * uses
                best = int(np.argmax(adjusted))
                if adjusted[best] <= -1e5:
                    break

                available = float(durations[best])
                take = min(settings.max_clip_s, max(settings.min_clip_s, remaining))
                take = min(take, available) if available > 0 else take
                if take <= 0.05:
                    # Hard exclude. Bumping the use count only adds a fraction
                    # of a point of penalty, so the same shot would win again.
                    score[best] = -1e6
                    continue

                # Centre the clip in its shot so it avoids the cut at either end.
                offset = max(0.0, (available - take) / 2.0)
                clip_start = float(starts[best]) + offset
                clips.append({
                    "shot_i": shots[best].get("i", best),
                    "src_start": round(clip_start, 3),
                    "src_end": round(clip_start + take, 3),
                    "duration": round(take, 3),
                    "score": round(float(adjusted[best]), 4),
                    "similarity": round(float(similarity[best]), 4) if use_clip else None,
                })
                uses[best] += 1
                remaining -= take

            if not clips:
                unfilled += 1
                continue

            # The last clip absorbs any shortfall, so the footage under a line is
            # exactly as long as the line itself. A gap here would desynchronise
            # everything after it.
            drift = need - sum(c["duration"] for c in clips)
            if abs(drift) > 0.001:
                last = clips[-1]
                # Clamped to the runtime. Absorbing drift into the last clip can
                # otherwise ask for footage from past the end of the film.
                new_end = min(runtime_s, last["src_end"] + drift)
                last["src_end"] = round(new_end, 3)
                last["duration"] = round(new_end - last["src_start"], 3)

            timeline.append({
                "i": len(timeline),
                "script_i": line.get("script_i"),
                "start_s": line.get("start_s"),
                "seconds": round(need, 3),
                "story_time": story_time,
                "spoiler_ceiling": ceiling,
                "wav": line.get("wav"),
                "narration": line.get("narration"),
                "clips": clips,
            })

        if not timeline:
            raise RuntimeError("no footage could be selected for any narration line")

        clip_count = sum(len(t["clips"]) for t in timeline)
        reused = int((uses > 1).sum())
        payload = {
            "schema": 1,
            "source_id": cache.sid,
            "source_name": cache.source.name,
            "source_file": cache.source.name,
            "total_seconds": round(
                sum(c["duration"] for t in timeline for c in t["clips"]), 3
            ),
            "gap_s": narration.get("gap_s", 0.0),
            "used_clip": use_clip,
            "timeline": timeline,
            "stats": {
                "line_count": len(timeline),
                "unfilled_lines": unfilled,
                "clip_count": clip_count,
                "clips_per_line": round(clip_count / len(timeline), 2),
                "distinct_shots": int((uses > 0).sum()),
                "reused_shots": reused,
                "mean_clip_s": round(
                    sum(c["duration"] for t in timeline for c in t["clips"]) / clip_count, 2
                ) if clip_count else 0.0,
                "used_clip": use_clip,
                "degraded": None if use_clip else "no_clip_similarity",
            },
        }
        write_json(cache.path(EDL_FILE), payload)
        return {"edl": EDL_FILE, **payload["stats"]}

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('clip_count', 0)} clips",
            f"{meta.get('line_count', 0)} lines",
            f"mean {meta.get('mean_clip_s', 0)}s",
            f"{meta.get('distinct_shots', 0)} distinct shots",
        ]
        if meta.get("unfilled_lines"):
            bits.append(f"{meta['unfilled_lines']} unfilled")
        if meta.get("degraded"):
            bits.append(f"DEGRADED {meta['degraded']}")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, [EDL_FILE], work,
        force=force, summarize=summarize,
    )
