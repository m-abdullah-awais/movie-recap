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
VERSION = 11  # bumped: footage window comes from the cited beat, and never repeats

EDL_FILE = "edl.json"

PARAM_NAMES = (
    "clip_weight", "proximity_weight", "band_weight",
    "reuse_penalty", "dark_penalty", "min_clip_s", "max_clip_s",
    "proximity_sigma_s", "dark_luma", "max_shot_uses", "max_shot_distance_s",
    "footage_mode", "clip_target_s", "max_span_s",
    "continuous_runs", "max_run_s", "similarity_floor",
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


def proximity_score(shot_middle: float, story_time: float, sigma_s: float) -> float:
    """Closeness of a shot to the moment being described.

    Measured to the middle of the shot rather than its first frame. A shot
    beginning exactly on the anchor scored a perfect 1.0 while its content was
    the following few seconds, which biased selection forward: footage ran a mean
    11.5 seconds ahead of the moment being narrated.

    A gaussian rather than a hard window, so a good match slightly further away
    can still win over a poor match nearby. The hard bound is applied separately.
    """
    delta = shot_middle - story_time
    return float(np.exp(-(delta * delta) / (2.0 * max(1.0, sigma_s) ** 2)))


def quantise(duration: float, fps: float | None) -> tuple[float, float]:
    """Round a clip length to whole frames, and return the out point to request.

    The concat demuxer keeps frames whose timestamp falls inside the in and out
    points, so an arbitrary out point yields however many whole frames happen to
    fit, rounding down. Measured on a 24 fps film that lost about half a frame
    per clip, which over 366 clips accumulated to 7.3 seconds and left the
    footage running steadily ahead of the narration it was chosen for.

    Asking for the frame count plus half a frame makes the arithmetic land on the
    intended count exactly. The returned pair is the duration to account for and
    the slightly longer span to request.
    """
    if not fps or fps <= 0:
        return duration, duration
    frames = max(1, int(round(duration * fps)))
    exact = frames / fps
    return exact, (frames + 0.5) / fps


def _beat_lookup(cache: Cache) -> dict:
    """Story beats by id, and the beat each script segment cites.

    Narration files written before this stage needed beat ids do not carry one,
    so the script is consulted to fill them in by position. That keeps an
    existing cache working instead of forcing the whole story to be bought
    again for the sake of one field.
    """
    beats: dict = {}
    story_path = cache.path("story.json")
    if story_path.is_file():
        try:
            story = read_json(story_path)
        except Exception:  # noqa: BLE001 - a missing story just means no windows
            story = {}
        for position, beat in enumerate(story.get("beats") or []):
            identifier = beat.get("id", position)
            try:
                beats[int(identifier)] = beat
            except (TypeError, ValueError):
                continue
    return beats


def _segment_beats(cache: Cache) -> dict:
    """Which beat each script segment cites, keyed by the segment's index."""
    script_path = cache.path("script.json")
    if not script_path.is_file():
        return {}
    try:
        script = read_json(script_path)
    except Exception:  # noqa: BLE001
        return {}
    mapping: dict = {}
    for position, segment in enumerate(script.get("segments") or []):
        identifier = segment.get("i", position)
        if segment.get("beat_id") is not None:
            mapping[identifier] = segment.get("beat_id")
    return mapping


def _group_key(line: dict):
    """What makes two consecutive lines part of the same scene.

    The beat they cite, when they cite one. Older narration files predate that
    field, so the anchor is the fallback.
    """
    beat_id = line.get("beat_id")
    if beat_id is not None:
        return ("beat", beat_id)
    return ("time", round(_num(line.get("story_time")), 2))


def _beat_range(
    line: dict, beats: dict, default_s: float, max_span_s: float
) -> tuple[float, float]:
    """Where in the film the beat this line narrates actually happens."""
    anchor = _num(line.get("story_time"))
    beat = beats.get(line.get("beat_id"))
    low = _num(beat.get("start_s"), anchor) if beat else anchor
    high = _num(beat.get("end_s"), low) if beat else low
    if high <= low:
        high = low + default_s
    return low, min(high, low + max_span_s)


def _group_split(
    group: list[int], lines: list[dict], low: float, high: float
) -> list[tuple]:
    """Divide one beat between the lines that narrate it.

    Splitting it into consecutive shares by spoken length treats the lines as
    interchangeable, which they are not: one describes the cockpit, the next
    describes the cabin. Each line instead sits over the moment it is about.

    That moment is the line of dialogue the script stage cited, looked up in
    the subtitles, so it is exact. It is the strongest signal available here by
    a wide margin: a beat runs a median 103 seconds, while subtitle cues land
    every 2.6, and CLIP disagrees with itself about where a scene is by a
    median of 348 seconds on the test film. So the words place the window and
    CLIP only ranks the shots inside it.

    Windows are kept in order and never overlap, because the narration runs
    forward and so must the picture.
    """
    spoken = [max(0.1, _num(lines[i].get("seconds"))) for i in group]
    total = sum(spoken)

    def proportional() -> list[tuple]:
        out = []
        cursor = low
        for share in spoken:
            width = (high - low) * share / total
            out.append((cursor, cursor + width))
            cursor += width
        return out

    # A window only has to be wide enough to offer a choice of shots for the
    # footage that line needs, not to own an equal share of the scene. Dividing
    # the beat exactly leaves no slack, and a window with no slack cannot move
    # onto the moment it describes.
    widths = [min(max(seconds * 1.6, 8.0), high - low) for seconds in spoken]
    if sum(widths) > high - low:
        return proportional()

    bounds = []
    cursor = low
    for order, index in enumerate(group):
        width = widths[order]
        # Centred just after the cited line is spoken, because a shot of a line
        # being said starts when it is said.
        centre = _num(lines[index].get("story_time"), low) + width / 4.0
        tail = sum(widths[order + 1:])
        start = max(cursor, centre - width / 2.0)
        start = min(start, high - tail - width)
        start = max(start, low)
        bounds.append((start, start + width))
        cursor = start + width
    return bounds


def line_windows(
    lines: list[dict],
    beats: dict,
    *,
    default_s: float,
    max_span_s: float,
    content_start: float,
    content_end: float,
) -> list[tuple]:
    """The stretch of film each line may show, and the ceiling that applies.

    Every line is written about one story beat and cites it by id, so the beat's
    own start and end are the window. Tiling by anchors instead, running each
    line from its anchor to the next one, read well on paper and was wrong in
    practice: where several lines share an anchor the later ones were pushed
    past it, so a line describing a cockpit at 1018 seconds was given 1092 to
    1125 to choose from and could not show that cockpit at all. Measured on the
    test film, the best visual match lay inside the allowed window for only 14
    lines out of 83.

    Lines sharing a beat divide it between themselves in proportion to how long
    they are spoken, so a group walks through the scene in order without showing
    each other's footage.

    The spoiler ceiling is raised to the end of the beat being narrated. Footage
    of the very beat the words describe cannot spoil it, and a ceiling below the
    window is what made eight lines collapse onto one repeated clip just under
    the limit, in one case showing the same three seconds eight times.
    """
    windows: list[tuple] = []
    index = 0
    while index < len(lines):
        key = _group_key(lines[index])
        group = [index]
        while (group[-1] + 1 < len(lines)
               and _group_key(lines[group[-1] + 1]) == key):
            group.append(group[-1] + 1)

        low, high = _beat_range(lines[index], beats, default_s, max_span_s)
        low = max(low, content_start)
        high = min(max(high, low + 1.0), content_end)

        split = _group_split(group, lines, low, high)
        for i, (start, end) in zip(group, split):
            ceiling = max(_num(lines[i].get("spoiler_ceiling"), high), high)
            windows.append((start, end, ceiling))

        index = group[-1] + 1
    return windows


def build_traversal(
    *,
    need: float,
    span: tuple,
    score: "np.ndarray",
    uses: "np.ndarray",
    starts: "np.ndarray",
    middles: "np.ndarray",
    durations: "np.ndarray",
    ceiling: float,
    content_start: float,
    content_end: float,
    similarity: "np.ndarray | None",
    shots: list[dict],
    settings: Settings,
    source_fps: float | None,
) -> list[dict]:
    """A few ordered clips spread across the film this line narrates.

    The picture has to keep up with the words. Narration compresses roughly 13
    seconds of plot into every second of speech, so a line that plays one
    continuous stretch shows only the opening of what it is describing and
    everything mentioned afterwards arrives late or never.

    Sampling evenly across the line's span fixes the pacing: at the halfway point
    of the sentence the picture is at the halfway point of the events. Clips come
    out in film order by construction, so the sequence still reads forward.
    """
    span_start, span_end = span
    span_start = max(span_start, content_start)
    span_end = max(span_start + 0.1, min(span_end, content_end, ceiling))

    count = max(1, min(5, int(round(need / max(1.0, settings.clip_target_s)))))
    piece = need / count

    clips: list[dict] = []
    taken: list[int] = []
    floor_time = span_start

    for step in range(count):
        # Where in the described stretch this clip should sit.
        target = span_start + (step + 0.5) / count * (span_end - span_start)
        reach = max(piece, (span_end - span_start) / count)

        local = score - settings.reuse_penalty * uses
        # Prefer shots near this point in the traversal, and never go backwards.
        offset = np.abs(middles - target)
        local = local + settings.proximity_weight * np.exp(
            -(offset * offset) / (2.0 * max(1.0, reach) ** 2)
        )
        local = np.where(starts + durations > floor_time, local, -1e6)
        local = np.where(offset <= max(reach * 3.0, 20.0), local, -1e6)

        best = int(np.argmax(local))
        if local[best] <= -1e5:
            # Nothing usable ahead. Carry straight on from where we are, and
            # report the shot that stretch actually falls in rather than
            # whichever index happened to win an argmax over excluded scores.
            begin = max(floor_time, span_start)
            chosen = int(np.searchsorted(starts, begin, side="right")) - 1
            chosen = max(0, min(chosen, len(shots) - 1))
        else:
            begin = max(float(starts[best]), floor_time)
            chosen = best
            taken.append(best)

        want = piece
        limit = min(ceiling, content_end)
        if begin + want > limit:
            begin = max(content_start, limit - want)

        # Never show the same seconds twice inside one line. Pinning against
        # the ceiling used to do exactly that: every clip in the line came back
        # as the same few seconds just under the limit. If the next clip cannot
        # be placed after the previous one, stop, and let the shortfall be
        # absorbed by lengthening the last clip rather than by repeating it.
        if clips:
            previous_end = clips[-1]["src_end"]
            if begin < previous_end - 0.04:
                begin = previous_end
            if begin + want > limit:
                break

        exact, request = quantise(want, source_fps)
        clips.append({
            "shot_i": shots[chosen].get("i", chosen),
            "src_start": round(begin, 4),
            "src_end": round(begin + request, 4),
            "duration": round(exact, 4),
            "score": round(float(local[best]), 4),
            "similarity": (round(float(similarity[chosen]), 4)
                           if similarity is not None else None),
            "target_s": round(target, 2),
        })
        floor_time = begin + exact

    for index in taken:
        uses[index] += 1
    return clips


def build_runs(
    *,
    need: float,
    score: "np.ndarray",
    uses: "np.ndarray",
    starts: "np.ndarray",
    durations: "np.ndarray",
    ceiling: float,
    content_start: float,
    content_end: float,
    similarity: "np.ndarray | None",
    shots: list[dict],
    settings: Settings,
    source_fps: float | None,
    resume_at: float | None = None,
) -> list[dict]:
    """One unbroken stretch of film for a narration line.

    Playing continuously through the film's own cuts is how a recap is normally
    cut by hand, and it removes two faults of stitching short clips together.
    There is no ordering to get wrong, because a run is already in film order.
    And nothing is filler, because the run is not padding out a remainder with
    progressively worse matches.

    A run starts on the chosen shot's first frame, so it begins on a cut rather
    than mid-shot. It is then slid backwards if it would cross the line's spoiler
    ceiling or the end of the film's content, and only ever shortened if there is
    genuinely nowhere to put it.

    ``resume_at`` continues from where the previous line's footage ended. Two
    thirds of lines narrate a beat that the line before also narrated, and
    scoring them independently sent the second one elsewhere, because the reuse
    penalty pushed it off the footage the first had just used. Continuing instead
    plays the scene straight through across both lines, which is what the
    narration is describing.

    Longer lines than ``max_run_s`` get a second run from a different part of the
    film rather than one very long stretch, which would drift out of the scene.
    """
    clips: list[dict] = []
    remaining = need
    working = score.copy()
    first = True

    for _ in range(6):
        if remaining <= 0.05:
            break
        adjusted = working - settings.reuse_penalty * uses
        best = int(np.argmax(adjusted))
        if adjusted[best] <= -1e5:
            break

        want = min(remaining, settings.max_run_s)

        if first and resume_at is not None and content_start <= resume_at < min(ceiling, content_end):
            # Carry straight on from the previous line rather than cutting away.
            begin = resume_at
            best = int(np.argmin(np.abs(starts - resume_at)))
        else:
            begin = float(starts[best])
        first = False

        # Slide back rather than forward, so the run never shows something the
        # narration has not reached.
        limit = min(ceiling, content_end)
        if begin + want > limit:
            begin = limit - want
        begin = max(begin, content_start)
        if begin + want > content_end:
            want = max(0.0, content_end - begin)
        if want <= 0.05:
            working[best] = -1e6
            continue

        exact, request = quantise(want, source_fps)
        clips.append({
            "shot_i": shots[best].get("i", best),
            "src_start": round(begin, 4),
            "src_end": round(begin + request, 4),
            "duration": round(exact, 4),
            "score": round(float(adjusted[best]), 4),
            "similarity": (round(float(similarity[best]), 4)
                           if similarity is not None else None),
            "continuous": True,
        })
        remaining -= exact
        # Do not pick the same neighbourhood again for the remainder.
        working[best] = -1e6

    return clips


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
    # Each line is written about one story beat, and the beat knows where in the
    # film it happens. That range is what the line is allowed to show, so the
    # story is read here rather than inferred from the gaps between anchors.
    beats = _beat_lookup(cache)
    runtime_s = _num(index.get("runtime_s"))
    source_fps = index.get("source_fps")
    lines = narration.get("segments") or []
    shots = [s for s in (index.get("shots") or []) if s.get("has_keyframe")]
    # Narration written before this stage needed beat ids does not carry one.
    segment_beats = _segment_beats(cache)
    for line in lines:
        if line.get("beat_id") is None:
            key = line.get("script_i", line.get("i"))
            if key in segment_beats:
                line["beat_id"] = segment_beats[key]

    if not lines:
        raise RuntimeError("there is no narration to lay footage against")
    if not shots:
        raise RuntimeError("the shot index is empty")

    params = settings.params(*PARAM_NAMES)
    params["line_count"] = len(lines)
    params["shot_count"] = len(shots)
    # Chaining the index stage's key means anything that changes the shot index
    # invalidates this stage too. Without it, a cached result produced before the
    # CLIP model arrived stays valid forever and selection silently keeps scoring
    # on time proximity alone even though embeddings are now available.
    # The narrate stage's key matters just as much: a different voice or speaking
    # rate changes every line's duration, and this stage sizes footage to those
    # durations. Serving a cached edit list across that change would leave the
    # video and the audio different lengths.
    narrate_meta = cache.path("narrate.meta.json")
    if narrate_meta.is_file():
        try:
            params["narrate_key"] = read_json(narrate_meta).get("key")
        except Exception:  # noqa: BLE001
            params["narrate_key"] = None
    index_meta = cache.path("index.meta.json")
    if index_meta.is_file():
        try:
            params["index_key"] = read_json(index_meta).get("key")
        except Exception:  # noqa: BLE001 - an unreadable meta just means recompute
            params["index_key"] = None
    params["clip"] = (
        cache.path("clip_index.npy").is_file() and cache.path("query_index.npy").is_file()
    )

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
        middles = starts + durations / 2.0
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

        content_start = _num(index.get("content_start_s"), 0.0)
        content_end = _num(index.get("content_end_s"), runtime_s) or runtime_s

        uses = np.zeros(len(shots), dtype=np.int32)
        timeline: list[dict] = []
        unfilled = 0
        weak_matches = 0
        # Where the previous line's footage ended, so a line continuing the same
        # beat picks up from there instead of cutting to somewhere else.
        last_story_time: float | None = None
        last_end: float | None = None
        # Every line against every shot, computed once. The windows need it to
        # place each line of a group on the moment it describes, and the loop
        # below then rescales one row at a time for scoring.
        raw_similarity = None
        if use_clip:
            raw_similarity = np.zeros((len(lines), len(shots)), dtype=np.float32)
            for shot_position, shot_row in enumerate(rows):
                if shot_row is not None and 0 <= shot_row < image_vectors.shape[0]:
                    wanted = [
                        min(int(line.get("script_i", line.get("i", 0))),
                            query_vectors.shape[0] - 1)
                        for line in lines
                    ]
                    raw_similarity[:, shot_position] = (
                        query_vectors[wanted] @ image_vectors[shot_row]
                    )

        # The stretch of film each line may show, taken from the beat it cites.
        spans = line_windows(
            lines, beats,
            default_s=settings.max_shot_distance_s,
            max_span_s=settings.max_span_s,
            content_start=content_start,
            content_end=content_end,
        )
        # The furthest point already shown. The narration runs in story order,
        # so the picture should too: without this, a line whose window starts
        # behind the previous line's footage replays film the viewer has just
        # seen, which is the repetition that is most obvious while watching.
        shown_until = content_start

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
            # Raised to the end of the beat being narrated, because footage of
            # the beat the words describe cannot spoil that beat.
            ceiling = spans[position][2]

            proximity = np.array(
                [proximity_score(float(m), story_time, settings.proximity_sigma_s)
                 for m in middles],
                dtype=np.float32,
            )

            similarity = np.zeros(len(shots), dtype=np.float32)
            if use_clip and raw_similarity is not None:
                similarity = raw_similarity[position].copy()
                # Cosine similarity for CLIP sits in a narrow positive band, so
                # it is rescaled per line. Otherwise proximity would dominate
                # simply because it already spans zero to one.
                spread = similarity.max() - similarity.min()
                if spread > 1e-6:
                    similarity = (similarity - similarity.min()) / spread

            # How good a shot is, independent of where it sits. Traversal
            # positions clips itself, so mixing in closeness to the line's
            # anchor would drag every clip back to the start of the span and
            # undo the pacing. The other modes add proximity below.
            quality = (
                settings.clip_weight * similarity
                + settings.band_weight * bands
                - settings.dark_penalty * dark
            )
            score = quality + settings.proximity_weight * proximity
            # Anything at or past the ceiling would show a reveal the narration
            # has not reached. This is a hard exclusion, not a penalty.
            score = np.where(starts <= ceiling, score, -1e6)
            # The reuse penalty only discourages. The cap is what stops one
            # striking shot appearing under half the recap.
            score = np.where(uses < settings.max_shot_uses, score, -1e6)
            # A bounded search radius. Proximity is only a score, so without
            # this a confident but spurious match wins from anywhere in the
            # film, which reads as footage that has nothing to do with the
            # narration. Widened only if nothing inside the radius qualifies.
            radius = settings.max_shot_distance_s
            for _ in range(4):
                near = np.abs(middles - story_time) <= radius
                if np.any(np.where(near, score, -1e6) > -1e5):
                    break
                radius *= 2.0
            score = np.where(np.abs(middles - story_time) <= radius, score, -1e6)

            # When nothing in range looks convincing, CLIP has not recognised
            # the scene and its opinion should not drag footage away from the
            # moment being described. Fall back to staying close.
            if use_clip and float(np.max(np.where(score > -1e5, similarity, 0.0))) \
                    < settings.similarity_floor:
                score = np.where(score > -1e5,
                                 settings.proximity_weight * proximity
                                 + settings.band_weight * bands
                                 - settings.dark_penalty * dark,
                                 -1e6)
                weak_matches += 1

            clips: list[dict] = []

            if settings.footage_mode == "traverse":
                # Bounded by the beat this line narrates rather than by a radius
                # around the anchor. A 240 second beat cannot be served from
                # within 90 seconds of its first moment.
                span_lo, span_hi, _ = spans[position]
                # Never go back over film already shown.
                span_lo = max(span_lo, shown_until)
                if span_hi <= span_lo + 0.5:
                    span_hi = min(content_end, span_lo + max(need, settings.clip_target_s))
                in_span = (middles >= span_lo - 10.0) & (middles <= span_hi + 10.0)
                usable = np.where(in_span, quality, -1e6)
                usable = np.where(uses < settings.max_shot_uses, usable, -1e6)
                usable = np.where(starts <= ceiling, usable, -1e6)
                if not np.any(usable > -1e5):
                    usable = np.where(starts <= ceiling, quality, -1e6)
                clips = build_traversal(
                    need=need,
                    span=(span_lo, span_hi),
                    score=usable,
                    uses=uses,
                    starts=starts,
                    middles=middles,
                    durations=durations,
                    ceiling=ceiling,
                    content_start=content_start,
                    content_end=content_end,
                    similarity=similarity if use_clip else None,
                    shots=shots,
                    settings=settings,
                    source_fps=source_fps,
                )
            elif settings.footage_mode == "continuous":
                # Only continue when this line narrates the same beat as the
                # last, otherwise a new beat should cut to its own scene.
                resume = (last_end if last_story_time is not None
                          and abs(story_time - last_story_time) < 0.01 else None)
                clips = build_runs(
                    need=need,
                    score=score,
                    uses=uses,
                    starts=starts,
                    durations=durations,
                    ceiling=ceiling,
                    content_start=content_start,
                    content_end=content_end,
                    similarity=similarity if use_clip else None,
                    shots=shots,
                    settings=settings,
                    source_fps=source_fps,
                    resume_at=resume,
                )
                last_story_time = story_time
                last_end = clips[-1]["src_end"] if clips else None

            if clips:
                shown_until = max(shown_until, max(c["src_end"] for c in clips))

            if clips and settings.footage_mode in ("traverse", "continuous"):
                for clip in clips:
                    # Every shot the footage passes through counts as used, so a
                    # later line is nudged away from repeating the same stretch.
                    touched = np.where((starts < clip["src_end"])
                                       & (starts + durations > clip["src_start"]))[0]
                    uses[touched] += 1

            remaining = need if not clips else 0.0
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
                exact, request = quantise(take, source_fps)
                take = exact
                clips.append({
                    "shot_i": shots[best].get("i", best),
                    "src_start": round(clip_start, 4),
                    # The out point is half a frame past the last wanted frame,
                    # so the demuxer's rounding lands on the intended count.
                    "src_end": round(clip_start + request, 4),
                    "duration": round(exact, 4),
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
            # Any shortfall is absorbed into the last clip, still on a frame
            # boundary, and clamped so it never asks for footage past the end of
            # the film. What remains is under half a frame per line rather than
            # accumulating across the whole recap.
            drift = need - sum(c["duration"] for c in clips)
            if abs(drift) > 0.001:
                last = clips[-1]
                wanted = max(0.0, last["duration"] + drift)
                exact, request = quantise(wanted, source_fps)
                end = min(runtime_s, last["src_start"] + request)
                last["src_end"] = round(end, 4)
                last["duration"] = round(min(exact, end - last["src_start"]), 4)

            timeline.append({
                "i": len(timeline),
                "script_i": line.get("script_i"),
                "start_s": line.get("start_s"),
                # Two durations, deliberately. 'seconds' is how much footage the
                # line needs, which includes the silence after it. 'spoken_s' is
                # how long the voice is actually talking, which is what a
                # subtitle should stay on screen for.
                "seconds": round(need, 3),
                "spoken_s": round(_num(line.get("seconds")), 3),
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
                "footage_mode": settings.footage_mode,
                "weak_matches": weak_matches,
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
