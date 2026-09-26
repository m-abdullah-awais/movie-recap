"""Stage 5, narration script.

The script is written before any footage is chosen. This is what makes
synchronisation automatic later: once a line of narration has been spoken and
measured, its duration says exactly how much video must sit behind it, so no
forced aligner is ever needed.

One Claude call per act, run in order rather than in parallel, because each act's
narration has to follow on from the last without repeating it.

Spoiler control is computed here in Python rather than asked of the model. Each
segment gets a ceiling on how late in the film its footage may come from, derived
from the next twist the narration has not reached yet.
"""

from __future__ import annotations

import re

from .. import ai
from ..cache import Cache, StageOutcome, read_json, run_stage, write_json
from ..config import Settings

STAGE = "script"
VERSION = 2  # bumped: lines anchor to a cited beat, not an invented time

SCRIPT_FILE = "script.json"
CALL_DIR = "script_calls"

PROMPT_VERSION = 2  # the prompt now asks for a beat id, so cached calls are stale

PARAM_NAMES = (
    "script_target_words", "script_words_per_minute",
    "spoiler_lookahead_s", "ai_model",
)

SYSTEM = (
    "You write narration for movie recap videos. The narration is spoken aloud "
    "by a synthetic voice over clips from the film.\n"
    "Return ONLY a JSON object. No prose, no markdown fences, no explanation."
)

TEMPLATE = """Write the narration for one act of a movie recap video.

FILM: {title}
PLOT IN ONE LINE: {logline}

CAST:
{cast}

You are writing act {act} of {act_count}, covering {start_s}s to {end_s}s of the film.
Write about {target_words} words for this act.

{previous}

BEATS TO COVER, in order:
{beats}

{spoilers}

Return JSON with exactly this shape:
{{
  "segments": [
    {{"beat_id": int, "narration": "string", "visual_query": "string",
      "characters": ["name"]}}
  ]
}}

Rules for narration:
- It is spoken aloud. Write plain sentences. No brackets, no markdown, no stage
  directions, no headings, no lists.
- Never use an em dash or an en dash. Use a comma or a full stop.
- Present tense, active voice. Never address the viewer, and never say "we" or "you".
- 15 to 45 words per segment. Aim for one clear idea per segment.
- Do not invent events that are not in the beats.

Rules for the other fields:
- beat_id is the number of the beat this segment is narrating, taken from the
  list above. Use the beats in order. Several consecutive segments may share one
  beat when it needs more than one sentence, but never go backwards.
- visual_query describes what should be ON SCREEN, in plain visual words, so that
  a shot can be matched to it. Describe the picture, not the plot. Name people
  only by what they look like or are doing. A good example is "a teenage boy
  running from police through a suburban street at night". A bad example is
  "Sean realises the signal is real".
- characters lists the cast names visible in that moment, using the names above."""

SPOILER_BLOCK = """WITHHOLD THESE REVEALS. They happen later in the film and must not be
mentioned, hinted at, or implied anywhere in this act:
{items}"""

# Narration is spoken, so anything that cannot be pronounced is stripped rather
# than read out as punctuation.
_UNSPEAKABLE_RE = re.compile(r"[\[\]{}<>*_#`|]")
# A raw string, so the two dash characters stay as escapes here and are resolved
# by the regex engine instead. That keeps a literal em or en dash out of every
# file in this project, which is a standing rule, without a stray backslash
# escape that Python would warn about.
_DASH_RE = re.compile(r"\s*[\u2014\u2013]\s*")
_WS_RE = re.compile(r"\s+")


def clean_narration(text: str) -> str:
    """Make a line safe to speak and to write into a subtitle file.

    Em dashes are replaced rather than removed because they usually join two
    clauses, and dropping one would run the clauses together.
    """
    cleaned = _DASH_RE.sub(", ", str(text or ""))
    cleaned = _UNSPEAKABLE_RE.sub("", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def spoiler_ceiling(story_time: float, twists: list[dict], runtime_s: float,
                    lookahead_s: float) -> float:
    """Latest source timestamp this segment's footage may come from.

    Two limits apply. Footage must not reach the next twist the narration has not
    covered yet, otherwise a shot would give away a reveal before it is told. It
    is also capped a fixed distance ahead of the narration's current position, so
    an early segment is not backed by a shot from the finale even when no twist
    sits in between.
    """
    ceiling = min(runtime_s, story_time + max(0.0, lookahead_s))
    for twist in twists:
        at = _num(twist.get("at_s"), -1.0)
        if at > story_time:
            # A small margin keeps the shot just before the reveal, not on it.
            ceiling = min(ceiling, max(story_time, at - 2.0))
            break
    return round(max(story_time, ceiling), 2)


def _act_windows(story: dict, runtime_s: float) -> list[dict]:
    """Acts to write, falling back to thirds when the story has no act structure."""
    acts = []
    for act in story.get("acts") or []:
        start = _num(act.get("start_s"), -1.0)
        end = _num(act.get("end_s"), -1.0)
        if 0.0 <= start < end <= runtime_s + 1.0:
            acts.append({"act": int(_num(act.get("act"), len(acts) + 1)),
                         "start_s": start, "end_s": min(end, runtime_s),
                         "summary": str(act.get("summary") or "")})
    if acts:
        return sorted(acts, key=lambda a: a["start_s"])

    third = runtime_s / 3.0
    return [
        {"act": i + 1, "start_s": i * third, "end_s": (i + 1) * third, "summary": ""}
        for i in range(3)
    ]


def _format_cast(cast: list[dict]) -> str:
    lines = []
    for member in cast[:12]:
        name = str(member.get("name") or "").strip()
        if not name:
            continue
        role = str(member.get("role") or "")
        description = str(member.get("description") or "")
        lines.append(f"- {name} ({role}): {description}")
    return "\n".join(lines) or "- not identified"


def _format_beats(beats: list[dict]) -> str:
    """Beats as a numbered list, deliberately without timestamps.

    Segments cite a beat number and the code looks up that beat's real time.
    Asking for a timestamp instead produced anchors that merely looked
    plausible: measured on a 94 minute film, only 22 percent of lines landed
    within 5 seconds of the beat they were describing, the median was 31 seconds
    out and the worst 95. With lines about 55 seconds apart, that routinely
    pointed a line at its neighbour's scene. Showing no timestamps also removes
    the temptation to invent one.
    """
    return "\n".join(
        f"- beat {int(_num(b.get('i')))}: {b.get('summary')}" for b in beats
    ) or "- no beats recorded for this stretch"


def run(
    cache: Cache,
    settings: Settings,
    *,
    engine: ai.Engine | None = None,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    story_path = cache.path("story.json")
    if not story_path.is_file():
        raise RuntimeError("no story.json found. Run the story stage first.")

    story = read_json(story_path)
    runtime_s = _num(story.get("runtime_s"))
    beats = story.get("beats") or []
    twists = sorted(story.get("twists") or [], key=lambda t: _num(t.get("at_s")))
    if not beats:
        raise RuntimeError("the story has no beats, so there is nothing to narrate")

    picked = engine or ai.select_engine()

    params = settings.params(*PARAM_NAMES)
    params["prompt_version"] = PROMPT_VERSION
    params["beat_count"] = len(beats)
    params["ai_engine"] = picked.name

    def work() -> dict:
        call_dir = cache.dir / CALL_DIR
        acts = _act_windows(story, runtime_s)
        title = story.get("title_guess") or cache.source.stem
        cast_block = _format_cast(story.get("cast") or [])

        # Words are shared out by how much of the film each act covers, so a long
        # middle act is not given the same budget as a short opening one.
        span_total = sum(a["end_s"] - a["start_s"] for a in acts) or 1.0

        if not quiet:
            print(f"  writing narration for {len(acts)} acts, "
                  f"target {settings.script_target_words} words")

        segments: list[dict] = []
        failures: list[str] = []
        cost = 0.0
        unresolved = 0
        previous_tail = ""

        for act in acts:
            act_beats = [b for b in beats
                         if act["start_s"] <= _num(b.get("start_s")) < act["end_s"]]
            if not act_beats:
                continue

            act_first = len(segments)
            share = (act["end_s"] - act["start_s"]) / span_total
            target = max(150, int(settings.script_target_words * share))

            # Only twists after this act begins are withheld. A reveal the
            # narration has already passed is fair game.
            future = [t for t in twists if _num(t.get("at_s")) > act["start_s"]]
            spoilers = (
                SPOILER_BLOCK.format(
                    items="\n".join(f"- {t.get('what')}" for t in future[:10])
                )
                if future else "There are no reveals left to withhold in this act."
            )

            prompt = TEMPLATE.format(
                title=title,
                logline=story.get("logline") or "",
                cast=cast_block,
                act=act["act"],
                act_count=len(acts),
                start_s=int(act["start_s"]),
                end_s=int(act["end_s"]),
                target_words=target,
                previous=previous_tail or "This is the opening of the recap.",
                beats=_format_beats(act_beats),
                spoilers=spoilers,
            )

            reply = ai.ask(
                ai.Call(tag=f"act{act['act']:02d}", system=SYSTEM, prompt=prompt),
                cache_dir=call_dir,
                version=PROMPT_VERSION,
                engine=picked,
                model=settings.ai_model or None,
                timeout=settings.ai_timeout_s,
            )
            cost += reply.cost_usd

            if not reply.ok:
                failures.append(f"act {act['act']}: {reply.error[:140]}")
                if not quiet:
                    print(f"    act {act['act']}  failed")
                continue

            written = 0
            # Beats for this act, by the id the model was shown.
            by_id = {int(_num(b.get("i"), -1)): b for b in act_beats}
            fallback_order = list(act_beats)
            for raw in reply.data.get("segments") or []:
                narration = clean_narration(raw.get("narration"))
                query = clean_narration(raw.get("visual_query"))
                if not narration or not query:
                    continue
                # The anchor comes from the cited beat, not from the model.
                beat_id = raw.get("beat_id")
                beat = by_id.get(int(_num(beat_id, -1))) if beat_id is not None else None
                if beat is None:
                    # An unusable id means the prompt drifted. Fall back to the
                    # next beat in order and count it, so a regression shows up
                    # in the stats instead of silently degrading the anchors.
                    beat = fallback_order[min(len(segments) - act_first,
                                              len(fallback_order) - 1)]
                    unresolved += 1
                story_time = _num(beat.get("start_s"), act["start_s"])
                story_time = min(max(story_time, act["start_s"]), act["end_s"])
                # Narration must move forward through the film. A segment that
                # goes backwards would show footage the recap has left behind.
                if segments:
                    story_time = max(story_time, segments[-1]["story_time"])
                words = len(narration.split())
                segments.append({
                    "i": len(segments),
                    "act": act["act"],
                    "story_time": round(story_time, 2),
                    "narration": narration,
                    "visual_query": query,
                    "beat_id": int(_num(beat.get("i"), -1)),
                    "characters": [str(c) for c in (raw.get("characters") or [])],
                    "spoiler_ceiling": spoiler_ceiling(
                        story_time, twists, runtime_s, settings.spoiler_lookahead_s
                    ),
                    "word_count": words,
                    "est_seconds": round(words / max(1, settings.script_words_per_minute) * 60.0, 2),
                })
                written += words

            if not quiet:
                print(f"    act {act['act']}  {written} words"
                      f"{'  cached' if reply.cached else ''}")

            tail = " ".join(s["narration"] for s in segments[-2:])
            previous_tail = (
                "The previous act's narration ended with these lines. Continue from "
                f"them without repeating them:\n{tail}"
            )

        if not segments:
            raise RuntimeError(
                "no narration was produced. " + (failures[0] if failures else "")
            )

        total_words = sum(s["word_count"] for s in segments)
        est_seconds = sum(s["est_seconds"] for s in segments)
        payload = {
            "schema": 1,
            "source_id": cache.sid,
            "source_name": cache.source.name,
            "runtime_s": round(runtime_s, 2),
            "title": title,
            "logline": story.get("logline") or "",
            "target_words": settings.script_target_words,
            "segments": segments,
            "stats": {
                "segment_count": len(segments),
                "word_count": total_words,
                "estimated_minutes": round(est_seconds / 60.0, 2),
                "act_count": len(acts),
                "act_failures": failures,
                "unresolved_beat_ids": unresolved,
                "mean_words_per_segment": round(total_words / len(segments), 1),
                "cost_usd": round(cost, 4),
                "degraded": "act_failures" if failures else None,
            },
        }
        write_json(cache.path(SCRIPT_FILE), payload)
        return {"script": SCRIPT_FILE, **payload["stats"]}

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('segment_count', 0)} segments",
            f"{meta.get('word_count', 0)} words",
            f"~{meta.get('estimated_minutes', 0)} min",
        ]
        if meta.get("cost_usd"):
            bits.append(f"${meta['cost_usd']:.2f}")
        if meta.get("degraded"):
            bits.append(f"DEGRADED {meta['degraded']}")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, [SCRIPT_FILE], work,
        force=force, summarize=summarize,
    )
