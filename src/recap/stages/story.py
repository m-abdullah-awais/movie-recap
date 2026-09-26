"""Stage 4, story understanding.

The AI reads the movie, it never watches it. This stage turns the dialogue
transcript into a structured account of the plot: who is in it, how it is shaped,
what happens when, what gets planted and paid off, and where the twists sit.

Work is split into two passes. One call per ten minute window extracts concrete
beats from that stretch, then a single synthesis call merges the windows into a
whole-film structure. Splitting it this way keeps each call's attention on a
manageable amount of text, and it means a film's worth of analysis survives one
bad response, because every call is cached individually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .. import ai
from ..cache import Cache, StageOutcome, read_json, run_stage, write_json
from ..config import Settings

STAGE = "story"
VERSION = 2  # bumped: setup, payoff and twist normalisation

STORY_FILE = "story.json"
CALL_DIR = "story_calls"

# Bumping this invalidates every cached call for the stage, which is what a
# prompt change should do.
PROMPT_VERSION = 1

PARAM_NAMES = ("story_chunk_s", "story_overlap_s", "ai_model")

SEQUENCE_SYSTEM = (
    "You are a film story analyst. You read dialogue transcripts and return "
    "structured JSON.\n"
    "Return ONLY a JSON object. No prose, no markdown fences, no explanation."
)

SEQUENCE_TEMPLATE = """Analyse this segment of a film's subtitle transcript.

The segment covers {start_s} to {end_s} seconds of a film that runs {runtime_s} seconds.

Return JSON with exactly these keys:
{{
  "characters": [{{"name": "as spoken or clearly implied", "note": "one short clause"}}],
  "beats": [{{"start_s": int, "end_s": int, "summary": "one sentence, present tense",
              "kind": "setup|escalation|turn|climax|resolution",
              "importance": 1, "characters": ["name"]}}],
  "setups": [{{"at_s": int, "what": "something planted that may pay off later"}}],
  "open_questions": ["string"]
}}

Rules:
- Each line is prefixed with its absolute time in seconds from the start of the
  film, like [1234]. Use those numbers directly for start_s and end_s.
- At most 8 beats. Describe what happens, not how it feels.
- importance is 1 to 5, where 5 means the plot cannot be told without it.
- If a character is only addressed and never named, describe them instead.

TRANSCRIPT:
{transcript}"""

SYNTHESIS_SYSTEM = (
    "You are a film story analyst assembling a whole-film structure from "
    "per-segment notes. Return ONLY a JSON object. No prose, no markdown fences."
)

SYNTHESIS_TEMPLATE = """Below are notes extracted from consecutive segments of one film,
in order. The film runs {runtime_s} seconds.

Merge them into a single whole-film structure. Return JSON with exactly these keys:
{{
  "title_guess": "your best guess at the film, or empty string",
  "logline": "one sentence covering the whole plot, spoilers allowed",
  "cast": [{{"name": "canonical name", "aliases": ["other names used"],
             "role": "protagonist|antagonist|ally|mentor|love_interest|minor",
             "description": "one clause"}}],
  "acts": [{{"act": 1, "start_s": int, "end_s": int, "summary": "one or two sentences"}}],
  "beats": [{{"start_s": int, "end_s": int, "summary": "one sentence, present tense",
              "kind": "setup|escalation|turn|climax|resolution",
              "importance": 1, "characters": ["name"]}}],
  "setups_payoffs": [{{"setup_s": int, "payoff_s": int, "what": "one clause"}}],
  "twists": [{{"at_s": int, "what": "one clause", "severity": 1}}]
}}

Rules:
- Merge duplicate characters that the segments named differently.
- The acts must tile the whole runtime with no gaps.
- Keep 20 to 35 beats, the ones a recap cannot omit, in time order.
- A twist is something whose revelation would spoil the film if said early.
  severity is 1 to 5, where 5 is the central reveal.
- Every timestamp must be within 0 and {runtime_s}.

SEGMENT NOTES:
{notes}"""


@dataclass
class Chunk:
    index: int
    start_s: float
    end_s: float
    lines: list[str]

    @property
    def transcript(self) -> str:
        return "\n".join(self.lines)


def chunk_transcript(cues: list[dict], runtime_s: float, settings: Settings) -> list[Chunk]:
    """Split the dialogue into overlapping windows.

    The overlap is trailing, so a beat straddling a boundary is visible in full
    to the earlier window rather than being cut in half by both. Windows with no
    dialogue are dropped, since there is nothing for the model to read.
    """
    chunks: list[Chunk] = []
    step = max(60, settings.story_chunk_s)
    overlap = max(0, settings.story_overlap_s)

    start = 0.0
    while start < runtime_s:
        end = min(runtime_s, start + step)
        window = [c for c in cues if start <= c["start"] < end + overlap]
        if window:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    start_s=start,
                    end_s=end,
                    # Absolute seconds are given directly so the model never has
                    # to convert a clock time, which it gets wrong past an hour.
                    lines=[f"[{int(c['start'])}] {c['text']}" for c in window],
                )
            )
        start = end

    return chunks


def _dedupe_beats(beats: list[dict]) -> list[dict]:
    """Drop beats that repeat one already kept at nearly the same time.

    The window overlap means the same moment can be described twice. Keeping
    whichever the synthesis pass ranked more important is good enough.
    """
    ordered = sorted(
        beats,
        key=lambda b: (_num(b.get("start_s")), -_num(b.get("importance"))),
    )
    kept: list[dict] = []
    for beat in ordered:
        start = _num(beat.get("start_s"))
        if kept and abs(start - _num(kept[-1].get("start_s"))) < 3.0:
            continue
        kept.append(beat)
    return kept


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_beats(beats: list[dict], runtime_s: float) -> list[dict]:
    """Keep only beats with a usable position on the timeline.

    A hallucinated timestamp outside the runtime would send stage 6 looking for
    footage that does not exist, so it is discarded here rather than carried
    forward.
    """
    out: list[dict] = []
    for beat in beats:
        start = _num(beat.get("start_s"), -1.0)
        if not 0.0 <= start <= runtime_s:
            continue
        end = _num(beat.get("end_s"), start)
        if end <= start:
            end = min(runtime_s, start + 5.0)
        out.append(
            {
                "i": len(out),
                "start_s": round(start, 2),
                "end_s": round(min(end, runtime_s), 2),
                "summary": str(beat.get("summary") or "").strip(),
                "kind": str(beat.get("kind") or "escalation"),
                "importance": int(max(1, min(5, _num(beat.get("importance"), 3)))),
                "characters": [str(c) for c in (beat.get("characters") or [])],
            }
        )
    for position, beat in enumerate(out):
        beat["i"] = position
    return out


def _normalise_pairs(pairs: list[dict], runtime_s: float) -> list[dict]:
    """Clean up setup and payoff pairs.

    Models occasionally return the two timestamps the wrong way round, which
    would describe a payoff that lands before the thing it pays off. Since the
    pair is a claim about order, the fix is to sort the two values rather than
    discard a genuine observation. Pairs outside the runtime are dropped.
    """
    out: list[dict] = []
    for pair in pairs:
        setup = _num(pair.get("setup_s"), -1.0)
        payoff = pair.get("payoff_s")
        payoff_s = _num(payoff, -1.0) if payoff is not None else None

        if not 0.0 <= setup <= runtime_s:
            continue
        if payoff_s is not None and not 0.0 <= payoff_s <= runtime_s:
            payoff_s = None
        if payoff_s is not None and payoff_s < setup:
            setup, payoff_s = payoff_s, setup

        out.append({
            "setup_s": round(setup, 2),
            "payoff_s": round(payoff_s, 2) if payoff_s is not None else None,
            "what": str(pair.get("what") or "").strip(),
        })
    return sorted(out, key=lambda p: p["setup_s"])


def _normalise_twists(twists: list[dict], runtime_s: float) -> list[dict]:
    """Keep twists that sit on the timeline, in time order.

    Stage 8 derives each narration segment's spoiler ceiling from these, so a
    twist with an invented timestamp would let footage of a reveal appear before
    the narration reaches it.
    """
    out: list[dict] = []
    for twist in twists:
        at = _num(twist.get("at_s"), -1.0)
        if not 0.0 <= at <= runtime_s:
            continue
        out.append({
            "at_s": round(at, 2),
            "what": str(twist.get("what") or "").strip(),
            "severity": int(max(1, min(5, _num(twist.get("severity"), 3)))),
        })
    return sorted(out, key=lambda t: t["at_s"])


def _mechanical_story(sequences: list[dict], runtime_s: float) -> dict:
    """Assemble a story without the synthesis call.

    Used when synthesis fails. The result is flatter, with no acts and no twist
    detection, but stage 5 can still write a script from it, which is the point
    of degrading rather than stopping.
    """
    beats: list[dict] = []
    names: dict[str, str] = {}
    setups: list[dict] = []
    for sequence in sequences:
        beats.extend(sequence.get("beats") or [])
        for character in sequence.get("characters") or []:
            name = str(character.get("name") or "").strip()
            if name and name not in names:
                names[name] = str(character.get("note") or "")
        for setup in sequence.get("setups") or []:
            setups.append({"setup_s": _num(setup.get("at_s")), "payoff_s": None,
                           "what": str(setup.get("what") or "")})

    return {
        "title_guess": "",
        "logline": "",
        "cast": [{"name": n, "aliases": [], "role": "minor", "description": d}
                 for n, d in names.items()],
        "acts": [{"act": 1, "start_s": 0, "end_s": round(runtime_s, 2),
                  "summary": "Act structure unavailable, synthesis did not complete."}],
        "beats": _clamp_beats(_dedupe_beats(beats), runtime_s),
        "setups_payoffs": setups,
        "twists": [],
    }


def run(
    cache: Cache,
    settings: Settings,
    *,
    transcript_file: str = "transcript.json",
    engine: ai.Engine | None = None,
    force: bool = False,
    quiet: bool = False,
) -> StageOutcome:
    transcript_path = cache.path(transcript_file)
    if not transcript_path.is_file():
        raise RuntimeError(
            "no transcript found, so there is no dialogue to read. Run the ingest stage first."
        )

    transcript = read_json(transcript_path)
    cues = transcript.get("cues") or []
    runtime_s = _num(transcript.get("runtime_s"))
    if not cues:
        raise RuntimeError("the transcript has no dialogue cues to analyse")

    # Resolved before the cache is consulted, because which engine wrote an
    # answer is part of what makes it reusable.
    picked = engine or ai.select_engine()

    params = settings.params(*PARAM_NAMES)
    params["prompt_version"] = PROMPT_VERSION
    params["cue_count"] = len(cues)
    params["ai_engine"] = picked.name

    def work() -> dict:
        call_dir = cache.dir / CALL_DIR
        chunks = chunk_transcript(cues, runtime_s, settings)
        if not quiet:
            print(f"  reading {len(chunks)} segments of dialogue through {picked.description}")

        calls = [
            ai.Call(
                tag=f"seq{chunk.index:03d}",
                system=SEQUENCE_SYSTEM,
                prompt=SEQUENCE_TEMPLATE.format(
                    start_s=int(chunk.start_s),
                    end_s=int(chunk.end_s),
                    runtime_s=int(runtime_s),
                    transcript=chunk.transcript,
                ),
            )
            for chunk in chunks
        ]

        def progress(reply: ai.Reply, done: int, total: int) -> None:
            if quiet:
                return
            mark = "cached" if reply.cached else ("failed" if reply.error else "ok")
            print(f"    segment {done}/{total}  {mark}")

        replies = ai.ask_many(
            calls,
            cache_dir=call_dir,
            version=PROMPT_VERSION,
            engine=picked,
            model=settings.ai_model or None,
            timeout=settings.ai_timeout_s,
            concurrency=settings.ai_concurrency,
            on_done=progress,
        )

        sequences: list[dict] = []
        failures: list[str] = []
        cost = 0.0
        for chunk, reply in zip(chunks, replies):
            cost += reply.cost_usd
            if reply.ok:
                sequences.append({"index": chunk.index, "start_s": chunk.start_s,
                                  "end_s": chunk.end_s, **reply.data})
            else:
                failures.append(f"segment {chunk.index}: {reply.error[:120]}")

        if not sequences:
            raise RuntimeError(
                "every segment failed, so no story could be built. "
                + (failures[0] if failures else "")
            )

        # Compact notes for synthesis. The full transcript is deliberately not
        # resent: the point of the first pass was to reduce it.
        notes = []
        for sequence in sequences:
            notes.append(
                f"--- segment {sequence['index']}, "
                f"{int(sequence['start_s'])}s to {int(sequence['end_s'])}s ---\n"
                + json.dumps(
                    {k: sequence.get(k) for k in
                     ("characters", "beats", "setups", "open_questions")},
                    ensure_ascii=False,
                )
            )

        synthesis = ai.ask(
            ai.Call(
                tag="synthesis",
                system=SYNTHESIS_SYSTEM,
                prompt=SYNTHESIS_TEMPLATE.format(
                    runtime_s=int(runtime_s), notes="\n".join(notes)
                ),
            ),
            cache_dir=call_dir,
            version=PROMPT_VERSION,
            engine=picked,
            model=settings.ai_model or None,
            timeout=settings.ai_timeout_s,
        )
        cost += synthesis.cost_usd

        degraded = None
        if synthesis.ok:
            story = dict(synthesis.data)
        else:
            degraded = "synthesis_failed"
            failures.append(f"synthesis: {synthesis.error[:160]}")
            story = _mechanical_story(sequences, runtime_s)
            if not quiet:
                print("  synthesis did not complete, assembling the story mechanically")

        beats = _clamp_beats(_dedupe_beats(story.get("beats") or []), runtime_s)
        if not beats:
            # Synthesis returned nothing usable on the timeline. The per-segment
            # beats are still there, so fall back to them rather than give up.
            degraded = degraded or "synthesis_beats_unusable"
            beats = _clamp_beats(
                _dedupe_beats(_mechanical_story(sequences, runtime_s)["beats"]), runtime_s
            )
        payload = {
            "schema": 1,
            "source_id": cache.sid,
            "source_name": cache.source.name,
            "runtime_s": round(runtime_s, 2),
            "dialogue_origin": transcript.get("origin"),
            "title_guess": str(story.get("title_guess") or ""),
            "logline": str(story.get("logline") or ""),
            "cast": story.get("cast") or [],
            "acts": story.get("acts") or [],
            "beats": beats,
            "setups_payoffs": _normalise_pairs(story.get("setups_payoffs") or [], runtime_s),
            "twists": _normalise_twists(story.get("twists") or [], runtime_s),
            "sequences": sequences,
            "stats": {
                "segment_count": len(chunks),
                "segments_ok": len(sequences),
                "segment_failures": failures,
                "cast_count": len(story.get("cast") or []),
                "act_count": len(story.get("acts") or []),
                "beat_count": len(beats),
                "twist_count": len(_normalise_twists(story.get("twists") or [], runtime_s)),
                "cost_usd": round(cost, 4),
                "degraded": degraded,
            },
        }
        write_json(cache.path(STORY_FILE), payload)
        return {"story": STORY_FILE, **payload["stats"]}

    def summarize(meta: dict) -> str:
        bits = [
            f"{meta.get('beat_count', 0)} beats",
            f"{meta.get('cast_count', 0)} cast",
            f"{meta.get('act_count', 0)} acts",
            f"{meta.get('twist_count', 0)} twists",
        ]
        if meta.get("cost_usd"):
            bits.append(f"${meta['cost_usd']:.2f}")
        failed = len(meta.get("segment_failures") or [])
        if failed:
            bits.append(f"{failed} segment(s) failed")
        if meta.get("degraded"):
            bits.append(f"DEGRADED {meta['degraded']}")
        return ", ".join(bits)

    return run_stage(
        cache, STAGE, VERSION, params, [STORY_FILE], work,
        force=force, summarize=summarize,
    )
