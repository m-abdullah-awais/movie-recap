"""Command line interface for the analysis stages.

Every stage is independently runnable so that a failure or a tuning question can
be investigated without re-running the expensive ones.
"""

from __future__ import annotations

import dataclasses
import json as jsonlib
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from . import claude, config, ffmpeg, models, probe
from .cache import Cache, StageOutcome, read_json, source_id
from .config import CACHE_ROOT, Settings
from .stages import (
    index, ingest, narrate, proxy, render, scenemap, script, select, story,
)
from .timing import Report, format_hms

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local movie recap generator. The full pipeline, stages 1 to 9.",
    # Typer's framed, syntax-highlighted traceback is far harder to read in a
    # terminal than a plain one, and it buries the actual message. Expected
    # failures are caught and reported as a single line; anything else raises a
    # normal Python traceback.
    pretty_exceptions_enable=False,
)

ALL_STAGES = (
    "ingest", "proxy", "scenemap", "story", "script",
    "index", "narrate", "select", "render",
)


def entry_point() -> str:
    """How the user invoked this tool, for the hints printed alongside errors.

    Taken from the command line rather than hardcoded, so moving or renaming the
    entry script cannot leave the printed advice pointing at a path that no
    longer exists. Falls back to the bare filename when the script sits outside
    the working directory, since a relative path would be meaningless there.
    """
    try:
        return str(Path(sys.argv[0]).resolve().relative_to(Path.cwd()))
    except (ValueError, OSError):
        return Path(sys.argv[0]).name or "analyze.py"


# Shared option definitions, declared once so every command stays consistent.
MovieArg = typer.Argument(
    None,
    help="Path to the movie. Omit it to use the single movie in the input folder.",
)
ForceOpt = typer.Option(False, "--force", help="Ignore the cache and recompute every stage.")
ForceStageOpt = typer.Option(None, "--force-stage", help="Recompute only these stages.")
HeightOpt = typer.Option(None, "--proxy-height", help="Proxy height in pixels. Default 480.")
ThresholdOpt = typer.Option(None, "--threshold", help="Scene change threshold, 0 to 100. Default 8.")
NoQsvOpt = typer.Option(False, "--no-qsv", help="Disable Quick Sync and decode in software.")
WithProxyOpt = typer.Option(
    False, "--with-proxy",
    help="Also write a full-film 480p proxy. Slower, and only useful for inspection.",
)
CacheDirOpt = typer.Option(None, "--cache-dir", help="Override the cache root.")
JsonOpt = typer.Option(False, "--json", help="Emit machine readable output.")
QuietOpt = typer.Option(False, "--quiet", "-q", help="Suppress progress output.")


def _settings(
    height: int | None,
    threshold: float | None,
    no_qsv: bool,
    with_proxy: bool = False,
) -> Settings:
    changes: dict[str, object] = {}
    if height is not None:
        changes["proxy_height"] = height
    if threshold is not None:
        changes["scene_threshold"] = threshold
    if no_qsv:
        changes["allow_qsv"] = False
    if with_proxy:
        changes["write_proxy_video"] = True
    return dataclasses.replace(Settings(), **changes) if changes else Settings()


def _forced(stage: str, force_all: bool, force_stages: list[str] | None) -> bool:
    return force_all or (bool(force_stages) and stage in force_stages)


def _resolve_movie(movie: Path | None) -> Path:
    """Turn an optional argument into a real file.

    With no argument, the drop-in input folder is used. Several movies there is
    treated as an error rather than resolved by guessing, because picking the
    wrong one costs minutes of encoding before the mistake becomes obvious.
    """
    if movie is not None:
        movie = movie.expanduser()
        if not movie.is_file():
            typer.secho(f"not a file: {movie}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        return movie

    found = config.discover_input()
    if len(found) == 1:
        typer.secho(f"using {found[0].name} from the input folder", fg=typer.colors.CYAN)
        return found[0]

    if not found:
        typer.secho(
            f"no movie found. Put one in {config.INPUT_DIR}, or pass its path as an argument.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)

    typer.secho(
        f"{len(found)} movies are in the input folder, so name the one you want:",
        fg=typer.colors.RED, err=True,
    )
    for item in found:
        typer.secho(f'  {entry_point()} all "input/{item.name}"', err=True)
    raise typer.Exit(2)


def _open(movie: Path | None, cache_dir: Path | None) -> tuple[Path, Cache, dict, float]:
    movie = _resolve_movie(movie)
    probe_data = ffmpeg.probe(movie)
    if not ffmpeg.streams(probe_data, "video"):
        typer.secho(f"{movie.name} has no video stream", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    sid = source_id(movie)
    cache = Cache(cache_dir or CACHE_ROOT, sid, movie)
    return movie, cache, probe_data, ffmpeg.duration_seconds(probe_data)


def _validate_stages(names: list[str] | None) -> list[str] | None:
    if not names:
        return None
    unknown = [n for n in names if n not in ALL_STAGES]
    if unknown:
        typer.secho(
            f"unknown stage(s): {', '.join(unknown)}. Valid: {', '.join(ALL_STAGES)}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)
    return names


class Progress:
    """Prints each stage's heading before it runs and its result after.

    The final timing table only appears at the end, which is no help during a
    twenty minute run. This gives a line per stage as it happens, with the time
    that stage took and the time spent so far, so it is always clear how much is
    done and how much is left.
    """

    def __init__(self, report: Report, total: int, quiet: bool):
        self.report = report
        self.total = total
        self.quiet = quiet
        self.index = 0
        self.started = time.monotonic()

    def start(self, name: str, note: str = "") -> None:
        self.index += 1
        if self.quiet:
            return
        label = f"[{self.index}/{self.total}] {name}"
        print()
        print(f"{label}{'  ' + note if note else ''}")

    def skip(self, name: str, why: str) -> None:
        self.index += 1
        if not self.quiet:
            print()
            print(f"[{self.index}/{self.total}] {name}  skipped, {why}")

    def done(self, outcome: StageOutcome) -> StageOutcome:
        self.report.add(outcome)
        if self.quiet:
            return outcome
        word = {"hit": "cached", "computed": "done", "failed": "FAILED"}.get(
            outcome.status, outcome.status
        )
        elapsed = time.monotonic() - self.started
        detail = f"  {outcome.summary}" if outcome.summary else ""
        line = (f"      {word} in {format_hms(outcome.seconds)}"
                f"   elapsed {format_hms(elapsed)}{detail}")
        if outcome.failed:
            typer.secho(line, fg=typer.colors.RED)
        else:
            print(line)
        return outcome


def _run_ingest(cache, movie, probe_data, settings, report, *, force_all,
                force_stages, quiet, progress=None):
    """Ingest, pulling the proxy forward when speech recognition is needed.

    Subtitle extraction needs nothing, but recognition needs the proxy audio, so
    the proxy is run first in that case only. Because the proxy is cached, a
    later explicit call to it costs nothing.

    A failure to obtain dialogue is recorded rather than raised. The proxy and
    the shot map do not depend on dialogue, so there is no reason to throw away
    minutes of completed encoding because a speech model could not be
    downloaded.
    """
    wav = cache.path(proxy.WAV_FILE)
    proxy_outcome = None
    force_ingest = _forced("ingest", force_all, force_stages)

    def attempt(wav_path: Path | None) -> StageOutcome:
        return ingest.run(
            cache, movie, probe_data, settings,
            wav=wav_path, force=force_ingest, quiet=quiet,
        )

    def failed(exc: Exception) -> StageOutcome:
        if not quiet:
            # Kept short here. The full reason is printed once in the final
            # report, so stating it at both points would just be noise.
            typer.secho(
                "  could not obtain dialogue, continuing with the remaining stages",
                fg=typer.colors.YELLOW,
            )
        return StageOutcome("ingest", "failed", 0.0, {}, "no dialogue obtained", str(exc))

    try:
        # When the proxy already exists the audio is available immediately, so
        # recognition can be attempted on this first call. Both call sites
        # therefore need the same failure handling.
        outcome = attempt(wav if wav.is_file() else None)
    except ingest.NeedsAudio:
        # Listed before RuntimeError because it is a subclass of it.
        if not quiet:
            print("  no usable subtitles, producing the proxy first so audio is available")
        proxy_outcome = proxy.run(
            cache, movie, probe_data, settings,
            force=_forced("proxy", force_all, force_stages), quiet=quiet,
        )
        if progress is not None:
            progress.done(proxy_outcome)
        else:
            report.add(proxy_outcome)
        try:
            outcome = attempt(wav)
        except RuntimeError as exc:
            outcome = failed(exc)
    except RuntimeError as exc:
        outcome = failed(exc)

    if progress is not None:
        progress.done(outcome)
    else:
        report.add(outcome)
    return outcome, proxy_outcome


@app.command("all")
def run_all(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    force_stage: Optional[list[str]] = ForceStageOpt,
    proxy_height: Optional[int] = HeightOpt,
    threshold: Optional[float] = ThresholdOpt,
    no_qsv: bool = NoQsvOpt,
    with_proxy: bool = WithProxyOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    json: bool = JsonOpt,
    quiet: bool = QuietOpt,
):
    """Run the whole pipeline, stages 1 to 9, then print per stage timings."""
    force_stage = _validate_stages(force_stage)
    settings = _settings(proxy_height, threshold, no_qsv, with_proxy)
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)

    if not quiet:
        print(f"{movie.name}  ({format_hms(duration)}, source id {cache.sid[:12]})")
        print(f"cache: {cache.dir}")

    steps = Progress(report, 9, quiet)

    steps.start("ingest", "reading the film's dialogue")
    ingest_outcome, proxy_outcome = _run_ingest(
        cache, movie, probe_data, settings, report,
        force_all=force, force_stages=force_stage, quiet=quiet, progress=steps,
    )

    if proxy_outcome is None:
        steps.start("proxy", "reading the film once, the longest stage")
        steps.done(proxy.run(
            cache, movie, probe_data, settings,
            force=_forced("proxy", force, force_stage), quiet=quiet,
        ))
        proxy_outcome = report.outcomes[-1]
    else:
        steps.index += 1  # the proxy already ran, pulled forward by ingest

    steps.start("scenemap", "finding the shot boundaries")
    steps.done(scenemap.run(
        cache, proxy_outcome.meta, settings,
        force=_forced("scenemap", force, force_stage), quiet=quiet,
    ))

    # Stage 4 needs dialogue. When ingest could not produce any there is nothing
    # to read, so it is skipped rather than reported as a failure.
    if ingest_outcome.failed:
        steps.skip("story", "no dialogue was obtained")
    else:
        steps.start("story", "Claude reads the plot")
        try:
            steps.done(story.run(
                cache, settings,
                force=_forced("story", force, force_stage), quiet=quiet,
            ))
        except (RuntimeError, claude.ClaudeUnavailable) as exc:
            steps.done(StageOutcome("story", "failed", 0.0, {}, "no story built", str(exc)))

    # Stage 5 narrates the story, so it only runs when there is one to narrate.
    if cache.path(story.STORY_FILE).is_file():
        steps.start("script", "Claude writes the narration")
        try:
            steps.done(script.run(
                cache, settings,
                force=_forced("script", force, force_stage), quiet=quiet,
            ))
        except (RuntimeError, claude.ClaudeUnavailable) as exc:
            steps.done(
                StageOutcome("script", "failed", 0.0, {}, "no script written", str(exc))
            )
    else:
        steps.skip("script", "no story to narrate")

    # Stage 6 indexes the regions the script points at, so it needs a script.
    if cache.path(script.SCRIPT_FILE).is_file():
        steps.start("index", "matching shots to the narration")
        try:
            steps.done(index.run(
                cache, movie, settings,
                force=_forced("index", force, force_stage), quiet=quiet,
            ))
        except RuntimeError as exc:
            steps.done(StageOutcome("index", "failed", 0.0, {}, "no shot index", str(exc)))
    else:
        steps.skip("index", "no script to index against")

    # Stages 7 to 9 each depend on the previous one having produced its file,
    # so a failure earlier stops the chain without raising.
    if cache.path(index.SHOTS_FILE).is_file():
        steps.start("narrate", "speaking the narration")
        try:
            steps.done(narrate.run(
                cache, settings,
                force=_forced("narrate", force, force_stage), quiet=quiet,
            ))
        except (RuntimeError, narrate.NoVoice) as exc:
            steps.done(StageOutcome(
                "narrate", "failed", 0.0, {}, "no narration audio", str(exc)))
    else:
        steps.skip("narrate", "no shot index")

    if cache.path(narrate.NARRATION_FILE).is_file():
        steps.start("select", "choosing footage for every line")
        try:
            steps.done(select.run(
                cache, settings,
                force=_forced("select", force, force_stage), quiet=quiet,
            ))
        except RuntimeError as exc:
            steps.done(StageOutcome(
                "select", "failed", 0.0, {}, "no edit decision list", str(exc)))
    else:
        steps.skip("select", "no narration audio")

    if cache.path(select.EDL_FILE).is_file():
        steps.start("render", "encoding the finished video")
        try:
            steps.done(render.run(
                cache, movie, probe_data, settings,
                force=_forced("render", force, force_stage), quiet=quiet,
            ))
        except (RuntimeError, ffmpeg.FfmpegError) as exc:
            steps.done(StageOutcome(
                "render", "failed", 0.0, {}, "no final video", str(exc)))
    else:
        steps.skip("render", "no edit decision list")

    timings = report.persist(cache.dir)

    if json:
        typer.echo(jsonlib.dumps({
            "source": str(movie),
            "source_id": cache.sid,
            "cache_dir": str(cache.dir),
            "total_seconds": round(report.total_seconds, 2),
            "stages": [
                {"name": o.name, "status": o.status, "seconds": round(o.seconds, 2),
                 "summary": o.summary, "meta": o.meta}
                for o in report.outcomes
            ],
        }, indent=2, default=str))
        return

    print("\n" + report.render())

    finished = [o for o in report.outcomes
                if o.name == "render" and not o.failed and o.meta.get("published")]
    if finished:
        print()
        typer.secho(f"Recap ready:  output\\{Path(finished[0].meta['published']).name}",
                    fg=typer.colors.GREEN)
    print(f"\nartifacts in {cache.dir}")
    print(f"timings appended to {timings.name}")

    failures = [o for o in report.outcomes if o.failed]
    if failures:
        print()
        for outcome in failures:
            typer.secho(
                f"{outcome.name} did not complete: {outcome.error}",
                fg=typer.colors.RED, err=True,
            )
        raise typer.Exit(1)


@app.command("ingest")
def ingest_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 1 only. Extract dialogue from subtitles or speech recognition."""
    settings = Settings()
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    _run_ingest(cache, movie, probe_data, settings, report,
                force_all=force, force_stages=None, quiet=quiet)
    print("\n" + report.render())


@app.command("proxy")
def proxy_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    proxy_height: Optional[int] = HeightOpt,
    threshold: Optional[float] = ThresholdOpt,
    no_qsv: bool = NoQsvOpt,
    with_proxy: bool = WithProxyOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 2 only. Read the film once for the wav and the raw scene data."""
    settings = _settings(proxy_height, threshold, no_qsv, with_proxy)
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(proxy.run(cache, movie, probe_data, settings, force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("scenemap")
def scenemap_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    threshold: Optional[float] = ThresholdOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 3 only. Build the shot map from the scene data stage 2 emitted."""
    settings = _settings(None, threshold, False)
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    meta_file = cache.path("proxy.meta.json")
    if not meta_file.is_file():
        typer.secho("the proxy stage has not run for this file yet", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    report = Report(movie, duration)
    report.add(scenemap.run(cache, read_json(meta_file), settings, force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("story")
def story_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 4 only. Read the dialogue through Claude and build the story."""
    settings = Settings()
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(story.run(cache, settings, force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("script")
def script_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 5 only. Turn the story into a narration script."""
    settings = Settings()
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(script.run(cache, settings, force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("index")
def index_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    refine: bool = typer.Option(
        False, "--refine",
        help="Refine boundaries with PySceneDetect. Much slower, slightly finer shots.",
    ),
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 6 only. Index shots in the regions the script references."""
    settings = Settings()
    if refine:
        settings = dataclasses.replace(settings, refine_shots=True)
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(index.run(cache, movie, settings, force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("narrate")
def narrate_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 7 only. Speak every narration line and measure it."""
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(narrate.run(cache, Settings(), force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("select")
def select_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 8 only. Choose footage for each line and write the edit list."""
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(select.run(cache, Settings(), force=force, quiet=quiet))
    print("\n" + report.render())


@app.command("render")
def render_stage(
    movie: Optional[Path] = MovieArg,
    force: bool = ForceOpt,
    copy_video: bool = typer.Option(
        False, "--copy-video",
        help="Stream copy instead of re-encoding. Faster, but clips snap to keyframes.",
    ),
    with_subtitles: bool = typer.Option(
        False, "--with-subtitles",
        help="Also place the subtitle file beside the video. Players will show it.",
    ),
    no_qsv: bool = NoQsvOpt,
    cache_dir: Optional[Path] = CacheDirOpt,
    quiet: bool = QuietOpt,
):
    """Stage 9 only. Render the final video and the sidecar subtitles."""
    settings = Settings()
    changes = {}
    if copy_video:
        changes["copy_video"] = True
    if no_qsv:
        changes["allow_qsv"] = False
    if with_subtitles:
        changes["publish_subtitles"] = True
    if changes:
        settings = dataclasses.replace(settings, **changes)
    movie, cache, probe_data, duration = _open(movie, cache_dir)
    report = Report(movie, duration)
    report.add(render.run(cache, movie, probe_data, settings, force=force, quiet=quiet))
    print("\n" + report.render())


@app.command()
def info(
    movie: Optional[Path] = MovieArg,
    cache_dir: Optional[Path] = CacheDirOpt,
    json: bool = JsonOpt,
):
    """Probe a file and report its streams. Does no work and writes nothing."""
    movie = _resolve_movie(movie)
    probe_data = ffmpeg.probe(movie)
    settings = Settings()
    duration = ffmpeg.duration_seconds(probe_data)
    subs = probe.classify_subtitles(probe_data, settings)
    audio = probe.select_audio(probe_data, settings)

    if json:
        typer.echo(jsonlib.dumps({
            "file": str(movie),
            "duration_s": duration,
            "container": (probe_data.get("format") or {}).get("format_name"),
            "subtitles": [c.describe() for c in subs],
            "audio_selected": audio.label if audio else None,
        }, indent=2, default=str))
        return

    fmt = (probe_data.get("format") or {}).get("format_name", "unknown")
    print(f"{movie.name}")
    print(f"  container   {fmt}")
    print(f"  runtime     {format_hms(duration)}")
    print(f"  size        {movie.stat().st_size / 1e9:.2f} GB")

    print("  video")
    for stream in ffmpeg.streams(probe_data, "video"):
        fps = f"{stream.fps:.3f}fps" if stream.fps else "unknown fps"
        print(f"    {stream.label}  {stream.width}x{stream.height}  {fps}")

    print("  audio")
    for stream in ffmpeg.streams(probe_data, "audio"):
        mark = " <- selected" if audio and stream.index == audio.index else ""
        print(f"    {stream.label}  {stream.channels}ch{mark}")

    print("  subtitles")
    if not subs:
        print("    none embedded")
    for candidate in subs:
        mark = "usable" if candidate.usable else f"unusable, {candidate.reason}"
        print(f"    {candidate.stream.label}  [{candidate.kind}] {mark}")

    sidecars = ingest.find_sidecars(movie)
    print("  sidecar files")
    print("    " + (", ".join(s.name for s in sidecars) if sidecars else "none"))

    usable = [c for c in subs if c.usable]
    if usable:
        plan = f"embedded {usable[0].stream.label}"
    elif sidecars:
        plan = f"sidecar {sidecars[0].name}"
    else:
        plan = "speech recognition, the model will be downloaded on first use"
    print(f"\n  dialogue source would be: {plan}")


# Where the narrator is set. There is no configuration file to point at, so the
# advice printed after sampling names the source line the user has to edit.
_SETTINGS_FILE = "src\\recap\\config.py"

# The sample line every voice reads, so comparisons are like for like.
SAMPLE_LINE = (
    "Some readers call Jules Verne a novelist. A small society of believers "
    "calls him a reporter. To a Vernian, every impossible island in those "
    "books is a place you can actually sail to."
)


def _kokoro_voices() -> list[str]:
    """English Kokoro voices, or nothing when the model is absent."""
    if not models.kokoro_available():
        return []
    try:
        speaker = narrate.KokoroSpeaker("am_michael", models.kokoro_paths())
    except narrate.NoVoice:
        return []
    # The first two letters encode accent then gender: a is American, b is
    # British, f is female, m is male. Other languages are present but this
    # project narrates in English.
    return [v for v in speaker.voices()
            if len(v) > 2 and v[0] in "ab" and v[1] in "fm" and v[2] == "_"]


@app.command("voices")
def voices(
    sample: bool = typer.Option(
        False, "--sample", help="Write a spoken sample of each voice."
    ),
    everything: bool = typer.Option(
        False, "--all", help="Sample female voices too, not just male."
    ),
):
    """List narrator voices and hear samples.

    Samples all read the same line, which is the only fair way to compare them.
    Every voice lives in the one Kokoro model, so sampling a dozen costs nothing
    beyond the synthesis and switching narrator downloads nothing.

    The narrator itself is set in code, in ``Settings.voice``, because this
    project keeps no configuration file.
    """
    kokoro = _kokoro_voices()
    current = models.voice_name(Settings().voice)

    if not kokoro:
        typer.secho(
            f"the Kokoro model is not in {config.MODELS_DIR}, so there are no "
            "voices to list.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)

    groups = (
        ("American male", "am_"), ("British male", "bm_"),
        ("American female", "af_"), ("British female", "bf_"),
    )
    print("Kokoro voices, all in one model, nothing to download:")
    for label, prefix in groups:
        names = [v for v in kokoro if v.startswith(prefix)]
        if names:
            print(f"  {label}:")
            for v in names:
                print(f"    {v}{'  <- in use' if v == current else ''}")

    if not sample:
        print()
        run = entry_point()
        print(f"Hear them:   {run} voices --sample        (male voices)")
        print(f"             {run} voices --sample --all  (every voice)")
        print(f"Choose one:  set voice in {_SETTINGS_FILE}")
        return

    target = config.OUTPUT_DIR / "voice-samples"
    target.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    words = len(SAMPLE_LINE.split())

    wanted = list(kokoro) if everything else [v for v in kokoro if v[1] == "m"]
    print()
    print(f"Writing {len(wanted)} samples to {target}")
    print("Every voice reads the same line.")
    print()

    try:
        engine = narrate.KokoroSpeaker(
            wanted[0], models.kokoro_paths(), settings.kokoro_speed
        )
    except narrate.NoVoice as exc:
        typer.secho(f"  Kokoro unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    for voice in wanted:
        # One model serves every voice, so only the selection changes.
        engine._voice = voice
        out = target / f"kokoro-{voice}.wav"
        if engine.speak(SAMPLE_LINE, out):
            seconds = narrate.wav_seconds(out)
            print(f"  {voice:<16} {seconds:5.1f}s  {words / (seconds / 60):5.0f} wpm")
        else:
            typer.secho(f"  {voice}: could not speak", fg=typer.colors.YELLOW)

    print()
    print(f"Listen in {target}")
    print(f"Then set voice in {_SETTINGS_FILE} and re-run narrate onward.")


@app.command("fetch-models")
def fetch_models(
    attempts: int = typer.Option(
        6, "--attempts", help="How many times to retry a rate limited host."
    ),
):
    """Download the Kokoro voice model and the CLIP encoders into the project.

    Separate from the stages so that a rate limited host is waited out here,
    deliberately, rather than in the middle of a pipeline run.
    """
    print(f"models directory: {config.MODELS_DIR}")

    print("\nKokoro voice model, about 340 MB")
    if models.kokoro_available():
        voice = True
        print("  ready: model and voices")
    else:
        voice = models.ensure_kokoro(attempts=attempts) is not None
        if voice:
            print("  ready: model and voices")
        else:
            typer.secho(
                "  unavailable. Narration falls back to the Windows system voice.",
                fg=typer.colors.YELLOW,
            )

    print("\nCLIP encoders")
    assets = models.ensure_clip(attempts=attempts)
    if assets is None:
        typer.secho(
            "  unavailable. Shot selection will fall back to time proximity.",
            fg=typer.colors.YELLOW,
        )
    else:
        print("  ready: vision, text and tokenizer")

    ok = voice and assets is not None
    print()
    print("all models present" if ok else "some models are missing, see above")
    raise typer.Exit(0 if ok else 1)


@app.command()
def doctor():
    """Check the environment and confirm nothing is installed outside the project."""
    ok = True
    print(f"project root      {config.ROOT}")
    print(f"python            {sys.version.split()[0]}")

    contained = config.is_contained(sys.prefix)
    print(f"interpreter       {sys.prefix}")
    if not contained:
        ok = False
        print("                  FAIL, not inside the project. Use .venv\\Scripts\\python.exe")
    elif not sys.version.startswith("3.11"):
        print("                  warning, expected Python 3.11")

    print("\ncontained locations")
    for name, target in config.CONTAINED_ENV.items():
        status = "ok" if config.is_contained(target) else "OUTSIDE PROJECT"
        if status != "ok":
            ok = False
        print(f"  {name:<24} {target}  [{status}]")

    print("\nexternal tools")
    for name, resolver in (("ffmpeg", ffmpeg.ffmpeg_bin), ("ffprobe", ffmpeg.ffprobe_bin)):
        try:
            found = resolver()
            where = "project" if config.is_contained(found) else "system"
            print(f"  {name:<24} {found}  [{where}]")
        except ffmpeg.MissingBinary as exc:
            ok = False
            print(f"  {name:<24} MISSING, {exc}")

    found = shutil.which("claude")
    if found:
        where = "project" if config.is_contained(found) else "system"
        print(f"  {'claude':<24} {found}  [{where}]")
    else:
        print(f"  {'claude':<24} MISSING, stages 4 and 5 cannot run without it")
        ok = False

    print("\nmodels")
    for name, present in (("kokoro", models.kokoro_available()),
                          ("clip", models.clip_available())):
        note = "present" if present else "missing, run fetch-models"
        print(f"  {name:<24} {note}")

    print("\nhardware acceleration")
    try:
        if ffmpeg.qsv_available():
            print("  Quick Sync h264_qsv     available, proxy encoding will use it")
        else:
            print("  Quick Sync h264_qsv     unavailable, proxy will encode in software")
    except ffmpeg.MissingBinary:
        print("  Quick Sync h264_qsv     cannot test, ffmpeg is missing")

    print("\npython packages")
    for module, note in (("typer", "required"), ("numpy", "required"),
                         ("faster_whisper", "needed only for films without subtitles")):
        try:
            __import__(module)
            print(f"  {module:<24} present")
        except ImportError:
            if note == "required":
                ok = False
            print(f"  {module:<24} missing, {note}")

    print("\n" + ("all checks passed" if ok else "problems found, see FAIL and MISSING above"))
    raise typer.Exit(0 if ok else 1)


@app.command("cache-list")
def cache_list(cache_dir: Optional[Path] = CacheDirOpt):
    """List cached analyses."""
    root = cache_dir or CACHE_ROOT
    if not root.is_dir():
        print(f"no cache yet at {root}")
        return
    entries = sorted([d for d in root.iterdir() if d.is_dir()])
    if not entries:
        print(f"cache is empty at {root}")
        return
    for entry in entries:
        stages = sorted(p.stem.replace(".meta", "") for p in entry.glob("*.meta.json"))
        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        name = entry.name
        for meta in entry.glob("*.meta.json"):
            try:
                name = read_json(meta).get("source_name", name)
                break
            except Exception:  # noqa: BLE001
                pass
        print(f"{entry.name}  {size / 1e6:8.1f} MB  {', '.join(stages) or 'empty'}  {name}")


@app.command("cache-clear")
def cache_clear(
    movie: Optional[Path] = typer.Argument(None, help="Clear only this movie's cache."),
    cache_dir: Optional[Path] = CacheDirOpt,
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation."),
):
    """Delete cached artifacts. The source movie is never touched."""
    import shutil

    root = cache_dir or CACHE_ROOT
    if movie:
        movie = movie.expanduser()
        if not movie.is_file():
            typer.secho(f"not a file: {movie}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        targets = [root / source_id(movie)[:16]]
    else:
        targets = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []

    targets = [t for t in targets if t.is_dir()]
    if not targets:
        print("nothing to clear")
        return

    total = sum(f.stat().st_size for t in targets for f in t.rglob("*") if f.is_file())
    print(f"about to delete {len(targets)} cache folder(s), {total / 1e6:.1f} MB:")
    for t in targets:
        print(f"  {t}")
    if not yes and not typer.confirm("proceed"):
        print("cancelled")
        return
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    print("cleared")


def main() -> None:
    """Entry point.

    Expected failures, such as a missing binary or an unreadable file, are
    reported as a single line rather than a traceback. Anything unexpected still
    raises in full, because that is a bug worth seeing.
    """
    try:
        app()
    except (ffmpeg.MissingBinary, ffmpeg.FfmpegError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        typer.secho(
            "\ninterrupted, no partial result was committed to the cache",
            fg=typer.colors.YELLOW, err=True,
        )
        raise SystemExit(130) from None
