"""Project paths, tunables, and environment containment.

Importing this module redirects every cache and install location into the
project directory. It must be imported before ``faster_whisper`` or
``huggingface_hub``, because those read ``HF_HOME`` at import time and would
otherwise cache model weights under the user profile.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# src/recap/config.py -> src/recap -> src -> project root
ROOT = Path(__file__).resolve().parents[2]

# Locations that several tools default to the user profile. Each is redirected
# into the project so that nothing is ever installed or cached outside it.
CONTAINED_ENV: dict[str, Path] = {
    "UV_PYTHON_INSTALL_DIR": ROOT / ".python",
    "UV_PROJECT_ENVIRONMENT": ROOT / ".venv",
    "UV_CACHE_DIR": ROOT / ".uv-cache",
    "PIP_CACHE_DIR": ROOT / ".uv-cache" / "pip",
    "XDG_CACHE_HOME": ROOT / ".uv-cache",
    "HF_HOME": ROOT / ".models",
    # npm is only used by setup, to install the Claude Code CLI into the
    # project, but its cache defaults to the user profile and is worth pinning
    # for the same reason as the rest.
    "npm_config_cache": ROOT / ".uv-cache" / "npm",
}

# External programs that Setup.bat installed into the project because the
# machine did not already have them. Each entry is a directory holding
# executables, listed in the order they should be searched.
TOOLS_DIR = ROOT / ".tools"
_CLAUDE_PKG = TOOLS_DIR / "claude" / "node_modules" / "@anthropic-ai" / "claude-code"
TOOL_BINS: tuple[Path, ...] = (
    TOOLS_DIR / "ffmpeg" / "bin",
    TOOLS_DIR / "node",
    TOOLS_DIR / "agy",
    # Claude Code's own executable comes first, ahead of the .cmd shim npm
    # writes next to it. A .cmd means cmd.exe re-parses every argument, and the
    # system prompt is passed as one: a percent sign or an ampersand in it would
    # be expanded or split, which would be invisible here and would only show up
    # on a machine that used the local install.
    _CLAUDE_PKG / "bin",
    TOOLS_DIR / "claude" / "node_modules" / ".bin",
    TOOLS_DIR / "uv",
)

# Which AI engine setup installed, one word, written by Setup.bat. This is an
# install record rather than a setting: nobody edits it, and it is deleted along
# with the tools it describes. It exists for the case where a machine has both
# coding agents installed already, where nothing else would say which one was
# chosen.
ENGINE_FILE = TOOLS_DIR / "engine.txt"

CACHE_ROOT = ROOT / "cache"
# Finished videos are published here under a timestamped name, so a re-run
# never overwrites a render you wanted to keep.
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / ".models"
VENV_DIR = ROOT / ".venv"

# Drop-in folder. A movie placed here is found without a path being given.
INPUT_DIR = ROOT / "input"

# Containers worth offering as a drop-in. This list only decides what counts as
# a candidate in the input folder. An explicitly named file is never filtered by
# extension, because ffmpeg detects the format from content.
MEDIA_EXTS = frozenset(
    {
        ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm", ".wmv", ".flv",
        ".mpg", ".mpeg", ".m2ts", ".ts", ".vob", ".ogv", ".3gp", ".divx",
    }
)


def discover_input(input_dir: Path | None = None) -> list[Path]:
    """Movies sitting in the drop-in folder, largest first.

    Sorted by size because a stray sample or trailer alongside the feature is
    always the smaller file, which makes the ordering useful when reporting
    several candidates.
    """
    folder = input_dir or INPUT_DIR
    if not folder.is_dir():
        return []
    found = [
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in MEDIA_EXTS
    ]
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def contain_environment() -> None:
    """Force every redirected location into the project directory.

    Existing values are overwritten rather than respected, because a stray
    ``HF_HOME`` pointing at the user profile would silently defeat the rule that
    nothing lives outside this directory.
    """
    for name, target in CONTAINED_ENV.items():
        os.environ[name] = str(target)


def prepend_local_tools() -> list[Path]:
    """Put the project's own copies of ffmpeg, node and claude first on PATH.

    Setup installs an external program into ``.tools`` only when the machine
    does not already have it, so in practice there is nothing to shadow. When
    there is, the project's copy wins: it is the one that was tested against
    this code, and it cannot disappear when the user tidies their PATH.

    Returning the directories that were added lets ``doctor`` report them.
    """
    present = [path for path in TOOL_BINS if path.is_dir()]
    if not present:
        return []
    prefix = os.pathsep.join(str(path) for path in present)
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{prefix}{os.pathsep}{current}" if current else prefix
    return present


contain_environment()
prepend_local_tools()


def is_contained(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` resolves inside the project directory."""
    try:
        Path(path).resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return False
    return True


@dataclass(frozen=True)
class Settings:
    """Tunables. Every value here feeds the cache key of the stage that uses it,
    so changing one invalidates only the affected stage."""

    # Stage 2, the single full-film read.
    #
    # By default no full-film video proxy is written. Measured on a 94 minute
    # HEVC 10 bit film on this hardware, decoding costs about 10 minutes and is
    # unavoidable, while encoding a 480p proxy on top added a further 6. Nothing
    # after stage 3 needs a full-film proxy: stage 6 indexes only the narrowed
    # regions the script actually references, and stage 9 stream-copies the
    # original. Enable it for debugging, or to eyeball what the pipeline saw.
    write_proxy_video: bool = False
    proxy_height: int = 480
    proxy_fps: int = 24
    # An fps filter is only inserted above this rate, so 23.976 and 24 fps films
    # are passed through untouched and only high frame rate sources are reduced.
    proxy_fps_threshold: float = 25.0
    allow_qsv: bool = True
    x264_preset: str = "veryfast"
    x264_crf: int = 28
    qsv_quality: int = 28
    audio_rate: int = 16000

    # Detection floor applied during the stage 2 pass. Deliberately permissive so
    # that a superset of candidate cuts is recorded once, together with each
    # cut's score. Stage 3 then filters by score, which means retuning the
    # threshold never requires re-encoding the proxy.
    scene_floor: float = 3.0

    # Stage 3, scenemap
    detect_width: int = 160
    scene_threshold: float = 8.0
    min_shot_s: float = 0.6
    max_shot_s: float = 20.0
    tail_fraction: float = 0.06
    fallback_shot_s: float = 4.0

    # Stage 4, story. All AI reasoning goes through a headless coding agent,
    # either Claude Code or the Antigravity CLI, whichever setup installed.
    #
    # Ten minute windows keep each call's attention on a manageable stretch of
    # film while still giving enough context to describe a scene. A trailing
    # overlap stops a beat that straddles a boundary from being lost by both
    # neighbours. Concurrency is low on purpose: each call waits around twenty
    # seconds on the network, so a handful in flight hides the latency without
    # pushing the account's rate limit.
    story_chunk_s: int = 600
    story_overlap_s: int = 45
    # Empty means whatever the engine would choose for itself, which is the
    # right default for both of them and keeps the flags identical between
    # calls. A name here is passed straight through to --model.
    ai_model: str = ""
    ai_concurrency: int = 3
    ai_timeout_s: int = 300

    # Stage 5, script.
    #
    # A 10 to 20 minute recap at a normal narration pace is roughly 1500 to 3000
    # words. The target sits mid-range. Speech rate is only used to predict
    # length here; stage 7 replaces the estimate with the real measured duration
    # of each rendered line.
    script_target_words: int = 2200
    script_words_per_minute: int = 150
    # How far ahead of the narration's current position footage may be drawn
    # from. Without a cap, a segment early in the recap could be backed by a
    # shot from the finale.
    spoiler_lookahead_s: float = 180.0

    # Stage 6, index.
    #
    # Narrow first. Only the stretches of film the narration actually references
    # are indexed, which for a feature is roughly 25 minutes rather than 120.
    # Indexing the whole film is the single easiest way to blow the budget.
    # The window is a starting point, not a fixed figure. It is halved until the
    # regions cover no more than the target fraction of the runtime, because a
    # fixed window narrows nothing when the narration is dense: measured on a 94
    # minute film with 77 segments a median 62 seconds apart, a 90 second window
    # merged into four regions covering 99 percent of the film.
    # Studio logos, opening titles and end credits are never usable footage.
    # The dialogue itself says where they are: on a 94 minute film the first
    # spoken line landed at 34 seconds and the last at 5143 of 5650, leaving 8.5
    # minutes of end credits with no speech at all. A lead in and lead out keep
    # the establishing shot before the first line and the closing beat after the
    # last one.
    credits_lead_in_s: float = 10.0
    credits_lead_out_s: float = 20.0

    # Index every shot in the film rather than only those near a narration
    # anchor. Narrowing was a speed optimisation and it capped retrieval quality
    # badly: shrinking the window to hit a coverage target left roughly three
    # candidate shots per line, so the timestamp effectively chose the footage
    # and CLIP only broke ties between near-duplicates. Worse, the window is
    # centred on an anchor that Claude inferred from subtitle timings, so an
    # anchor off by half a minute put every candidate in the wrong scene with no
    # way to recover. Searching the whole film lets a good visual match win from
    # anywhere, with time reduced to a preference in the scoring rather than a
    # gate. It costs a few more minutes of keyframe extraction and embedding.
    index_whole_film: bool = True

    # How a frame is fitted to CLIP's square input. Cropping is what CLIP's own
    # preprocessing does, but on a 1920x1080 frame it keeps only the middle 224
    # of 398 pixels and throws away 44 percent of the width, which in widescreen
    # film often contains the subject. Padding keeps the whole frame at the cost
    # of some detail and black bars.
    keyframe_fit: str = "pad"

    index_window_s: float = 90.0
    index_target_coverage: float = 0.30
    index_min_window_s: float = 5.0
    # Raised because the whole film is indexed now. Embedding is cheap, about a
    # tenth of a second per shot, so the cap exists only to stop something
    # pathological rather than to save meaningful time.
    index_max_shots: int = 3000
    # Refinement with PySceneDetect inside the narrowed regions is off by
    # default. Measured on the 94 minute film it cost 6 minutes 54 seconds and
    # found 24 extra shots out of 243, because the regions are only a few seconds
    # long and the stage 3 boundaries already average about three seconds. Nearly
    # seven minutes of a twenty five minute budget is too much for a ten percent
    # change in shot count. Enable it with --refine when accuracy matters more.
    refine_shots: bool = False
    refine_threshold: float = 27.0
    keyframe_workers: int = 4
    clip_batch: int = 8

    # Stage 7, narrate.
    #
    # The narrator, chosen from spoken samples. Every Kokoro voice lives in the
    # one model, so the name is all that changes: run the voices command to hear
    # the alternatives, then edit this line. It is set here rather than in a
    # configuration file because this project deliberately has none.
    voice: str = "am_liam"
    # Larger is faster. Measured on am_michael: 0.9 gives 143 words per minute,
    # 1.0 gives 154, and 1.3 gives 190, where 150 to 170 reads well for
    # something a viewer listens to for a quarter of an hour.
    kokoro_speed: float = 1.0
    # A short gap between lines stops the narration sounding rushed and gives
    # the render a natural place to change shot.
    narrate_gap_s: float = 0.35
    narrate_workers: int = 2

    # Stage 8, select.
    #
    # Weights are deliberately exposed. What looks right varies by film, and the
    # stage is cheap to re-run because it touches no video. Similarity is
    # rescaled per line before weighting, since CLIP cosine scores occupy a
    # narrow positive band and would otherwise be swamped by proximity.
    clip_weight: float = 1.0
    proximity_weight: float = 0.6
    band_weight: float = 0.25
    reuse_penalty: float = 0.35
    dark_penalty: float = 0.5
    dark_luma: float = 18.0
    max_shot_uses: int = 3
    min_clip_s: float = 1.5
    max_clip_s: float = 4.0
    proximity_sigma_s: float = 60.0
    # How far from the narrated moment footage may be taken. Proximity alone is
    # only a score, and a spuriously confident CLIP match will beat a mediocre
    # nearby shot from anywhere in the film. Measured with no limit at all, 8.7
    # percent of clips came from more than five minutes away and one from 58
    # minutes away, which reads as footage unrelated to the narration. Three
    # minutes is wide enough to survive an anchor that Claude placed wrongly,
    # and tight enough to keep the recap in the scene being described.
    max_shot_distance_s: float = 90.0

    # One unbroken stretch of film per narration line, playing through the
    # natural cuts, rather than several short clips stitched from different
    # places. Stitching was the source of two visible faults: clips within a
    # line ran out of order, only 10 of 86 lines were in ascending film time,
    # and the later clips existed to pad out the remaining duration, their
    # relevance falling from 0.785 to 0.690. A continuous run has no ordering to
    # get wrong and nothing to pad.
    # How a line's footage relates to the stretch of film it narrates.
    #
    # "traverse" walks through that stretch, taking a few ordered clips spread
    # across it. This is the only mode that keeps the picture level with the
    # words. Measured on the test film, a line describes a median 128 seconds of
    # plot while its footage showed 10, just 8 percent of the span, so the words
    # ran through the whole beat while the picture stayed at its opening. That is
    # why the narration kept arriving before the scene it described.
    #
    # "continuous" plays one unbroken run from the start of the beat. Coherent
    # but it falls steadily behind, because film plays at real speed while
    # narration compresses about 13 seconds of plot into every second of speech.
    #
    # "montage" is the original behaviour, kept only for comparison.
    footage_mode: str = "traverse"
    clip_target_s: float = 3.2
    # An upper bound on the film a single line may travel across, so a sparse
    # patch of beats does not make one line leap through half the picture.
    max_span_s: float = 240.0
    continuous_runs: bool = True
    max_run_s: float = 20.0

    # Below this rescaled similarity, CLIP has not found anything convincing and
    # its opinion should not outrank simply staying near the narrated moment.
    similarity_floor: float = 0.15

    # Stage 9, render.
    #
    # copy_video is off by default. Stream copying is much faster but can only
    # start a clip on a keyframe, and these clip boundaries come from shot
    # detection and narration timing, so they fall wherever they fall. Copying
    # would shift every clip to an earlier keyframe or emit corrupt leading
    # frames, so the default is a frame accurate re-encode.
    # No subtitles are ever drawn onto the picture. This controls only whether
    # the sidecar file is placed next to the published video, and it is off
    # because players auto-load a subtitle file that shares the video's name and
    # show it without being asked. The file is still written into the cache, so
    # it is there if you want it for an upload.
    publish_subtitles: bool = False
    copy_video: bool = False
    render_height: int = 1080
    render_crf: int = 21
    # Ducking the film under the narration. The threshold is deliberately low:
    # the sidechain is the narration itself, so any speech at all should pull the
    # film down.
    # The film's own audio is dropped rather than ducked. Ducking leaves the
    # original dialogue and music audible underneath, which competes with the
    # narrator instead of supporting him. Set this false to bring the film's
    # audio back as a ducked bed, controlled by the settings below.
    mute_source_audio: bool = True
    # Loudness normalisation to a broadcast style target, in LUFS. Narration on
    # its own measured a mean of -25 dB with peaks at -3.2, so there was no
    # headroom left to simply turn it up: raising the gain would clip before it
    # reached a comfortable level. Normalising to -16 LUFS is the usual target
    # for spoken word on video platforms and reaches it by evening out the
    # dynamics rather than by amplifying. Set to 0 to leave the level alone.
    loudness_lufs: float = -16.0
    duck_threshold: float = 0.03
    duck_ratio: float = 8.0
    narration_gain: float = 1.0
    source_gain: float = 0.8

    # Stage 1, ingest
    preferred_langs: tuple[str, ...] = ("eng", "en", "english")
    asr_model: str = "small.en"
    asr_compute: str = "int8"
    min_coverage_ratio: float = 0.10

    def params(self, *names: str) -> dict[str, object]:
        """Subset of settings for a stage cache key, keeping unrelated tunables
        out so they do not cause spurious invalidation."""
        data = asdict(self)
        return {n: data[n] for n in names}


# Subtitle codecs that can yield text. Bitmap formats are deliberately excluded:
# extracting them produces images, not dialogue, so they must fall through to
# speech recognition instead of yielding a garbage transcript.
TEXT_SUB_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "subviewer", "microdvd"}
)
BITMAP_SUB_CODECS = frozenset(
    {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "dvb_teletext"}
)
SIDECAR_SUB_EXTS = (".srt", ".ass", ".ssa", ".vtt")
