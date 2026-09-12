"""Project paths, tunables, and environment containment.

Importing this module redirects every cache and install location into the
project directory. It must be imported before ``faster_whisper`` or
``huggingface_hub``, because those read ``HF_HOME`` at import time and would
otherwise cache model weights under the user profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
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
}

CACHE_ROOT = ROOT / "cache"
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


contain_environment()


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

    # Stage 4, story. All AI reasoning goes through headless Claude Code.
    #
    # Ten minute windows keep each call's attention on a manageable stretch of
    # film while still giving enough context to describe a scene. A trailing
    # overlap stops a beat that straddles a boundary from being lost by both
    # neighbours. Concurrency is low on purpose: each call waits around twenty
    # seconds on the network, so a handful in flight hides the latency without
    # pushing the account's rate limit.
    story_chunk_s: int = 600
    story_overlap_s: int = 45
    claude_model: str = ""  # empty means the session default
    claude_concurrency: int = 3
    claude_timeout_s: int = 300

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
    index_window_s: float = 90.0
    index_target_coverage: float = 0.30
    index_min_window_s: float = 5.0
    index_max_shots: int = 900
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
    # A short gap between lines stops the narration sounding rushed and gives the
    # render a natural place to change shot. Length scale is Piper's speaking
    # rate, where above 1.0 is slower.
    narrate_gap_s: float = 0.35
    piper_length_scale: float = 1.0
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

    # Stage 9, render.
    #
    # copy_video is off by default. Stream copying is much faster but can only
    # start a clip on a keyframe, and these clip boundaries come from shot
    # detection and narration timing, so they fall wherever they fall. Copying
    # would shift every clip to an earlier keyframe or emit corrupt leading
    # frames, so the default is a frame accurate re-encode.
    copy_video: bool = False
    render_height: int = 1080
    render_crf: int = 21
    # Ducking the film under the narration. The threshold is deliberately low:
    # the sidechain is the narration itself, so any speech at all should pull the
    # film down.
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
