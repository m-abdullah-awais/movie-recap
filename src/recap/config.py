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
