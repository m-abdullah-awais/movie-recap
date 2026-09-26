"""Model asset acquisition.

Everything downloaded here lands in the project's ``.models`` directory, never in
a user-level cache. Assets are fetched once and reused.

Downloads retry with backoff because Hugging Face rate limits aggressively from
some networks, and a 429 is a "come back later" rather than a real failure. When
an asset genuinely cannot be fetched the caller gets ``None`` and degrades, rather
than the pipeline stopping.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import MODELS_DIR

_USER_AGENT = "movie-recap/0.1 (local, no telemetry)"

# Hugging Face throttles anonymous traffic far harder than authenticated
# traffic, so a token is used when one is available. It is read from the
# environment or from the standard credential file, never stored in this project
# and never printed.
_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN")
_TOKEN_FILES = (
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
)


def hf_token() -> str | None:
    """Hugging Face credential, if the user has one configured."""
    for name in _TOKEN_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for path in _TOKEN_FILES:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None

# CLIP ViT-B/32 exported to ONNX and quantized to int8. PyTorch is deliberately
# not used anywhere in this project: it is two gigabytes for capability the
# pipeline never touches.
CLIP_FILES = {
    "clip_vision_int8.onnx":
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
    "clip_text_int8.onnx":
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx",
    "clip_tokenizer.json":
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
}

# Kokoro is the only speech engine. It carries around fifty voices in one model
# rather than one file per voice, so switching narrator or sampling a dozen of
# them costs nothing once the model is present.
#
# A male narrator, which is what this project wants for recap voiceover.
DEFAULT_VOICE = "am_liam"

_LEGACY_PREFIX = "kokoro:"


def voice_name(name: str) -> str:
    """The bare voice name.

    Voices were once written "kokoro:am_liam" to tell them apart from Piper's,
    which had its own naming. Piper has been removed, so the prefix is stripped
    rather than rejected, and an old name still resolves.
    """
    name = name.strip()
    if name.startswith(_LEGACY_PREFIX):
        name = name[len(_LEGACY_PREFIX):]
    return name or DEFAULT_VOICE


@dataclass(frozen=True)
class KokoroAssets:
    model: Path
    voices: Path


def kokoro_paths() -> KokoroAssets:
    folder = MODELS_DIR / "kokoro"
    return KokoroAssets(
        model=folder / "kokoro-v1.0.onnx",
        voices=folder / "voices-v1.0.bin",
    )


def kokoro_available() -> bool:
    assets = kokoro_paths()
    return assets.model.is_file() and assets.voices.is_file()


# The model and its voice pack come from the kokoro-onnx release on GitHub. That
# host needs no credentials and does not rate limit the way Hugging Face does,
# which matters because this is the one asset the tool cannot narrate without.
_KOKORO_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
)
KOKORO_FILES = {
    "kokoro-v1.0.onnx": _KOKORO_RELEASE + "kokoro-v1.0.onnx",
    "voices-v1.0.bin": _KOKORO_RELEASE + "voices-v1.0.bin",
}


def ensure_kokoro(*, quiet: bool = False, attempts: int = 4) -> KokoroAssets | None:
    """Fetch the Kokoro model, or return None so the caller can degrade.

    About 340 MB in total, so progress is reported. Without it the narrator
    falls back to the robotic Windows system voice, which still produces a
    finished video.
    """
    if kokoro_available():
        return kokoro_paths()

    target_dir = MODELS_DIR / "kokoro"
    try:
        for name, url in KOKORO_FILES.items():
            destination = target_dir / name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            download(
                url, destination, attempts=attempts, timeout=600.0,
                on_progress=None if quiet else _percent(name),
            )
            if not quiet:
                print()
    except DownloadFailed as exc:
        if not quiet:
            print(f"  Kokoro model unavailable: {exc}")
        return None
    return kokoro_paths()


def _percent(name: str) -> Callable[[int, int], None]:
    """Progress printer for a large download.

    Written with a carriage return so a 300 MB file occupies one line rather
    than a thousand, and throttled to whole percents for the same reason.
    """
    state = {"shown": -1}

    def report(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct == state["shown"]:
            return
        state["shown"] = pct
        print(f"\r  fetching {name}  {pct:3d}%  {done / 1048576:6.1f} MB",
              end="", flush=True)

    return report


class DownloadFailed(RuntimeError):
    pass


def _retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait, preferring the server's own advice."""
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return min(120.0, float(header))
        except ValueError:
            pass
    return min(60.0, 4.0 * (2 ** attempt))


def download(
    url: str,
    target: Path,
    *,
    attempts: int = 4,
    timeout: float = 120.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch one file, retrying on rate limits and transient errors.

    Written to a temporary name and moved into place only once complete, so an
    interrupted download is never mistaken for a usable model.
    """
    if target.is_file() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    last = ""

    for attempt in range(attempts):
        try:
            headers = {"User-Agent": _USER_AGENT}
            token = hf_token() if "huggingface.co" in url else None
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with tmp.open("wb") as handle:
                    while True:
                        block = response.read(262144)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        if on_progress:
                            on_progress(done, total)
            tmp.replace(target)
            return target

        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            tmp.unlink(missing_ok=True)
            # 429 and 5xx are worth waiting out. A 404 never becomes a 200.
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                break
            time.sleep(_retry_after(exc, attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
            tmp.unlink(missing_ok=True)
            if attempt == attempts - 1:
                break
            time.sleep(min(30.0, 4.0 * (2 ** attempt)))

    raise DownloadFailed(f"could not fetch {url.rsplit('/', 1)[-1]}: {last}")


@dataclass(frozen=True)
class ClipAssets:
    vision: Path
    text: Path
    tokenizer: Path


def clip_paths() -> ClipAssets:
    return ClipAssets(
        vision=MODELS_DIR / "clip" / "clip_vision_int8.onnx",
        text=MODELS_DIR / "clip" / "clip_text_int8.onnx",
        tokenizer=MODELS_DIR / "clip" / "clip_tokenizer.json",
    )


def clip_available() -> bool:
    paths = clip_paths()
    return all(p.is_file() and p.stat().st_size > 0
               for p in (paths.vision, paths.text, paths.tokenizer))


def ensure_clip(*, quiet: bool = False, attempts: int = 4) -> ClipAssets | None:
    """Fetch the CLIP encoders, or return None so the caller can degrade."""
    if clip_available():
        return clip_paths()

    target_dir = MODELS_DIR / "clip"
    try:
        for name, url in CLIP_FILES.items():
            destination = target_dir / name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            if not quiet:
                print(f"  fetching {name}")
            download(url, destination, attempts=attempts)
    except DownloadFailed as exc:
        if not quiet:
            print(f"  CLIP model unavailable: {exc}")
        return None
    return clip_paths()
