"""Model asset acquisition.

Everything downloaded here lands in the project's ``.models`` directory, never in
a user-level cache. Assets are fetched once and reused.

Downloads retry with backoff because Hugging Face rate limits aggressively from
some networks, and a 429 is a "come back later" rather than a real failure. When
an asset genuinely cannot be fetched the caller gets ``None`` and degrades, rather
than the pipeline stopping.
"""

from __future__ import annotations

import shutil
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import MODELS_DIR

_USER_AGENT = "movie-recap/0.1 (local, no telemetry)"

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

# The Piper voice comes from a GitHub release rather than Hugging Face. It is the
# same asset, and GitHub stays reachable on networks where Hugging Face rate
# limits, which was the case on this machine.
PIPER_VOICE_NAME = "en_US-lessac-medium"
PIPER_VOICE_ARCHIVE = (
    "https://github.com/rhasspy/piper/releases/download/v0.0.2/"
    "voice-en-us-lessac-medium.tar.gz"
)


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
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
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


@dataclass(frozen=True)
class PiperVoice:
    model: Path
    config: Path
    name: str


def piper_paths() -> PiperVoice:
    voice_dir = MODELS_DIR / "piper"
    return PiperVoice(
        model=voice_dir / f"{PIPER_VOICE_NAME}.onnx",
        config=voice_dir / f"{PIPER_VOICE_NAME}.onnx.json",
        name=PIPER_VOICE_NAME,
    )


def piper_available() -> bool:
    voice = piper_paths()
    return voice.model.is_file() and voice.config.is_file()


def ensure_piper_voice(*, quiet: bool = False, attempts: int = 4) -> PiperVoice | None:
    """Fetch and unpack the Piper voice, or return None so the caller can degrade."""
    voice = piper_paths()
    if piper_available():
        return voice

    voice_dir = voice.model.parent
    archive = voice_dir / "voice.tar.gz"
    try:
        if not archive.is_file():
            if not quiet:
                print(f"  fetching the {PIPER_VOICE_NAME} voice, about 58 MB")
            download(PIPER_VOICE_ARCHIVE, archive, attempts=attempts, timeout=300.0)
    except DownloadFailed as exc:
        if not quiet:
            print(f"  Piper voice unavailable: {exc}")
        return None

    # The archive nests the two files under a directory, so they are pulled out
    # by suffix rather than by an assumed path.
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if member.name.endswith(".onnx"):
                    _extract_to(tar, member, voice.model)
                elif member.name.endswith(".onnx.json"):
                    _extract_to(tar, member, voice.config)
    except (tarfile.TarError, OSError) as exc:
        if not quiet:
            print(f"  the voice archive could not be unpacked: {exc}")
        return None

    if not piper_available():
        return None
    archive.unlink(missing_ok=True)
    return voice


def _extract_to(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:
        return
    with source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
