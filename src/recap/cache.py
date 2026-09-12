"""Content-hash cache layer.

Every stage reads and writes artifacts keyed by a hash of the source file plus
the stage's own version and parameters. Re-running a stage is therefore skippable
by default, which is what makes iterating on the expensive later stages
affordable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# Size of each region sampled when fingerprinting the source file.
_SAMPLE_BYTES = 8 * 1024 * 1024


def source_id(path: Path) -> str:
    """Fast content fingerprint for a large media file.

    Hashing a multi gigabyte film in full would cost minutes on a mechanical or
    heavily used disk, so this digests the file size together with three 8 MB
    regions taken at the start, middle, and end. That runs in well under a second
    and, unlike modification time, survives renames and copies while still
    distinguishing two different films of identical size.
    """
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"v1:{size}:".encode())

    offsets = [0]
    if size > _SAMPLE_BYTES * 2:
        offsets.append(max(0, size // 2 - _SAMPLE_BYTES // 2))
        offsets.append(max(0, size - _SAMPLE_BYTES))

    with path.open("rb") as handle:
        for offset in sorted(set(offsets)):
            handle.seek(offset)
            digest.update(handle.read(_SAMPLE_BYTES))
    return digest.hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stage_key(sid: str, stage: str, version: int, params: dict[str, Any]) -> str:
    """Cache key for one stage.

    Parameters are part of the key, so lowering the scene threshold invalidates
    scenemap while leaving the expensive proxy untouched.
    """
    blob = _canonical({"source": sid, "stage": stage, "version": version, "params": params})
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Yield a temporary path that is moved into place only on success.

    Without this, an interrupted run could leave a truncated artifact that a
    later run would treat as a valid cache hit.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"nothing was written to {tmp}")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_json(target: Path, payload: Any) -> None:
    with atomic_path(target) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(target: Path) -> Any:
    return json.loads(target.read_text(encoding="utf-8"))


@dataclass
class StageOutcome:
    """Result of one stage, whether it ran, was served from cache, or failed."""

    name: str
    status: str  # "hit", "computed", or "failed"
    seconds: float
    meta: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str = ""

    @property
    def was_cached(self) -> bool:
        return self.status == "hit"

    @property
    def failed(self) -> bool:
        return self.status == "failed"


class Cache:
    """Per-source cache directory."""

    def __init__(self, root: Path, sid: str, source: Path):
        self.sid = sid
        self.source = source
        self.dir = root / sid[:16]
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def _meta_path(self, stage: str) -> Path:
        return self.dir / f"{stage}.meta.json"

    def lookup(
        self,
        stage: str,
        version: int,
        params: dict[str, Any],
        artifacts: Sequence[str],
        may_be_empty: Sequence[str] = (),
    ) -> dict[str, Any] | None:
        """Return the stored metadata on a valid hit, otherwise None.

        A hit requires the metadata to exist, its key to match, and every
        declared artifact to be present. A metadata file whose artifact was
        deleted is a miss rather than an error, so clearing disk space never
        corrupts a later run.

        Artifacts are also required to be non-empty, which catches a zero byte
        file left by a run that was interrupted while overwriting a previously
        committed result. Names in ``may_be_empty`` are exempt, because some
        artifacts have a legitimate empty form. The scene detection file is one:
        a film in which no cut clears the detection floor correctly produces no
        records at all.
        """
        meta_file = self._meta_path(stage)
        if not meta_file.is_file():
            return None
        try:
            meta = read_json(meta_file)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(meta, dict):
            return None
        if meta.get("key") != stage_key(self.sid, stage, version, params):
            return None

        exempt = set(may_be_empty)
        for name in artifacts:
            artifact = self.path(name)
            try:
                if not artifact.is_file():
                    return None
                if name not in exempt and artifact.stat().st_size == 0:
                    return None
            except OSError:
                return None
        return meta

    def commit(
        self,
        stage: str,
        version: int,
        params: dict[str, Any],
        artifacts: Sequence[str],
        extra: dict[str, Any],
        seconds: float,
    ) -> dict[str, Any]:
        meta = {
            "key": stage_key(self.sid, stage, version, params),
            "stage": stage,
            "version": version,
            "params": params,
            "artifacts": list(artifacts),
            "source_id": self.sid,
            "source_name": self.source.name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(seconds, 3),
            **extra,
        }
        write_json(self._meta_path(stage), meta)
        return meta


def run_stage(
    cache: Cache,
    stage: str,
    version: int,
    params: dict[str, Any],
    artifacts: Sequence[str],
    work: Callable[[], dict[str, Any]],
    *,
    force: bool = False,
    summarize: Callable[[dict[str, Any]], str] | None = None,
    may_be_empty: Sequence[str] = (),
) -> StageOutcome:
    """Run ``work`` unless a valid cached result already exists.

    A plain function is used instead of a context manager because a ``with``
    block cannot skip its own body, which is exactly what a cache hit needs to
    do. ``work`` returns the extra metadata to record.
    """
    if not force:
        cached = cache.lookup(stage, version, params, artifacts, may_be_empty)
        if cached is not None:
            summary = summarize(cached) if summarize else ""
            return StageOutcome(stage, "hit", 0.0, cached, summary)

    started = time.perf_counter()
    extra = work() or {}
    elapsed = time.perf_counter() - started
    meta = cache.commit(stage, version, params, artifacts, extra, elapsed)
    summary = summarize(meta) if summarize else ""
    return StageOutcome(stage, "computed", elapsed, meta, summary)
