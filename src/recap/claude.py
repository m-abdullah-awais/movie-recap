"""Headless Claude Code wrapper.

All AI reasoning in this project goes through ``claude -p`` on the user's
existing subscription. No API keys, no local language model, and deliberately no
Ollama.

Every call is cached on disk by the hash of its system prompt, user prompt, and
prompt version. That matters because a stage makes one call per ten minutes of
film, and a single bad response should not cost a re-run of the ones that
already succeeded.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .cache import read_json, write_json

# A fenced block is the most common way a model wraps JSON despite being told
# not to, so it is stripped before parsing rather than treated as a failure.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class ClaudeUnavailable(RuntimeError):
    """The ``claude`` executable is not on PATH."""


class ClaudeFailed(RuntimeError):
    """A call failed, or never produced parseable JSON."""


def claude_bin() -> str:
    found = shutil.which("claude")
    if not found:
        raise ClaudeUnavailable(
            "the 'claude' command was not found on PATH. All AI reasoning in this "
            "project runs through headless Claude Code, so it is required."
        )
    return found


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Tries the whole string first, then strips markdown fences, then falls back to
    the outermost brace pair. Models comply with "return only JSON" almost
    always, and this covers the remainder without a retry.
    """
    candidates = [text, _FENCE_RE.sub("", text)]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ClaudeFailed("response contained no parseable JSON object")


@dataclass
class Call:
    """One prompt to send. ``tag`` names the on-disk cache entry."""

    tag: str
    system: str
    prompt: str


@dataclass
class Reply:
    tag: str
    data: dict
    cost_usd: float
    seconds: float
    cached: bool
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _key(system: str, prompt: str, version: int, model: str | None) -> str:
    blob = json.dumps(
        {"system": system, "prompt": prompt, "version": version, "model": model},
        sort_keys=True,
    )
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def _invoke(system: str, prompt: str, model: str | None, timeout: float) -> dict:
    """Run one ``claude -p`` call and return the parsed envelope.

    The prompt goes in on stdin rather than as an argument. Windows caps a
    command line at about 32,000 characters, and a transcript chunk plus its
    instructions can approach that.

    ``--system-prompt`` replaces the default system prompt outright, which keeps
    the project's own CLAUDE.md and memory files out of an analysis call where
    they would only be noise. Flags are kept identical across calls so every call
    after the first reuses the same prompt cache prefix, which is the difference
    between five cents and twenty-five cents per call.
    """
    args = [
        claude_bin(),
        "-p",
        "--system-prompt", system,
        "--output-format", "json",
    ]
    if model:
        args += ["--model", model]

    proc = subprocess.run(
        args,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        raise ClaudeFailed(f"claude exited {proc.returncode}: {' '.join(tail)}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeFailed(f"claude did not return a JSON envelope: {exc}") from exc


def ask(
    call: Call,
    *,
    cache_dir: Path,
    version: int,
    model: str | None = None,
    timeout: float = 300.0,
    retries: int = 2,
) -> Reply:
    """Send one call, using the on-disk result if this exact prompt was asked before."""
    key = _key(call.system, call.prompt, version, model)
    target = cache_dir / f"{call.tag}.{key}.json"

    if target.is_file():
        try:
            stored = read_json(target)
            return Reply(call.tag, stored["data"], 0.0, 0.0, cached=True)
        except (json.JSONDecodeError, KeyError, OSError):
            target.unlink(missing_ok=True)  # unreadable cache entry, just redo it

    prompt = call.prompt
    last_error = ""
    for attempt in range(retries + 1):
        try:
            envelope = _invoke(call.system, prompt, model, timeout)
            data = extract_json(envelope.get("result") or "")
            cost = float(envelope.get("total_cost_usd") or 0.0)
            seconds = float(envelope.get("duration_ms") or 0.0) / 1000.0
            cache_dir.mkdir(parents=True, exist_ok=True)
            write_json(target, {"key": key, "tag": call.tag, "data": data,
                                "cost_usd": cost, "seconds": seconds})
            return Reply(call.tag, data, cost, seconds, cached=False)
        except (ClaudeFailed, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
            if attempt < retries:
                # Only the JSON discipline is restated. Repeating the whole
                # instruction set tends to make responses longer, not better.
                prompt = (
                    call.prompt
                    + "\n\nYour previous reply could not be parsed. Return ONLY a "
                    "single JSON object, with no prose and no markdown fences."
                )

    return Reply(call.tag, {}, 0.0, 0.0, cached=False, error=last_error)


def ask_many(
    calls: Sequence[Call],
    *,
    cache_dir: Path,
    version: int,
    model: str | None = None,
    timeout: float = 300.0,
    retries: int = 2,
    concurrency: int = 3,
    on_done: Callable[[Reply, int, int], None] | None = None,
) -> list[Reply]:
    """Send several calls, a few at a time, preserving input order.

    Calls run concurrently because each spends around twenty seconds waiting on
    the network, and a feature film needs one per ten minutes of runtime.
    Concurrency is kept low on purpose: the aim is to hide latency, not to push
    the account's rate limit.
    """
    if not calls:
        return []

    results: list[Reply | None] = [None] * len(calls)
    done = 0

    def work(index: int) -> None:
        results[index] = ask(
            calls[index], cache_dir=cache_dir, version=version,
            model=model, timeout=timeout, retries=retries,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, i): i for i in range(len(calls))}
        for future in futures:
            future.result()
            done += 1
            reply = results[futures[future]]
            if on_done and reply is not None:
                on_done(reply, done, len(calls))

    return [r for r in results if r is not None]
