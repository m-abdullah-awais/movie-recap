"""Headless AI engine wrapper.

All AI reasoning in this project goes through a coding agent's own command line
tool, running on the user's existing subscription. Two are supported, Claude Code
and the Antigravity CLI, and setup installs whichever one was chosen. No API
keys, no local language model, and deliberately no Ollama.

Only three things differ between the two, so only those live in an engine class:
how the process is invoked, whether the prompt goes on stdin or in an argument,
and what the reply envelope is called. Everything else, the caching, the retries
and the JSON salvage, is shared.

Every call is cached on disk by the hash of its engine, system prompt, user
prompt, and prompt version. That matters because a stage makes one call per ten
minutes of film, and a single bad response should not cost a re-run of the ones
that already succeeded. Keying on the engine as well means the two keep separate
answers, so switching back to one you have used before costs nothing.
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

from . import config
from .cache import read_json, write_json

# A fenced block is the most common way a model wraps JSON despite being told
# not to, so it is stripped before parsing rather than treated as a failure.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class EngineUnavailable(RuntimeError):
    """No AI engine could be found to run the call."""


class EngineFailed(RuntimeError):
    """A call failed, or never produced parseable JSON."""


# Windows builds one command line string from the whole argument list and caps
# it a little under 32,768 characters. Antigravity takes its prompt as an
# argument, so a long transcript window has to be refused with a name rather
# than handed over and silently cut in half by the operating system.
MAX_COMMAND_LINE = 30000


class Engine:
    """How one command line tool is driven.

    Subclasses provide the three things that differ. Everything else in this
    module is shared between them.
    """

    name = ""
    program = ""
    description = ""

    def binary(self) -> str:
        found = shutil.which(self.program)
        if not found:
            raise EngineUnavailable(
                f"the '{self.program}' command was not found. All AI reasoning in "
                f"this project runs through {self.description}, so stages 4 and 5 "
                "need it. Run Setup.bat to install it into this folder."
            )
        return found

    def available(self) -> bool:
        return shutil.which(self.program) is not None

    def command(
        self, system: str, prompt: str, model: str | None
    ) -> tuple[list[str], str | None]:
        """The argv to run, and the text to write to its stdin, if any."""
        raise NotImplementedError

    def read(self, envelope: dict) -> tuple[str, float, float]:
        """The reply text, the cost in dollars, and the seconds it took."""
        raise NotImplementedError


class ClaudeEngine(Engine):
    name = "claude"
    program = "claude"
    description = "headless Claude Code"

    def command(
        self, system: str, prompt: str, model: str | None
    ) -> tuple[list[str], str | None]:
        """
        The prompt goes in on stdin rather than as an argument, because Windows
        caps a command line at about 32,000 characters and a transcript chunk
        plus its instructions can approach that.

        ``--system-prompt`` replaces the default system prompt outright, which
        keeps the project's own CLAUDE.md and memory files out of an analysis
        call where they would only be noise. Flags are kept identical across
        calls so every call after the first reuses the same prompt cache prefix,
        which is the difference between five cents and twenty-five cents a call.
        """
        args = [
            self.binary(),
            "-p",
            "--system-prompt", system,
            "--output-format", "json",
        ]
        if model:
            args += ["--model", model]
        return args, prompt

    def read(self, envelope: dict) -> tuple[str, float, float]:
        return (
            envelope.get("result") or "",
            float(envelope.get("total_cost_usd") or 0.0),
            float(envelope.get("duration_ms") or 0.0) / 1000.0,
        )


class AntigravityEngine(Engine):
    name = "antigravity"
    program = "agy"
    description = "the headless Antigravity CLI"

    def command(
        self, system: str, prompt: str, model: str | None
    ) -> tuple[list[str], str | None]:
        """
        This CLI has no separate system prompt, and takes its prompt as an
        argument rather than on stdin, so the two are joined and passed as one.
        The length is checked here because the operating system would otherwise
        truncate it without saying so.
        """
        joined = f"{system}\n\n{prompt}" if system else prompt
        if len(joined) > MAX_COMMAND_LINE:
            raise EngineFailed(
                f"the prompt is {len(joined)} characters, and the Antigravity CLI "
                f"takes it as a command line argument, which Windows caps near "
                f"{MAX_COMMAND_LINE}. Lower story_chunk_s, or use --engine claude, "
                "which sends the prompt on stdin instead."
            )
        args = [self.binary(), "-p", joined, "--output-format", "json"]
        if model:
            args += ["--model", model]
        return args, None

    def read(self, envelope: dict) -> tuple[str, float, float]:
        status = str(envelope.get("status") or "").upper()
        if status and status != "SUCCESS":
            raise EngineFailed(
                f"the Antigravity CLI reported {status}: "
                f"{envelope.get('error') or 'no reason given'}"
            )
        # This CLI reports tokens rather than money, so there is no cost to
        # report and the stage prints a blank instead of a wrong number.
        return (
            envelope.get("response") or "",
            0.0,
            float(envelope.get("duration_seconds") or 0.0),
        )


ENGINES: dict[str, Engine] = {
    "claude": ClaudeEngine(),
    "antigravity": AntigravityEngine(),
}


def select_engine(name: str | None = None) -> Engine:
    """The engine to use, in order of how deliberate the choice was.

    An explicit name wins. Otherwise the record setup wrote when it installed
    one is used, then whatever happens to be present. Falling back to whatever
    is present matters on a machine that already had a coding agent installed
    before this project arrived.
    """
    if name:
        wanted = ENGINES.get(name.strip().lower())
        if wanted is None:
            raise EngineUnavailable(
                f"unknown engine {name}. Choose one of: {', '.join(sorted(ENGINES))}"
            )
        return wanted

    recorded = installed_engine()
    if recorded and recorded in ENGINES:
        return ENGINES[recorded]

    for engine in ENGINES.values():
        if engine.available():
            return engine

    raise EngineUnavailable(
        "no AI engine was found. Stages 4 and 5 need either Claude Code or the "
        "Antigravity CLI. Run Setup.bat and choose one."
    )


def installed_engine() -> str:
    """The engine setup installed, or an empty string if setup has not run.

    A one word record rather than a configuration file: it is written by setup,
    read by nothing else, and deleted along with the tools it describes.
    """
    try:
        return config.ENGINE_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""


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
    raise EngineFailed("response contained no parseable JSON object")


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


def _key(
    engine: str, system: str, prompt: str, version: int, model: str | None
) -> str:
    blob = json.dumps(
        {"engine": engine, "system": system, "prompt": prompt,
         "version": version, "model": model},
        sort_keys=True,
    )
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def _invoke(
    engine: Engine, system: str, prompt: str, model: str | None, timeout: float
) -> dict:
    """Run one call and return the parsed envelope."""
    args, stdin_text = engine.command(system, prompt, model)

    proc = subprocess.run(
        args,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        raise EngineFailed(
            f"{engine.program} exited {proc.returncode}: {' '.join(tail)}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise EngineFailed(
            f"{engine.program} did not return a JSON envelope: {exc}"
        ) from exc


def ask(
    call: Call,
    *,
    cache_dir: Path,
    version: int,
    engine: Engine | None = None,
    model: str | None = None,
    timeout: float = 300.0,
    retries: int = 2,
) -> Reply:
    """Send one call, using the on-disk result if this exact prompt was asked before."""
    engine = engine or select_engine()
    key = _key(engine.name, call.system, call.prompt, version, model)
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
            envelope = _invoke(engine, call.system, prompt, model, timeout)
            text, cost, seconds = engine.read(envelope)
            data = extract_json(text)
            cache_dir.mkdir(parents=True, exist_ok=True)
            write_json(target, {"key": key, "tag": call.tag, "data": data,
                                "engine": engine.name,
                                "cost_usd": cost, "seconds": seconds})
            return Reply(call.tag, data, cost, seconds, cached=False)
        except (EngineFailed, subprocess.TimeoutExpired) as exc:
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
    engine: Engine | None = None,
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

    # Resolved once rather than per call, so a missing engine is reported before
    # any work starts and every call in the batch uses the same one.
    engine = engine or select_engine()
    results: list[Reply | None] = [None] * len(calls)
    done = 0

    def work(index: int) -> None:
        results[index] = ask(
            calls[index], cache_dir=cache_dir, version=version,
            engine=engine, model=model, timeout=timeout, retries=retries,
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
