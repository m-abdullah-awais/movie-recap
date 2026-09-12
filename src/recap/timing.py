"""Per-stage timing report.

Measuring real timings on a real film is the whole point of this build phase, so
the numbers are persisted to disk rather than only printed. That lets successive
runs be compared after tuning.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cache import StageOutcome, write_json


def format_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@dataclass
class Report:
    source: Path
    source_duration_s: float
    outcomes: list[StageOutcome] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, outcome: StageOutcome) -> StageOutcome:
        self.outcomes.append(outcome)
        return outcome

    @property
    def total_seconds(self) -> float:
        return sum(o.seconds for o in self.outcomes)

    @property
    def realtime_factor(self) -> float | None:
        total = self.total_seconds
        if total <= 0 or self.source_duration_s <= 0:
            return None
        return self.source_duration_s / total

    def render(self) -> str:
        name_w = max([len("stage")] + [len(o.name) for o in self.outcomes])
        lines = [
            f"{'stage'.ljust(name_w)}  {'status':<8}  {'time':>8}  detail",
            f"{'-' * name_w}  {'-' * 8}  {'-' * 8}  {'-' * 44}",
        ]
        for o in self.outcomes:
            lines.append(
                f"{o.name.ljust(name_w)}  {o.status:<8}  "
                f"{format_hms(o.seconds):>8}  {o.summary}"
            )
        lines.append(f"{'-' * name_w}  {'-' * 8}  {'-' * 8}  {'-' * 44}")

        failed = [o.name for o in self.outcomes if o.failed]
        if failed:
            lines.append(
                f"{'':<{name_w}}  {'':<8}  {'':>8}  did not complete: {', '.join(failed)}"
            )

        detail = f"source runtime {format_hms(self.source_duration_s)}"
        factor = self.realtime_factor
        if factor:
            detail += f", {factor:.1f}x realtime"
        lines.append(f"{'total'.ljust(name_w)}  {'':<8}  {format_hms(self.total_seconds):>8}  {detail}")
        return "\n".join(lines)

    def persist(self, cache_dir: Path) -> Path:
        """Append this run to timings.json so numbers survive across runs."""
        target = cache_dir / "timings.json"
        history: list[dict] = []
        if target.is_file():
            try:
                previous = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(previous, dict):
                    history = previous.get("runs", [])
            except Exception:
                history = []

        history.append(
            {
                "when": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                "source": self.source.name,
                "source_duration_s": round(self.source_duration_s, 2),
                "total_seconds": round(self.total_seconds, 2),
                "realtime_factor": round(self.realtime_factor, 2) if self.realtime_factor else None,
                "stages": [
                    {
                        "name": o.name,
                        "status": o.status,
                        "seconds": round(o.seconds, 2),
                        "summary": o.summary,
                    }
                    for o in self.outcomes
                ],
            }
        )
        write_json(
            target,
            {
                "machine": f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}",
                "runs": history[-50:],
            },
        )
        return target
