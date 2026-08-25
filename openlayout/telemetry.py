"""Telemetry — records token usage, model runtime and tool runtime per stage,
plus worker/debugger/optimizer activity counts.  Written to
``<output>/telemetry.json`` continuously and summarized in the final report.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path


def _zero() -> dict:
    return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "model_time_s": 0.0, "tool_runs": 0, "tool_time_s": 0.0,
            "errors": []}


class Telemetry:
    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.t_start = time.time()
        self.stages: dict[str, dict] = defaultdict(_zero)
        self.agent_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    def add_llm_call(self, *, stage: str, agent: str, prompt_tokens: int,
                     completion_tokens: int, seconds: float,
                     error: str | None = None) -> None:
        s = self.stages[stage or "flow"]
        s["llm_calls"] += 1
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += completion_tokens
        s["model_time_s"] += seconds
        if error:
            s["errors"].append(f"{agent}: {error[:300]}")
        self.agent_counts[f"{stage}:{agent}"] += 1
        self.save()

    def add_tool_run(self, *, stage: str, seconds: float,
                     label: str = "") -> None:
        s = self.stages[stage or "flow"]
        s["tool_runs"] += 1
        s["tool_time_s"] += seconds
        self.save()

    def merge_external(self, name: str, data: dict) -> None:
        """Merge a sub-system's telemetry (e.g. macroflow) under one stage."""
        s = self.stages[name]
        s["external"] = data
        # best-effort token totals from macroflow telemetry.json layouts
        for k in ("prompt_tokens", "total_prompt_tokens", "input_tokens"):
            if isinstance(data.get(k), (int, float)):
                s["prompt_tokens"] += int(data[k])
                break
        for k in ("completion_tokens", "total_completion_tokens",
                  "output_tokens"):
            if isinstance(data.get(k), (int, float)):
                s["completion_tokens"] += int(data[k])
                break
        self.save()

    # ------------------------------------------------------------------
    def totals(self) -> dict:
        tot = _zero()
        tot.pop("errors")
        for s in self.stages.values():
            for k in tot:
                tot[k] += s.get(k, 0)
        tot["wall_time_s"] = time.time() - self.t_start
        return tot

    def save(self) -> None:
        try:
            payload = {"stages": dict(self.stages),
                       "agent_counts": dict(self.agent_counts),
                       "totals": self.totals()}
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(json.dumps(payload, indent=2))
        except OSError:
            pass
