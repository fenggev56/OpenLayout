"""Telemetry — token usage, time overhead, iteration counts, runtime.

A single :class:`Telemetry` is carried on the :class:`~macroflow.context.FlowContext`
and threaded through the run:

  * **tokens** — the VLM reports prompt/completion tokens per `chat()`; the Agent
    attributes them to its role (Planner/Placer/Checker), so we get per-agent and
    total token usage.
  * **LLM time vs system time** — every `chat()` is timed; per-stage wall time is
    timed by the orchestrator. LLM seconds = model overhead; (stage − LLM) = the
    OpenROAD/geometry system time.
  * **iterations** — compaction rounds, Gate-B rounds, agent turns, LLM calls.
  * **runtime** — total wall clock + per-stage breakdown.

Written to ``model_output/telemetry.json`` and summarized in ``summary.txt``.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.seconds += float(seconds or 0.0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class OpStat:
    count: int = 0
    seconds: float = 0.0


class Telemetry:
    def __init__(self, clock=time.time):
        self._clock = clock
        self.t0 = clock()
        self.agents: dict[str, LLMUsage] = {}
        self.stages: dict[str, float] = {}     # stage -> wall seconds
        self.ops: dict[str, OpStat] = {}       # fine-grained op -> count/seconds
        self.counters: dict[str, int] = {}     # iterations / rounds / calls

    # ------------------------------------------------------------------
    def record_llm(self, agent: str, prompt_tokens: int, completion_tokens: int,
                   seconds: float) -> None:
        self.agents.setdefault(agent, LLMUsage()).add(
            prompt_tokens, completion_tokens, seconds)

    @contextmanager
    def stage(self, name: str):
        t = self._clock()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (self._clock() - t)

    @contextmanager
    def op(self, name: str):
        t = self._clock()
        try:
            yield
        finally:
            s = self.ops.setdefault(name, OpStat())
            s.count += 1
            s.seconds += (self._clock() - t)

    def incr(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def set(self, key: str, value: int) -> None:
        self.counters[key] = value

    # ------------------------------------------------------------------
    def wall_seconds(self) -> float:
        return self._clock() - self.t0

    def llm_calls(self) -> int:
        return sum(u.calls for u in self.agents.values())

    def llm_seconds(self) -> float:
        return sum(u.seconds for u in self.agents.values())

    def prompt_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.agents.values())

    def completion_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.agents.values())

    def total_tokens(self) -> int:
        return self.prompt_tokens() + self.completion_tokens()

    # ------------------------------------------------------------------
    def report(self) -> dict:
        wall = self.wall_seconds()
        llm_s = self.llm_seconds()
        return {
            "runtime_s": round(wall, 1),
            "llm_time_s": round(llm_s, 1),                 # model overhead
            "system_time_s": round(max(0.0, wall - llm_s), 1),  # OpenROAD + geometry
            "tokens": {
                "prompt": self.prompt_tokens(),
                "completion": self.completion_tokens(),
                "total": self.total_tokens(),
            },
            "llm_calls": self.llm_calls(),
            "by_agent": {
                a: {"calls": u.calls, "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens, "seconds": round(u.seconds, 1)}
                for a, u in sorted(self.agents.items())
            },
            "stage_time_s": {k: round(v, 1) for k, v in self.stages.items()},
            "ops": {k: {"count": v.count, "seconds": round(v.seconds, 1)}
                    for k, v in self.ops.items()},
            "iterations": dict(self.counters),
        }

    def render(self) -> str:
        r = self.report()
        lines = ["TELEMETRY", "-" * 40,
                 f"runtime:       {r['runtime_s']}s "
                 f"(llm {r['llm_time_s']}s + system {r['system_time_s']}s)",
                 f"llm calls:     {r['llm_calls']}",
                 f"tokens:        {r['tokens']['total']} "
                 f"(prompt {r['tokens']['prompt']}, completion {r['tokens']['completion']})",
                 "by agent:"]
        for a, u in r["by_agent"].items():
            lines.append(f"  {a:<8} calls={u['calls']} tokens={u['total_tokens']} "
                         f"(p{u['prompt_tokens']}/c{u['completion_tokens']}) "
                         f"time={u['seconds']}s")
        lines.append("stage time:")
        for k, v in r["stage_time_s"].items():
            lines.append(f"  {k:<14} {v}s")
        if r["ops"]:
            lines.append("ops:")
            for k, v in r["ops"].items():
                lines.append(f"  {k:<14} x{v['count']}  {v['seconds']}s")
        lines.append(f"iterations:    {r['iterations']}")
        return "\n".join(lines)

    def write(self, path) -> None:
        Path(path).write_text(json.dumps(self.report(), indent=2))
