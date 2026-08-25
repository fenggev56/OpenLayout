"""Goal specification + post-route PPA check.

The exploration loop is **goal-driven**: every macro-placement candidate is taken
all the way to a routed design, its PPA metrics are read, and the :class:`Goal`
decides whether the objective is met. A :class:`Goal` is a list of
:class:`Objective` (metric / target / comparison), so it is metric-generic — the
only kind exercised today is ``timing`` (``WNS>=0 AND TNS>=0``), but adding an
area / wirelength / drc / power objective is just another line in ``goal.yaml``.

:class:`PPAChecker` turns the goal evaluation into the stop/continue decision
defined by ``skills/skills_Nangate45/5.PPA_checker/SKILL.md`` and prints that
skill's **exact, parseable** report block so the loop log is stable:

    Iteration: <N> / <try_limit>
    WNS: <value> ns   (goal: >= 0)
    TNS: <value> ns   (goal: >= 0)
    Decision: STOP (goal met) | STOP (try limit exhausted) | CONTINUE
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Metrics reported in nanoseconds (only these get the " ns" unit in the report).
_TIME_METRICS = {"wns", "tns"}

_CMP = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">":  lambda v, t: v > t,
    "<":  lambda v, t: v < t,
    "==": lambda v, t: v == t,
}


@dataclass
class Objective:
    """One goal condition, e.g. ``wns >= 0.0``."""

    metric: str
    target: float = 0.0
    direction: str = ">="

    def met(self, value) -> bool:
        if value is None:
            return False
        return _CMP.get(self.direction, _CMP[">="])(float(value), float(self.target))

    @property
    def larger_better(self) -> bool:
        """True when a larger metric value is closer to satisfying this objective."""
        return self.direction in (">=", ">")

    def margin(self, value) -> float:
        """Signed distance toward satisfaction (>=0 means met). Missing -> -inf so a
        candidate with no measurement always ranks last."""
        if value is None:
            return float("-inf")
        v, t = float(value), float(self.target)
        return (v - t) if self.larger_better else (t - v)


@dataclass
class GoalStatus:
    met: bool
    per_objective: list           # [(metric, value, target, direction, met_bool)]
    missing: list                 # metric names absent from the routed metrics
    score: tuple                  # ranking key, higher == better


@dataclass
class Goal:
    """A named bundle of objectives + the exploration try budget."""

    kind: str = "timing"
    objectives: list = field(default_factory=list)
    max_tries: int = 6
    source: str = "built-in default (timing)"

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "Goal":
        return cls(kind="timing",
                   objectives=[Objective("wns", 0.0, ">="),
                               Objective("tns", 0.0, ">=")],
                   max_tries=6, source="built-in default (timing)")

    @classmethod
    def load(cls, path) -> "Goal":
        """Parse a goal YAML. Accepts objectives under a top-level ``goal:`` key or
        at the document root."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        g = raw.get("goal", raw)
        objs = []
        for o in g.get("objectives", []) or []:
            objs.append(Objective(metric=str(o["metric"]).lower(),
                                   target=float(o.get("target", 0.0)),
                                   direction=str(o.get("direction", ">=")).strip()))
        if not objs:
            objs = cls.default().objectives
        max_tries = int(g.get("max_tries", raw.get("max_tries", 6)))
        return cls(kind=str(g.get("kind", "timing")), objectives=objs,
                   max_tries=max_tries, source=str(path))

    @classmethod
    def resolve(cls, cli_path, data_dir, project_root) -> "Goal":
        """First existing of: ``--goal-file`` > ``<data_dir>/goal.yaml`` >
        ``<project_root>/goal.yaml`` > built-in timing default."""
        for p in (cli_path and Path(cli_path),
                  Path(data_dir) / "goal.yaml",
                  Path(project_root) / "goal.yaml"):
            if p and Path(p).exists():
                try:
                    return cls.load(p)
                except Exception as e:                 # malformed -> fall through
                    print(f"[goal] could not parse {p}: {e}", flush=True)
        return cls.default()

    # ------------------------------------------------------------------
    def score(self, metrics: dict) -> tuple:
        """Ranking key (higher is better): number of objectives met, then each
        objective's margin in declaration order — so for timing it ranks by WNS
        first, then TNS, exactly the conventional priority."""
        n_met = sum(1 for o in self.objectives if o.met(metrics.get(o.metric)))
        per = tuple(o.margin(metrics.get(o.metric)) for o in self.objectives)
        return (n_met,) + per

    def evaluate(self, metrics: dict) -> GoalStatus:
        metrics = metrics or {}
        per, missing = [], []
        for o in self.objectives:
            v = metrics.get(o.metric)
            if v is None:
                missing.append(o.metric)
            per.append((o.metric, v, o.target, o.direction, o.met(v)))
        met = all(row[4] for row in per) and bool(per)
        return GoalStatus(met=met, per_objective=per, missing=missing,
                          score=self.score(metrics))

    def is_better(self, a: dict, b: dict | None) -> bool:
        """Is candidate ``a`` better than ``b`` for this goal? Ranks by score
        (timing dominates per the user's goal); DRC then wirelength break exact
        score ties, so a DRC-broken candidate never out-ranks a clean one of
        identical timing."""
        if b is None:
            return True
        sa, sb = self.score(a or {}), self.score(b or {})
        if sa != sb:
            return sa > sb
        da = (a or {}).get("drc", 0) or 0
        db = (b or {}).get("drc", 0) or 0
        if da != db:
            return da < db
        return (a or {}).get("wirelength", float("inf")) < \
               (b or {}).get("wirelength", float("inf"))

    def describe(self) -> str:
        conds = ", ".join(f"{o.metric} {o.direction} {o.target:g}" for o in self.objectives)
        return f"{self.kind} [{conds}] (budget {self.max_tries} tries, src: {self.source})"


# ---------------------------------------------------------------------------
# PPA checker — the stop/continue decision from 5.PPA_checker/SKILL.md
# ---------------------------------------------------------------------------
class Decision(enum.Enum):
    STOP_GOAL = "STOP (goal met)"
    STOP_BUDGET = "STOP (try limit exhausted)"
    CONTINUE = "CONTINUE"


class PPAChecker:
    """Deterministic post-route checker. ``skill_text`` (the SKILL.md body) is kept
    for provenance/logging; the decision logic *is* the skill's two stop
    conditions, implemented in code so the report is parseable and never hallucinated."""

    def __init__(self, goal: Goal, skill_text: str = ""):
        self.goal = goal
        self.skill = skill_text or ""

    def decide(self, try_index: int, max_tries: int, metrics: dict):
        """``try_index`` is 0-based. Returns ``(Decision, GoalStatus, report_str)``."""
        status = self.goal.evaluate(metrics)
        if status.met:
            decision = Decision.STOP_GOAL
        elif try_index + 1 >= max_tries:
            decision = Decision.STOP_BUDGET
        else:
            decision = Decision.CONTINUE
        return decision, status, self.report(try_index, max_tries, status, decision)

    def report(self, try_index: int, max_tries: int, status: GoalStatus,
               decision: Decision) -> str:
        lines = [f"Iteration: {try_index + 1} / {max_tries}"]
        for metric, value, target, direction, _met in status.per_objective:
            unit = " ns" if metric in _TIME_METRICS else ""
            vstr = "n/a" if value is None else f"{float(value):g}"
            lines.append(f"{metric.upper()}: {vstr}{unit}   (goal: {direction} {target:g})")
        lines.append(f"Decision: {decision.value}")
        return "\n".join(lines)
