"""Candidate file-management for the goal-driven exploration loop.

Each fully-routed macro-placement candidate is persisted under
``<data_dir>/candidates/cand_NN/`` so the loop can:

  * **rank** candidates by the goal and remember the best,
  * **re-load** a candidate's exact coordinates to fine-refine it (the user's
    "give the previous coordinate and image for the model to modify"), and
  * give a human a per-candidate audit trail (coords + images + metrics).

Per-candidate files: ``coords.json`` (the placement, by macro name so it survives
re-init), ``metrics.json`` (routed PPA), ``meta.json`` (kind/shape/parent/goal),
and copied PNGs (``macro.png`` / ``detail.png`` / ``routed.png``). A top-level
``index.json`` is the ledger + the ``best`` pointer.

**Disk control:** routed ``.odb`` files are large, so only the *current best*
keeps its routed odb (the orchestrator promotes it to ``best_overall.odb``);
losers' odbs are pruned on commit. The small coords/metrics/images are kept for
every candidate.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .goal import Goal


@dataclass
class Candidate:
    id: int
    kind: str                       # "seed" | "coarse" | "fine"
    shape: str = ""                 # outline name (coarse) or parent's shape (fine)
    parent: int | None = None       # candidate id this was derived from (fine)
    coords: list = field(default_factory=list)     # [{name,x,y,orient}]
    metrics: dict = field(default_factory=dict)    # routed PPA
    goal_met: bool = False
    score: tuple = ()
    decision: str = ""
    routed_odb: str | None = None
    images: dict = field(default_factory=dict)     # {macro,detail,routed} -> path
    timestamp: float = 0.0
    dir: str = ""

    @property
    def tag(self) -> str:
        return f"cand_{self.id:02d}"

    def summary_row(self) -> str:
        m = self.metrics or {}
        wns = m.get("wns", "?"); tns = m.get("tns", "?"); drc = m.get("drc", "?")
        par = "" if self.parent is None else f" <-#{self.parent}"
        return (f"  #{self.id:<2} {self.kind:<6} {self.shape:<10}{par:<6} "
                f"wns={wns} tns={tns} drc={drc} "
                f"{'GOAL-MET' if self.goal_met else ''} {self.decision}")


class CandidateStore:
    def __init__(self, root: Path, goal: Goal):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.goal = goal
        self._items: list[Candidate] = []
        self._best_id: int | None = None
        self._seq = 0

    # ------------------------------------------------------------------
    def new(self, kind: str, coords: list, shape: str = "",
            parent: int | None = None) -> Candidate:
        """Open a candidate: assign an id, snapshot its coordinates to disk. Call
        :meth:`commit` once it has been routed and measured."""
        cid = self._seq
        self._seq += 1
        cdir = self.root / f"cand_{cid:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        coords = [{"name": e["name"], "x": round(float(e["x"]), 3),
                   "y": round(float(e["y"]), 3), "orient": e.get("orient", "R0")}
                  for e in coords]
        cand = Candidate(id=cid, kind=kind, shape=shape, parent=parent,
                         coords=coords, timestamp=time.time(), dir=str(cdir))
        (cdir / "coords.json").write_text(json.dumps(coords, indent=2))
        self._items.append(cand)
        return cand

    def commit(self, cand: Candidate, metrics: dict, goal_status, decision: str,
               routed_odb: str | None = None, images: dict | None = None) -> bool:
        """Record a routed candidate's results, copy its images, update the best
        pointer (pruning the loser's routed odb), and rewrite the ledger. Returns
        True if this candidate is the new best."""
        cand.metrics = dict(metrics or {})
        cand.goal_met = bool(getattr(goal_status, "met", False))
        cand.score = tuple(getattr(goal_status, "score", ()))
        cand.decision = decision
        cand.routed_odb = str(routed_odb) if routed_odb else None
        cdir = Path(cand.dir)
        (cdir / "metrics.json").write_text(json.dumps(cand.metrics, indent=2, default=str))
        if images:
            for key, src in images.items():
                if src and Path(src).exists():
                    dst = cdir / f"{key}.png"
                    try:
                        shutil.copy2(src, dst)
                        cand.images[key] = str(dst)
                    except OSError:
                        cand.images[key] = str(src)
        self._write_meta(cand)

        is_new_best = self._update_best(cand)
        self.write_index()
        return is_new_best

    # ------------------------------------------------------------------
    def _update_best(self, cand: Candidate) -> bool:
        prev = self.best()
        if prev is None or self.goal.is_better(cand.metrics, prev.metrics):
            # New best: prune the previous best's routed odb to bound disk use.
            if prev is not None and prev.routed_odb and prev.id != cand.id:
                self._prune_odb(prev.routed_odb)
            self._best_id = cand.id
            return True
        # Not the best — its routed odb is dead weight.
        if cand.routed_odb:
            self._prune_odb(cand.routed_odb)
            cand.routed_odb = None
        return False

    @staticmethod
    def _prune_odb(path: str) -> None:
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    def best(self) -> Candidate | None:
        if self._best_id is None:
            return None
        return next((c for c in self._items if c.id == self._best_id), None)

    def all(self) -> list[Candidate]:
        return list(self._items)

    def load_coords(self, cand_id: int) -> list:
        """Re-read a candidate's coordinates (for fine refinement)."""
        cdir = self.root / f"cand_{cand_id:02d}"
        f = cdir / "coords.json"
        return json.loads(f.read_text()) if f.exists() else []

    # ------------------------------------------------------------------
    def _write_meta(self, cand: Candidate) -> None:
        meta = {k: v for k, v in asdict(cand).items() if k != "coords"}
        (Path(cand.dir) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def write_index(self) -> None:
        index = {
            "best": self._best_id,
            "goal": self.goal.describe(),
            "candidates": [
                {"id": c.id, "kind": c.kind, "shape": c.shape, "parent": c.parent,
                 "metrics": c.metrics, "goal_met": c.goal_met, "decision": c.decision,
                 "dir": c.dir, "routed_odb": c.routed_odb, "images": c.images}
                for c in self._items
            ],
        }
        (self.root / "index.json").write_text(json.dumps(index, indent=2, default=str))

    def render_table(self) -> str:
        lines = [f"candidates ({len(self._items)}), goal: {self.goal.describe()}"]
        for c in self._items:
            star = "  <- BEST" if c.id == self._best_id else ""
            lines.append(c.summary_row() + star)
        return "\n".join(lines)


@dataclass
class MacroCandidate:
    id: int
    tag: str
    strategy: str = ""
    coords: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    gate_a: dict = field(default_factory=dict)
    gate_b: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    odbs: dict = field(default_factory=dict)
    timestamp: float = 0.0
    dir: str = ""

    def summary_row(self) -> str:
        density = self.metrics.get("best_density", self.metrics.get("density", "?"))
        routed = self.metrics.get("routed") or {}
        wns = routed.get("wns", self.metrics.get("routed_wns", "?"))
        tns = routed.get("tns", self.metrics.get("routed_tns", "?"))
        drc = routed.get("drc", self.metrics.get("routed_drc", "?"))
        power = routed.get("power", self.metrics.get("routed_power", "?"))
        area = routed.get("area", self.metrics.get("routed_area", "?"))
        ga = self.gate_a.get("verdict", "?")
        gb = self.gate_b.get("verdict", "?")
        return (f"  #{self.id:<2} {self.tag:<8} density={density} "
                f"GateA={ga} GateB={gb} wns={wns} tns={tns} "
                f"drc={drc} power={power} area={area}")


class MacroCandidateStore:
    """Macro-placement candidate ledger.

    The audit trail stores coordinates, gate verdicts, density, routed timing
    metrics, images and OpenROAD checkpoints for each macro-placement candidate.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._items: list[MacroCandidate] = []
        self._best_id: int | None = None
        self._seq = 0

    def new(self, strategy: str = "") -> MacroCandidate:
        cid = self._seq
        self._seq += 1
        tag = f"cand{cid:02d}"
        cdir = self.root / f"macro_cand_{cid:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        cand = MacroCandidate(id=cid, tag=tag, strategy=strategy,
                              timestamp=time.time(), dir=str(cdir))
        (cdir / "strategy.txt").write_text(strategy or "")
        self._items.append(cand)
        return cand

    def commit(self, cand: MacroCandidate, coords: list, metrics: dict | None = None,
               gate_a: dict | None = None, gate_b: dict | None = None,
               images: dict | None = None, odbs: dict | None = None) -> bool:
        cdir = Path(cand.dir)
        cand.coords = [{"name": e["name"], "x": round(float(e["x"]), 3),
                        "y": round(float(e["y"]), 3),
                        "orient": e.get("orient", "R0")}
                       for e in (coords or [])]
        cand.metrics = dict(metrics or {})
        cand.gate_a = dict(gate_a or {})
        cand.gate_b = dict(gate_b or {})
        (cdir / "coords.json").write_text(json.dumps(cand.coords, indent=2))
        (cdir / "metrics.json").write_text(
            json.dumps(cand.metrics, indent=2, default=str))
        (cdir / "gate_a.json").write_text(
            json.dumps(cand.gate_a, indent=2, default=str))
        (cdir / "gate_b.json").write_text(
            json.dumps(cand.gate_b, indent=2, default=str))

        if images:
            for key, src in images.items():
                copied = self._copy(src, cdir / f"{key}.png")
                if copied:
                    cand.images[key] = copied
        if odbs:
            for key, src in odbs.items():
                copied = self._copy(src, cdir / f"{key}.odb")
                if copied:
                    cand.odbs[key] = copied

        is_best = self._update_best(cand)
        self._prune_odbs()
        for item in self._items:
            self._write_meta(item)
        self.write_index()
        return is_best

    def _update_best(self, cand: MacroCandidate) -> bool:
        prev = self.best()
        if prev is None or self._score(cand) > self._score(prev):
            self._best_id = cand.id
            return True
        return False

    @staticmethod
    def _score(cand: MacroCandidate) -> tuple:
        ga = 1 if cand.gate_a.get("proceed") else 0
        gb = 1 if cand.gate_b.get("proceed") else 0
        routed = cand.metrics.get("routed") or {}
        if not isinstance(routed, dict):
            routed = {}
        has_route = 1 if routed or any(k.startswith("routed_")
                                      for k in cand.metrics) else 0
        wns = MacroCandidateStore._metric(
            routed.get("wns", cand.metrics.get("routed_wns")), default=-1e9)
        tns = MacroCandidateStore._metric(
            routed.get("tns", cand.metrics.get("routed_tns")), default=-1e12)
        drc = MacroCandidateStore._metric(
            routed.get("drc", cand.metrics.get("routed_drc")), default=0.0)
        area = MacroCandidateStore._metric(
            routed.get("area", cand.metrics.get("routed_area")), default=1e18)
        wirelength = MacroCandidateStore._metric(
            routed.get("wirelength", cand.metrics.get("routed_wirelength")),
            default=1e18)
        density = cand.metrics.get("best_density", cand.metrics.get("density", 0.0))
        try:
            density = float(density or 0.0)
        except (TypeError, ValueError):
            density = 0.0
        overlaps = cand.metrics.get("overlaps", 0)
        try:
            overlaps = int(overlaps if overlaps is not None else 0)
        except (TypeError, ValueError):
            overlaps = 0
        if has_route:
            return has_route, -drc, tns, wns, density, -area, -wirelength, ga, gb, -overlaps
        return has_route, ga, gb, density, -overlaps

    @staticmethod
    def _metric(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _copy(src, dst: Path) -> str | None:
        try:
            if src and Path(src).exists():
                shutil.copy2(src, dst)
                return str(dst)
        except OSError:
            pass
        return None

    def _write_meta(self, cand: MacroCandidate) -> None:
        cdir = Path(cand.dir)
        (cdir / "meta.json").write_text(json.dumps({
            "id": cand.id,
            "tag": cand.tag,
            "timestamp": cand.timestamp,
            "strategy": cand.strategy,
            "metrics": cand.metrics,
            "gate_a": cand.gate_a,
            "gate_b": cand.gate_b,
            "images": cand.images,
            "odbs": cand.odbs,
        }, indent=2, default=str))

    def _prune_odbs(self) -> None:
        """Keep disk usage bounded: only the best candidate's macro-placed and
        routed ODBs remain.

        Candidate JSON, images and metrics are the audit trail. Intermediate fast
        and detail ODBs are regenerated from coordinates when needed, and non-best
        ODBs are not useful after scoring. The best candidate keeps both its
        macro-placed and routed ODBs (both are requested deliverables).
        """
        for cand in self._items:
            keep: dict[str, str] = {}
            for key, path in list((cand.odbs or {}).items()):
                should_keep = cand.id == self._best_id and key in ("routed", "macro")
                if should_keep:
                    keep[key] = path
                    continue
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
            cand.odbs = keep

    def best(self) -> MacroCandidate | None:
        if self._best_id is None:
            return None
        return next((c for c in self._items if c.id == self._best_id), None)

    def all(self) -> list[MacroCandidate]:
        return list(self._items)

    def write_index(self) -> None:
        index = {
            "best": self._best_id,
            "candidates": [
                {k: v for k, v in asdict(c).items() if k != "coords"}
                for c in self._items
            ],
        }
        (self.root / "index.json").write_text(json.dumps(index, indent=2, default=str))

    def render_table(self) -> str:
        lines = [f"macro candidates ({len(self._items)})"]
        for c in self._items:
            star = "  <- BEST" if c.id == self._best_id else ""
            lines.append(c.summary_row() + star)
        return "\n".join(lines)
