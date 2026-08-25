"""Snapshot registry — *which* image to look at when there are many.

The flow saves a PNG at every stage and every refinement round, so after a few
iterations there are dozens of images. Rather than flood the VLM (each image is
~1k+ vision tokens), Python keeps a catalog of every snapshot tagged with its
stage, sequence and a short description, and exposes:

  * ``catalog()`` — a compact listing the agent reads to decide what to view,
    surfacing only the latest few per stage so the list stays small.
  * ``resolve(which)`` — turn an agent request ("latest", a stage key, or a
    filename fragment) into a concrete path.
  * ``recommend(stage)`` — the single most relevant image for a decision
    (e.g. Gate B -> latest fast-placed), so a checker is always pointed at the
    right picture without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Snapshot:
    seq: int
    stage: str            # init | macro_placed | fast_placed | detail_placed | routed | ...
    path: str
    desc: str = ""
    meta: dict = field(default_factory=dict)   # density / overflow / round / ...

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


# Map a decision context -> the stage whose latest image answers it.
_RECOMMEND = {
    "gate_a": "macro_pin_sides",
    "gate_b": "fast_placed",
    "detail": "detail_placed",
    "route":  "routed",
}


class SnapshotRegistry:
    def __init__(self) -> None:
        self._items: list[Snapshot] = []
        self._seq = 0

    # ------------------------------------------------------------------
    def register(self, stage: str, path: str, desc: str = "", **meta) -> Snapshot:
        self._seq += 1
        snap = Snapshot(seq=self._seq, stage=stage, path=str(path),
                        desc=desc, meta=meta)
        self._items.append(snap)
        return snap

    def all(self) -> list[Snapshot]:
        return list(self._items)

    def latest(self, stage: str | None = None) -> Snapshot | None:
        items = self._items if stage is None else [s for s in self._items if s.stage == stage]
        return items[-1] if items else None

    def stages(self) -> list[str]:
        seen: list[str] = []
        for s in self._items:
            if s.stage not in seen:
                seen.append(s.stage)
        return seen

    # ------------------------------------------------------------------
    def resolve(self, which: str | None) -> Snapshot | None:
        """Turn an agent request into a snapshot.

        ``which`` may be: None / "latest" (most recent overall), a known stage
        key (latest of that stage), or any substring of a filename/stage (latest
        match). Returns None if nothing matches.
        """
        if which is None or which.strip().lower() in ("latest", "current", "most recent"):
            return self.latest()
        key = which.strip()
        # Exact stage key first.
        by_stage = self.latest(key)
        if by_stage is not None:
            return by_stage
        # Otherwise substring match on filename or stage, newest wins.
        low = key.lower()
        matches = [s for s in self._items
                   if low in s.name.lower() or low in s.stage.lower()]
        return matches[-1] if matches else None

    def recommend(self, context: str) -> Snapshot | None:
        """Most relevant snapshot for a decision context (gate_a/gate_b/detail/
        route), falling back to the latest overall."""
        stage = _RECOMMEND.get(context.strip().lower())
        if stage:
            s = self.latest(stage)
            if s is not None:
                return s
            if context.strip().lower() == "gate_a":
                s = self.latest("macro_placed")
                if s is not None:
                    return s
        return self.latest()

    # ------------------------------------------------------------------
    def catalog(self, per_stage: int = 3) -> str:
        """Compact, agent-readable listing grouped by stage, newest first,
        capped at ``per_stage`` per stage so it never floods the context."""
        if not self._items:
            return "(no snapshots saved yet)"
        groups: dict[str, list[Snapshot]] = {}
        for s in self._items:
            groups.setdefault(s.stage, []).append(s)
        lines = [f"{len(self._items)} snapshot(s) across {len(groups)} stage(s); "
                 f"showing the latest {per_stage} per stage:"]
        # Preserve first-seen stage order.
        for stage in self.stages():
            items = groups[stage][-per_stage:][::-1]
            lines.append(f"  [{stage}]")
            for s in items:
                extra = ""
                if s.meta:
                    bits = [f"{k}={v}" for k, v in s.meta.items()
                            if k in ("density", "overflow", "overlaps", "round", "drc")]
                    if bits:
                        extra = "  (" + ", ".join(bits) + ")"
                tag = "  <- latest" if s is self._items[-1] else ""
                desc = f" — {s.desc}" if s.desc else ""
                lines.append(f"    #{s.seq} {s.name}{desc}{extra}{tag}")
        lines.append('Use view_layout(which="<stage or filename or latest>") to inspect one.')
        return "\n".join(lines)
