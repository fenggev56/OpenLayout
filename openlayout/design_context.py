"""Design context — compact, current-state record of the design.

``design_context.json`` holds ONLY the current snapshot: circuit facts plus,
per stage, {status, params, metrics, key_issues, last_action}. Every field is
UPDATED in place, never appended, so the context uploaded to the model stays
short.  Everything historical (attempts, superseded issues/actions, notes)
is appended to ``design_context_history.jsonl`` — one JSON event per line —
for the human audit trail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# metrics worth showing to the model, in display order
_KEY_METRICS = ("wns_setup", "wns_hold", "tns", "drc_violations",
                "placement_violations", "grt_overflow", "wirelength_um",
                "design_area_um2", "utilization_pct", "power_total_w",
                "ir_worst_drop_v", "crashed")


class DesignContext:
    def __init__(self, path: Path, design: str = "", top: str = ""):
        self.path = Path(path)
        self.history_path = self.path.with_name("design_context_history.jsonl")
        self.data: dict = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                self.data = {}
        self.data.setdefault("design", design)
        self.data.setdefault("top", top)
        self.data.setdefault("info", {})
        self.data.setdefault("stages", {})
        self._migrate_legacy()
        self.save()

    def _migrate_legacy(self) -> None:
        """Move any old append-style lists out of the snapshot into history."""
        for name, rec in self.data.get("stages", {}).items():
            legacy = {}
            for key in ("issues", "actions", "attempts", "prior_issues",
                        "prior_actions", "prior_attempts"):
                if rec.get(key):
                    legacy[key] = rec.pop(key)
            if legacy:
                self.record(name, "legacy_migration", legacy)
            rec.setdefault("key_issues", [])
        self.data.pop("notes", None)

    # ------------------------------------------------------------------
    # current snapshot (update in place)
    # ------------------------------------------------------------------
    def stage_record(self, stage: str) -> dict:
        return self.data["stages"].setdefault(stage, {
            "status": "pending", "params": {}, "metrics": {},
            "key_issues": [], "last_action": None})

    def set_info(self, **kv) -> None:
        self.data["info"].update({k: v for k, v in kv.items()
                                  if v is not None})
        self.save()

    def update_stage(self, stage: str, *, status: str | None = None,
                     metrics: dict | None = None,
                     params: dict | None = None) -> None:
        rec = self.stage_record(stage)
        if status is not None:
            rec["status"] = status
        if metrics is not None:
            rec["metrics"] = metrics
        if params is not None:
            rec["params"] = params
        self.save()

    def set_issues(self, stage: str, issues: list[str]) -> None:
        """REPLACE the stage's key issues with the latest check's findings."""
        rec = self.stage_record(stage)
        rec["key_issues"] = [str(i)[:200] for i in issues[:3]]
        self.save()

    def set_last_action(self, stage: str, action: dict) -> None:
        """REPLACE the stage's last optimizer action (full detail -> history)."""
        rec = self.stage_record(stage)
        rec["last_action"] = action
        self.record(stage, "action", action)
        self.save()

    # ------------------------------------------------------------------
    # history (append-only, separate file, never uploaded)
    # ------------------------------------------------------------------
    def record(self, stage: str, kind: str, payload) -> None:
        try:
            with open(self.history_path, "a") as f:
                f.write(json.dumps({"t": time.strftime("%F %T"),
                                    "stage": stage, "kind": kind,
                                    "data": payload}) + "\n")
        except OSError:
            pass

    def note(self, text: str) -> None:
        self.record("flow", "note", text)

    def stage_history(self, stage: str, kinds=("attempt", "action"),
                      limit: int = 8, max_chars: int = 1600) -> str:
        """Compact recall of THIS stage's own past attempts/actions from the
        history file — given only to that stage's Optimizer so it can learn
        from earlier invocations without other stages inflating the prompt."""
        events: list[str] = []
        try:
            for line in self.history_path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("stage") != stage or e.get("kind") not in kinds:
                    continue
                d = e.get("data")
                events.append(f"- {e.get('kind')}: {json.dumps(d)[:220]}")
        except OSError:
            return ""
        return "\n".join(events[-limit:])[:max_chars]

    # ------------------------------------------------------------------
    # compact summary uploaded to the model
    # ------------------------------------------------------------------
    def summary_for_prompt(self, max_chars: int = 1800) -> str:
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4g}"
            return str(v)

        lines = [f"design={self.data.get('design')} "
                 f"top={self.data.get('top')}"]
        info = self.data.get("info", {})
        if info:
            lines.append("circuit: " + " ".join(
                f"{k}={fmt(v)}" for k, v in info.items()
                if not isinstance(v, dict))
                + (" clocks=" + json.dumps(info.get("clocks"))
                   if info.get("clocks") else ""))
        for name, rec in self.data.get("stages", {}).items():
            m = rec.get("metrics", {})
            mtxt = " ".join(f"{k}={fmt(m[k])}" for k in _KEY_METRICS
                            if k in m)
            line = f"[{name}] {rec.get('status')} | {mtxt}"
            if rec.get("key_issues"):
                line += " | issues: " + "; ".join(rec["key_issues"][:2])
            lines.append(line)
        return "\n".join(lines)[:max_chars]

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2))
        except OSError:
            pass
