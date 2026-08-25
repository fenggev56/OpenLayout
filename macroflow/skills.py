"""Skill + stage library — turns ``skills/skills_Nangate45/<n>.<stage>/`` into
first-class, agent-drivable flow stages.

Each numbered directory is a stage that contributes:
  * **knowledge** — one or more SKILL markdown files (the agent's playbook),
  * **an executable** — a TCL template under ``template/`` (rendered per design
    by :class:`macroflow.config.FlowConfig`), and
  * **a kind** — ``"agentic"`` (macro placement: Planner/Placer/Checker drive
    it) or ``"script"`` (init / fast-place / detail-place / cts-route: Python
    renders + sources the template; an agent may still choose to run it).

This is what lets the *same* orchestrator implement macro placement **and** the
surrounding stages from the skills directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import SKILLS_ROOT


@dataclass
class Stage:
    idx: int
    key: str                       # short stable id used by tools/orchestrator
    title: str
    dirname: str                   # directory under SKILLS_ROOT
    kind: str                      # "agentic" | "script"
    template: str | None           # filename under <dir>/template, or None
    skills: dict[str, str]         # role -> SKILL markdown filename
    purpose: str = ""              # one-line description for the agent

    def template_path(self, root: Path = SKILLS_ROOT) -> Path | None:
        if not self.template:
            return None
        return root / self.dirname / "template" / self.template


# Ordered flow. macro_place is the rich multi-agent stage; the rest are
# script stages whose templates are proven and parameter-light.
STAGES: list[Stage] = [
    Stage(0, "init", "Initialize design", "0.init", "script", "floorplan.tcl",
          {"default": "SKILL_init.md"},
          "Read LEF/lib/verilog/def, set RC, floorplan, write floorplan.odb."),
    Stage(1, "macro_place", "Macro placement", "1.macro_place", "agentic", None,
          {"planner": "SKILL_planner.md", "placer": "SKILL_planner.md",
           "checker": "SKILL_checker.md"},
          "Plan + place macro locations with Planner+Checker agents (Gate A)."),
    Stage(2, "detail_place", "Detailed placement", "2.detail_place", "script",
          "detail_place.tcl", {"default": "skill.md"},
          "FIRM macros, cut_rows, 5um halo, PDN, global+detailed placement."),
    Stage(3, "post_route", "CTS + route", "3.post_detail_placement", "script",
          "cts_route_light.tcl", {"default": "skill.md"},
          "Clock tree synthesis + global/detailed routing -> PPA."),
    Stage(4, "ppa_check", "Post-route PPA check", "4.PPA_checker", "evaluator", None,
          {"default": "SKILL.md"},
          "Read post-route TNS/WNS; decide STOP (goal met / budget) or Gate-B CONTINUE."),
    Stage(5, "optimizer", "Macro-placement optimizer", "5.optimizer", "agentic", None,
          {"default": "SKILL.md"},
          "Refine the winning macro placement to improve PPA, then re-run backend."),
]

_BY_KEY = {s.key: s for s in STAGES}


class SkillLibrary:
    """Loads stage definitions, skill knowledge, and template paths."""

    def __init__(self, root: Path = SKILLS_ROOT) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    def stages(self) -> list[Stage]:
        return list(STAGES)

    def stage(self, key: str) -> Stage:
        if key not in _BY_KEY:
            raise KeyError(f"unknown stage '{key}'. Known: {sorted(_BY_KEY)}")
        return _BY_KEY[key]

    def has_stage(self, key: str) -> bool:
        return key in _BY_KEY

    # ------------------------------------------------------------------
    def skill_text(self, stage_key: str, role: str = "default") -> str:
        """Return the SKILL markdown for a stage/role (front-matter stripped).

        Falls back to the stage's only skill if the role isn't present.
        """
        stage = self.stage(stage_key)
        fname = stage.skills.get(role)
        if fname is None:
            # Single-skill stage, or unknown role: use the first available.
            fname = next(iter(stage.skills.values()))
        path = self.root / stage.dirname / fname
        if not path.exists():
            return ""
        return _strip_front_matter(path.read_text())

    def template_path(self, stage_key: str) -> Path | None:
        return self.stage(stage_key).template_path(self.root)

    def overview(self) -> str:
        """One-line-per-stage summary the orchestrator can show an agent."""
        lines = ["Flow stages (skill-driven):"]
        for s in STAGES:
            lines.append(f"  {s.idx}. {s.key} [{s.kind}] — {s.purpose}")
        return "\n".join(lines)


def _strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block (``---`` ... ``---``)."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            return text[nl + 1:].lstrip("\n") if nl != -1 else ""
    return text
