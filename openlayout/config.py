"""Flow configuration — one YAML file records paths, per-stage budgets,
design goals, LLM provider and agent knobs.  Everything the user should be
able to edit lives here; see ``configs/bp_fe.yaml`` for the reference file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _expand(v: str | None) -> str | None:
    if v is None:
        return None
    return os.path.expandvars(os.path.expanduser(str(v)))


@dataclass
class StageBudget:
    name: str
    optimizer_budget: int = 3
    # debugger budget = optimizer_budget + debugger_extra (set flow-wide)


@dataclass
class FlowConfig:
    raw: dict = field(default_factory=dict)
    config_path: Path | None = None

    # --- design ---
    design: str = "bp_fe"
    top: str = "bp_fe_top"
    data_dir: Path = Path(".")
    netlist: str = ""                 # relative to data_dir unless absolute
    sdc: str = "constraint.sdc"
    floorplan_def: str = ""           # predefined floorplan DEF (rel to data_dir)
    cluster_file: str | None = None

    # --- paths ---
    output_dir: Path = Path("output")
    image_dir: Path | None = None

    # --- goal ---
    goal: dict = field(default_factory=dict)

    # --- llm ---
    llm: dict = field(default_factory=dict)

    # --- macro stage ---
    macro: dict = field(default_factory=dict)

    # --- stages/budgets ---
    stages: list[StageBudget] = field(default_factory=list)

    # --- agents ---
    optimizer_suggestions: int = 2
    debugger_extra_budget: int = 1
    parallel_eval: bool = True
    allow_tool_queries: bool = True
    max_queries: int = 4

    # --- skills ---
    skills_flow_dir: str = "skills/skills_openroad_flow"

    # --- technology files (std-cell lib + tech/cell LEF) ---
    tech_dir: Path | None = None      # None -> fall back to data_dir

    # --- openroad ---
    openroad_exe: str = "openroad"
    threads: int = 8
    tool_timeout_s: int = 7200

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "FlowConfig":
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        cfg = cls(raw=raw, config_path=path)

        d = raw.get("design", {})
        cfg.design = d.get("name", cfg.design)
        cfg.top = d.get("top", cfg.top)
        cfg.data_dir = Path(_expand(d.get("data_dir", "."))).resolve()
        cfg.netlist = d.get("netlist", f"{cfg.design}.v")
        cfg.sdc = d.get("sdc", "constraint.sdc")
        cfg.floorplan_def = d.get("floorplan_def", "")
        cfg.cluster_file = _expand(d.get("cluster_file"))

        p = raw.get("paths", {})
        cfg.output_dir = Path(_expand(p.get("output_dir", "output"))).resolve()
        img = p.get("image_dir")
        cfg.image_dir = Path(_expand(img)).resolve() if img else \
            cfg.output_dir / "images"

        cfg.goal = raw.get("goal", {"timing_met": True, "drc_clean": True})
        cfg.llm = raw.get("llm", {})
        cfg.macro = raw.get("macro_stage", {})

        for s in raw.get("stages", []):
            cfg.stages.append(StageBudget(
                name=s["name"],
                optimizer_budget=int(s.get("optimizer_budget", 3)),
            ))

        a = raw.get("agents", {})
        cfg.optimizer_suggestions = int(a.get("optimizer_suggestions", 2))
        cfg.debugger_extra_budget = int(a.get("debugger_extra_budget", 1))
        cfg.parallel_eval = bool(a.get("parallel_eval", True))
        cfg.allow_tool_queries = bool(a.get("allow_tool_queries", True))
        cfg.max_queries = int(a.get("max_queries", 4))

        sk = raw.get("skills", {})
        cfg.skills_flow_dir = sk.get("flow_dir", cfg.skills_flow_dir)

        t = raw.get("tech", {}) or {}
        td = _expand(t.get("dir"))
        if td:
            tp = Path(td)
            cfg.tech_dir = (tp if tp.is_absolute() else PROJECT_ROOT / tp).resolve()

        o = raw.get("openroad", {})
        cfg.openroad_exe = o.get("executable", "openroad")
        cfg.threads = int(o.get("threads", 8))
        cfg.tool_timeout_s = int(o.get("timeout_s", 7200))
        return cfg

    # ------------------------------------------------------------------
    def budget_for(self, stage: str) -> StageBudget:
        for s in self.stages:
            if s.name == stage:
                return s
        return StageBudget(name=stage)

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    @property
    def goal_priority(self) -> list[str]:
        """Metric priority order for selecting the best attempt (config:
        goal.priority). Timing first by default."""
        pr = self.goal.get("priority")
        if isinstance(pr, list) and pr:
            return [str(p) for p in pr]
        return ["timing", "drc", "tns", "placement", "wirelength",
                "power", "area"]

    # resolved paths ----------------------------------------------------
    @property
    def netlist_path(self) -> Path:
        p = Path(self.netlist)
        return p if p.is_absolute() else self.data_dir / p

    @property
    def sdc_path(self) -> Path:
        p = Path(self.sdc)
        return p if p.is_absolute() else self.data_dir / p

    @property
    def floorplan_def_path(self) -> Path:
        p = Path(self.floorplan_def)
        return p if p.is_absolute() else self.data_dir / p

    @property
    def tech_dir_path(self) -> Path:
        """Directory holding ``lib/`` and ``lef/`` of the PDK (config
        ``tech.dir``); defaults to the design data dir."""
        return self.tech_dir or self.data_dir

    @property
    def stdcell_lib(self) -> Path:
        return self.tech_dir_path / "lib" / "NangateOpenCellLibrary_typical.lib"

    @property
    def save_dir(self) -> Path:
        return self.output_dir / "saves_db"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def gen_dir(self) -> Path:
        return self.output_dir / "generated_scripts"

    @property
    def dialog_dir(self) -> Path:
        return self.output_dir / "dialogs"

    def make_dirs(self) -> None:
        for p in (self.output_dir, self.save_dir, self.report_dir,
                  self.log_dir, self.gen_dir, self.dialog_dir,
                  self.image_dir):
            Path(p).mkdir(parents=True, exist_ok=True)

    # llm helpers -------------------------------------------------------
    def resolve_api_key(self) -> str | None:
        # the configured key file wins over a (possibly stale) env var
        kf = self.llm.get("api_key_file")
        if kf:
            p = Path(_expand(kf))
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                for tok in p.read_text().split():
                    if tok.startswith("sk-"):
                        return tok.strip()
                return p.read_text().strip()
        env = os.environ.get("OPENAI_API_KEY")
        if env:
            return env.strip()
        return self.llm.get("api_key")
