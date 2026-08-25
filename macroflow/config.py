"""Flow configuration — configurable circuit paths + per-stage TCL rendering.

A single :class:`FlowConfig` carries everything that is design/circuit specific
so the rest of the package is circuit-agnostic. Point it at any dataset dir
(``<data_dir>/<design>.v``, ``def/pre_place.def``, ``constraint.sdc``, ``lef/``,
``lib/``) and it derives the output dirs. By default outputs are written under
``<project>/log_debug``; pass ``work_dir`` only when a run needs a different
artifact root while still reading inputs from ``data_dir``.

Rendering (rather than send-time ``set``) is deliberate: OpenROAD session
restarts re-``source`` the init file, so the vars must live *in* the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

# Repo root = parent of this package directory. Skills + legalize.py live here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills" / "skills_Nangate45"

# Defaults for the model-config ``image:`` block (save_image rendering).
IMAGE_DEFAULTS = {
    "save_width": 1024,       # save_image -width (px); height keeps aspect
    "save_resolution": 0.1,   # fallback save_image -resolution (um/px)
    "margin": 0.03,           # expand die/pin bounds so boundary pins aren't clipped
    "pin_overlay": True,      # draw bterm boxes over the PNG for visibility
}

ROUTE_MODES = {"light", "timing"}


def load_image_settings(model_config_path) -> dict:
    """Parse the ``image:`` block of the model-config YAML into normalized fields.

    Shared by :class:`FlowConfig` and the standalone ``save_die_image`` tool so
    both render byte-for-byte identical "whole die with pins" snapshots. Missing
    or malformed values fall back to :data:`IMAGE_DEFAULTS`.

    Returns a dict with keys ``save_width`` (int|None), ``save_resolution``
    (float>0), ``margin`` (float>=0) and ``pin_overlay`` (bool).
    """
    data: dict = {}
    try:
        p = Path(model_config_path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        data = {}
    image = data.get("image", {}) if isinstance(data, dict) else {}
    if not isinstance(image, dict):
        image = {}

    out: dict = {}
    try:
        width_raw = image.get("save_width", IMAGE_DEFAULTS["save_width"])
        out["save_width"] = max(1, int(width_raw)) if width_raw else None
    except (TypeError, ValueError):
        out["save_width"] = IMAGE_DEFAULTS["save_width"]
    try:
        resolution = float(image.get("save_resolution", IMAGE_DEFAULTS["save_resolution"]))
        out["save_resolution"] = resolution if resolution > 0 else IMAGE_DEFAULTS["save_resolution"]
    except (TypeError, ValueError):
        out["save_resolution"] = IMAGE_DEFAULTS["save_resolution"]
    try:
        margin = float(image.get("margin", IMAGE_DEFAULTS["margin"]))
        out["margin"] = max(0.0, margin)
    except (TypeError, ValueError):
        out["margin"] = IMAGE_DEFAULTS["margin"]
    out["pin_overlay"] = bool(image.get("pin_overlay", IMAGE_DEFAULTS["pin_overlay"]))
    return out


@dataclass
class FlowConfig:
    """All circuit-specific paths and the model-selection yaml.

    Args:
        data_dir: Dataset directory (holds ``<design>.v``, ``def/``, ``lef/``,
            ``lib/``, ``constraint.sdc``). Outputs are created under
            ``<project>/log_debug`` unless ``work_dir`` is set.
        design: Design / proj_name; netlist expected at ``<data_dir>/<design>.v``.
        top: Top-level module name.
        model_config: YAML that selects the VLM (``config_macro.yaml`` = local
            Qwen3-VL). Resolved relative to the project root.
        work_dir: Optional workspace for all generated run artifacts.
    """

    data_dir: Path
    design: str
    top: str
    model_config: str = "config_macro.yaml"
    work_dir: Path | None = None
    route_mode: str | None = None
    validate_inputs: bool = True
    project_root: Path = PROJECT_ROOT
    # Predefined floorplan DEF the init stage reads (relative to data_dir).
    # The rewritten workflow places macros into this floorplan rather than
    # building one from def/pre_place.def (see Instructions/Get_claude_results.md).
    floorplan_def: str = "def/2_1_floorplan.def"
    # PDK directory holding ``lib/NangateOpenCellLibrary_typical.lib`` and
    # ``lef/NangateOpenCellLibrary.{tech,macro.mod}.lef``. None -> data_dir.
    tech_dir: Path | None = None

    # Derived (filled in __post_init__).
    output_root: Path = field(init=False)
    image_dir: Path = field(init=False)
    save_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    model_output: Path = field(init=False)
    gen_dir: Path = field(init=False)
    session_dir: Path = field(init=False)
    gate_a_max_refine: int = field(init=False)
    gate_b_max_refine: int = field(init=False)
    macro_candidates: int = field(init=False)
    parallel_workers: int = field(init=False)
    optimizer_variants: int = field(init=False)
    vlm_place_coords: bool = field(init=False)
    route_mode_effective: str = field(init=False)
    save_image_width: int | None = field(init=False)
    save_image_resolution: float = field(init=False)
    save_image_margin: float = field(init=False)
    save_pin_overlay: bool = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.tech_dir = Path(self.tech_dir).expanduser().absolute() \
            if self.tech_dir else self.data_dir
        if self.validate_inputs:
            self._validate_inputs()
        self.work_dir = Path(self.work_dir) if self.work_dir is not None else \
            self.project_root / "log_debug"
        # Derived script/session paths are sourced by the OpenROAD subprocess,
        # which runs in its own cwd (session_dir). A relative work_dir would make
        # those paths unresolvable there, so anchor it to an absolute path.
        self.work_dir = self.work_dir.expanduser().absolute()
        self.output_root = self.work_dir
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_root / "images"
        self.save_dir = self.output_root / "saves_db"
        self.report_dir = self.output_root / "report"
        self.model_output = self.output_root / "model_output"
        for d in (self.image_dir, self.save_dir, self.report_dir, self.model_output):
            d.mkdir(parents=True, exist_ok=True)
        script_root = self.output_root / "_generated_scripts"
        self.gen_dir = script_root / self.design
        self.gen_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_root / "openroad_session"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._load_flow_settings()

    def _validate_inputs(self) -> None:
        """Fail early on missing design inputs.

        OpenROAD TCL errors do not currently stop the Python flow by themselves.
        Without this guard a typo in ``--data-dir`` leaves the session with no
        loaded design, then later macro-placement stages emit confusing
        "No design loaded" and ``floorplan.odb`` errors.
        """
        required = [
            self.data_dir,
            self.data_dir / f"{self.design}.v",
            self.data_dir / "constraint.sdc",
            self.data_dir / self.floorplan_def,
            self.tech_dir / "lib" / "NangateOpenCellLibrary_typical.lib",
            self.tech_dir / "lef" / "NangateOpenCellLibrary.tech.lef",
            self.tech_dir / "lef" / "NangateOpenCellLibrary.macro.mod.lef",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            details = "\n  - ".join(missing)
            raise FileNotFoundError(
                "Missing required flow input(s):\n"
                f"  - {details}\n"
                "Check --data-dir, --design, and --top. For bp_fe the known "
                "local dataset path is "
                "Benchmark/PlaceReasoner-Bench/bp_fe"
            )

    @staticmethod
    def _safe_stage(stage: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage).strip())
        return name.strip("._") or "misc"

    def stage_image_dir(self, stage: str, *parts: str) -> Path:
        """Image subdirectory for one flow stage."""
        d = self.image_dir / self._safe_stage(stage)
        for part in parts:
            d = d / self._safe_stage(part)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_model_dir(self, stage: str, *parts: str) -> Path:
        """Model-output subdirectory for one flow stage/category."""
        d = self.model_output / self._safe_stage(stage)
        for part in parts:
            d = d / self._safe_stage(part)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_report_dir(self, stage: str, *parts: str) -> Path:
        """Report/log subdirectory for one flow stage."""
        d = self.report_dir / self._safe_stage(stage)
        for part in parts:
            d = d / self._safe_stage(part)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    @property
    def model_config_path(self) -> Path:
        return self.project_root / self.model_config

    def _load_flow_settings(self) -> None:
        """Read macro-loop knobs from the same YAML used to select the model.

        The YAML keeps model settings at the top level and may add:

            flow:
              macro_candidates: 3
              gate_a_max_refine: 2
              gate_b_max_refine: 4
              parallel_workers: 1
              post_route_opt_loops: 0
        """
        defaults = {
            "macro_candidates": 1,
            "gate_a_max_refine": 4,
            "gate_b_max_refine": 4,
            "parallel_workers": 1,
            "post_route_opt_loops": 0,
            # Number of macro-placement variants explored per optimizer loop
            # (decoupled from macro_candidates so a single-candidate base run
            # can still explore several refinements). Mix of VLM-guided strategy
            # variants + geometric local-search variants.
            "optimizer_variants": 4,
        }
        data = {}
        try:
            if self.model_config_path.exists():
                data = yaml.safe_load(self.model_config_path.read_text()) or {}
        except Exception:
            data = {}
        flow = data.get("flow", {}) if isinstance(data, dict) else {}
        if not isinstance(flow, dict):
            flow = {}

        def as_pos_int(key: str) -> int:
            try:
                return max(1, int(flow.get(key, defaults[key])))
            except (TypeError, ValueError):
                return defaults[key]

        def as_nonneg_int(key: str) -> int:
            try:
                return max(0, int(flow.get(key, defaults[key])))
            except (TypeError, ValueError):
                return defaults[key]

        self.macro_candidates = as_pos_int("macro_candidates")
        self.gate_a_max_refine = as_pos_int("gate_a_max_refine")
        self.gate_b_max_refine = as_pos_int("gate_b_max_refine")
        self.parallel_workers = as_pos_int("parallel_workers")
        self.post_route_opt_loops = as_nonneg_int("post_route_opt_loops")
        self.optimizer_variants = as_pos_int("optimizer_variants")
        # When true, the Placer asks the VLM for explicit macro coordinates
        # (submit_placement_plan) instead of taking the cluster-aware / large-N
        # deterministic Python placement path. This makes the model's placement
        # ability actually determine PPA (needed to evaluate a placement
        # fine-tune); with it false the flow is deterministic w.r.t. the model.
        self.vlm_place_coords = bool(flow.get("vlm_place_coords", False))
        route_mode = self.route_mode
        if route_mode is None:
            route_mode = str(flow.get("route_mode", "light")).strip().lower()
        route_mode = str(route_mode).strip().lower()
        if route_mode not in ROUTE_MODES:
            raise ValueError(
                f"invalid route_mode '{route_mode}'. Expected one of "
                f"{sorted(ROUTE_MODES)}")
        self.route_mode_effective = route_mode

        # Image rendering knobs (shared with the standalone save_die_image tool).
        img = load_image_settings(self.model_config_path)
        self.save_image_width = img["save_width"]
        self.save_image_resolution = img["save_resolution"]
        self.save_image_margin = img["margin"]
        self.save_pin_overlay = img["pin_overlay"]

    @property
    def header(self) -> str:
        """TCL header injected at the top of every rendered stage script.

        Matches the guarded ``if {![info exists ...]}`` vars the templates
        expect (``data_dir`` / ``proj_name`` / ``top_design`` / ``save_dir`` /
        ``report_dir``).
        """
        return (
            f'set data_dir "{self.data_dir}"\n'
            f'set tech_dir "{self.tech_dir}"\n'
            f'set work_dir "{self.output_root}"\n'
            f'set proj_name "{self.design}"\n'
            f'set top_design "{self.top}"\n'
            f'set save_dir "{self.save_dir}"\n'
            f'set report_dir "{self.report_dir}"\n'
            f'set fp_def "{self.data_dir / self.floorplan_def}"\n'
        )

    def render(self, template: Path) -> Path:
        """Render a stage template into ``_generated_scripts/<design>/`` with the
        design/path vars baked into the file header. Returns the rendered path.
        """
        template = Path(template)
        out = self.gen_dir / template.name
        out.write_text(self.header + "\n" + template.read_text())
        return out

    @property
    def floorplan_odb(self) -> Path:
        """Pristine post-init checkpoint (written by the init template)."""
        return self.save_dir / "floorplan.odb"

    @property
    def best_overall_odb(self) -> Path:
        """Final routed deliverable."""
        return self.save_dir / "best_overall.odb"

    @property
    def candidates_dir(self) -> Path:
        """Per-candidate exploration artifacts (coords / metrics / images)."""
        return self.output_root / "candidates"
