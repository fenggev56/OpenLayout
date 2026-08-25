#!/usr/bin/env python3
"""macroflow entry point — multi-agent macro placement & physical-design flow.

A structured rebuild of the macro-placement system: LLM agents (Planner /
Placer / Checker) make the design decisions; the :mod:`macroflow` package
handles every system operation; the flow is skill-driven so the same
orchestrator can run the surrounding stages from ``skills/``.

The validated single-file ``run_macro_agent.py`` is left untouched as the
reference implementation; this is the new structured package.

Usage:
    conda activate coder
    python run_flow.py \
        --data-dir Benchmark/PlaceReasoner-Bench/<design> \
        --design <design> --top <top-module> \
        --gate-b-max-refine 4

Data dir must contain ``<design>.v``, ``constraint.sdc``, ``def/pre_place.def``,
``lef/`` and ``lib/``. Output dirs (images/saves_db/report/model_output) are
created under a unique ``./log_debug/<timestamp>_<design>/`` run directory by
default, or under exactly ``--work-dir`` when provided. The current flow saves
macro-placement candidates, fast-placement feedback, detailed placement,
CTS/route timing, images, ODBs and summaries.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = "Benchmark/PlaceReasoner-Bench/ariane133/"
DEFAULT_DESIGN = "ariane133"
DEFAULT_TOP = "ariane133"
DEFAULT_WORK_DIR = str(PROJECT_ROOT / "log_debug")


def _default_run_dir(design: str) -> Path:
    """Return a collision-free default run directory under log_debug."""
    root = Path(DEFAULT_WORK_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_design = "".join(c if c.isalnum() or c in "._-" else "_" for c in design)
    base = root / f"{stamp}_{safe_design}"
    if not base.exists():
        return base
    for i in range(1, 1000):
        candidate = root / f"{stamp}_{safe_design}_{i:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique run directory under {root}")


def _record_latest_run(run_dir: Path) -> None:
    """Best-effort pointer to the most recent default run."""
    try:
        root = Path(DEFAULT_WORK_DIR)
        root.mkdir(parents=True, exist_ok=True)
        (root / "latest_run.txt").write_text(str(run_dir) + "\n")
        latest = root / "latest"
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        pass


def main() -> int:
    # Headless Qt for save_image.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_FONTDIR", None)
    os.environ["QT_QPA_PLATFORM"] = "minimal"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    p = argparse.ArgumentParser(description="macroflow — multi-agent macro placement")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="dataset dir (<design>.v, def/, lef/, lib/, constraint.sdc)")
    p.add_argument("--design", default=DEFAULT_DESIGN,
                   help="design / proj_name; netlist at <data-dir>/<design>.v")
    p.add_argument("--top", default=DEFAULT_TOP, help="top-level module name")
    p.add_argument("--config", default="config_macro.yaml",
                   help="model YAML (local Qwen3-VL by default)")
    p.add_argument("--floorplan-def", default=None,
                   help="predefined floorplan DEF the init stage reads, relative "
                        "to --data-dir (e.g. def/1_1_u60_floorplan.def for ar1 or "
                        "def/2_1_floorplan.def for ar2). Defaults to the "
                        "FlowConfig default (def/2_1_floorplan.def).")
    p.add_argument("--tech-dir", default=None,
                   help="PDK dir holding lib/NangateOpenCellLibrary_typical.lib and "
                        "lef/NangateOpenCellLibrary.{tech,macro.mod}.lef "
                        "(default: --data-dir)")
    p.add_argument("--work-dir", default=None,
                   help="workspace for generated artifacts; inputs stay in "
                        "--data-dir, outputs go here. If omitted, a unique "
                        f"run directory is created under {DEFAULT_WORK_DIR}")
    p.add_argument("--max-refine", type=int, default=None,
                   help="deprecated alias for --gate-b-max-refine")
    p.add_argument("--gate-a-max-refine", type=int, default=None,
                   help="Gate-A macro-placement check/refinement tries "
                        "(default from config flow.gate_a_max_refine)")
    p.add_argument("--gate-b-max-refine", type=int, default=None,
                   help="Gate-B fast-placement check/refinement tries "
                        "(default from config flow.gate_b_max_refine)")
    p.add_argument("--macro-candidates", type=int, default=None,
                   help="number of distinct macro-placement shape candidates for "
                        "the Planner to generate (default from config "
                        "flow.macro_candidates)")
    p.add_argument("--parallel-workers", type=int, default=None,
                   help="macro candidate worker count from config; values >1 run "
                        "candidate pipelines in isolated worker processes")
    p.add_argument("--post-route-opt-loops", type=int, default=None,
                   help="after the first routed batch, repeat this many "
                        "best-candidate local refinement rounds; each round "
                        "generates flow.macro_candidates variants and reruns "
                        "detail placement, CTS and route")
    p.add_argument("--route-mode", choices=["light", "timing"], default=None,
                   help="CTS/route template to use. 'light' skips expensive "
                        "repair_timing/repair_design passes; 'timing' uses the "
                        "full cts_route.tcl with timing repair.")
    p.add_argument("--goal-file", default=None,
                   help="deprecated; routed timing is recorded per candidate, but "
                        "goal exploration is disabled")
    p.add_argument("--max-tries", type=int, default=None,
                   help="deprecated; routed timing is recorded per candidate, but "
                        "goal exploration is disabled")
    p.add_argument("--stop-after", choices=["placement", "gateb", "route"],
                   default=None,
                   help="halt after a macro-flow checkpoint")
    p.add_argument("--gui", action="store_true",
                   help="render std-cell placement live in a real openroad -gui window "
                        "on $DISPLAY (cells flowing into the rows); needs an X display")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    from macroflow.config import FlowConfig
    from macroflow.orchestrator import MacroFlowOrchestrator

    work_dir = Path(args.work_dir).expanduser() if args.work_dir else \
        _default_run_dir(args.design)

    fp_kwargs = {}
    if args.floorplan_def is not None:
        fp_kwargs["floorplan_def"] = args.floorplan_def
    if args.tech_dir is not None:
        fp_kwargs["tech_dir"] = args.tech_dir
    cfg = FlowConfig(data_dir=args.data_dir, design=args.design, top=args.top,
                     model_config=args.config, work_dir=work_dir,
                     route_mode=args.route_mode, **fp_kwargs)
    if args.work_dir is None:
        _record_latest_run(cfg.output_root)
    gate_b_max = args.gate_b_max_refine if args.gate_b_max_refine is not None \
        else args.max_refine
    eff_macro_candidates = args.macro_candidates if args.macro_candidates is not None \
        else cfg.macro_candidates
    eff_gate_a = args.gate_a_max_refine if args.gate_a_max_refine is not None \
        else cfg.gate_a_max_refine
    eff_gate_b = gate_b_max if gate_b_max is not None else cfg.gate_b_max_refine
    eff_parallel = args.parallel_workers if args.parallel_workers is not None \
        else cfg.parallel_workers
    eff_post_route = args.post_route_opt_loops if args.post_route_opt_loops is not None \
        else cfg.post_route_opt_loops
    print(f"[config] data_dir={cfg.data_dir} design={cfg.design} top={cfg.top} "
          f"output_root={cfg.output_root} macro_candidates={eff_macro_candidates} "
          f"gate_a={eff_gate_a} gate_b={eff_gate_b} "
          f"parallel_workers={eff_parallel} "
          f"post_route_opt_loops={eff_post_route} "
          f"route_mode={cfg.route_mode_effective}", flush=True)

    watcher = None
    have_display = "DISPLAY" in os.environ
    if args.gui:
        if eff_parallel > 1:
            print("[gui] WARNING: --gui is ignored for parallel worker runs.",
                  flush=True)
        elif have_display:
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            from macroflow import gui
            watcher = gui.GuiWatcher(cfg.save_dir / "_gui_show.tcl")
            watcher.start()
            print("[gui] live OpenROAD window will open + refresh per stage "
                  "(kill+relaunch design; uses your X display)", flush=True)
        else:
            print("[gui] WARNING: --gui set but $DISPLAY is unset — no GUI window.",
                  flush=True)

    orch = MacroFlowOrchestrator(cfg, max_refine=gate_b_max,
                                 verbose=not args.quiet, stop_after=args.stop_after,
                                 gui=args.gui and have_display and eff_parallel <= 1,
                                 goal_file=args.goal_file, max_tries=args.max_tries,
                                 gate_a_max_refine=args.gate_a_max_refine,
                                 macro_candidates=args.macro_candidates,
                                 parallel_workers=args.parallel_workers,
                                 post_route_opt_loops=args.post_route_opt_loops)
    try:
        metrics = orch.run()
        print(f"[done] flow metrics: {metrics}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n[interrupted]", flush=True)
        return 130
    finally:
        if watcher is not None:
            watcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
