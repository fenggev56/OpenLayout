#!/usr/bin/env python3
"""OpenLayout — open-source multi-agent physical design flow on OpenROAD.

Usage:
    python run_openlayout.py --config configs/bp_fe.yaml
    python run_openlayout.py --config configs/bp_fe.yaml --stages floorplan macro_place
    python run_openlayout.py --config configs/bp_fe.yaml --stages detail_place cts route finish
    python run_openlayout.py --config configs/bp_fe.yaml --no-llm     # worker-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from openlayout.config import FlowConfig            # noqa: E402
from openlayout.flow import Flow                    # noqa: E402
from openlayout.stages import STAGE_ORDER           # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="OpenLayout multi-agent flow")
    p.add_argument("--config", default="configs/bp_fe.yaml")
    p.add_argument("--stages", nargs="+", default=None,
                   choices=STAGE_ORDER + ["all"],
                   help="subset of stages to run (default: all from config)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip Debugger/Optimizer agents (worker-only)")
    args = p.parse_args()

    cfg = FlowConfig.load(args.config)
    cfg.make_dirs()

    log_file = cfg.output_dir / "flow.log"
    lf = open(log_file, "a")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)
        lf.write(f"[{time.strftime('%F %T')}] {msg}\n")
        lf.flush()

    log(f"OpenLayout start — design={cfg.design} output={cfg.output_dir}")
    stages = None
    if args.stages and "all" not in args.stages:
        stages = [s for s in STAGE_ORDER if s in args.stages]

    flow = Flow(cfg, use_llm=not args.no_llm, log=log)
    report = flow.run(stages)
    log(f"OpenLayout done — goal_met={report.get('goal_met')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
