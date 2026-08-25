"""Flow orchestrator — runs the whole OpenROAD flow with the multi-agent
Worker / Debugger / Optimizer loop per stage, plus the macro-placement
candidate-sampling stage merged from ``macroflow``.

Per stage:
  Worker gives the initial run; the Debugger checks it; while the check
  fails and budget remains, the Optimizer proposes up to K diverse parameter
  suggestions which run in parallel, the best (goal score) is promoted, and
  the Debugger re-checks (its budget is one larger than the Optimizer's).
  When budgets run out the stage keeps its best result and the flow
  continues — it never aborts.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import metrics as M
from .agents import (Debugger, Optimizer, Worker, goal_check, promote,
                     score_metrics)
from .config import FlowConfig, PROJECT_ROOT
from .design_context import DesignContext
from .llm import LLMClient
from .stages import ALL_STAGES, PARAM_OWNER, STAGE_ORDER, StageSpec
from .telemetry import Telemetry


class Flow:
    def __init__(self, cfg: FlowConfig, *, use_llm: bool = True,
                 log=print):
        self.cfg = cfg
        self.log = log
        cfg.make_dirs()
        self.telemetry = Telemetry(cfg.output_dir / "telemetry.json")
        self.context = DesignContext(cfg.output_dir / "design_context.json",
                                     design=cfg.design, top=cfg.top)
        self.llm = None
        if use_llm:
            try:
                self.llm = LLMClient(cfg, telemetry=self.telemetry)
            except Exception as exc:  # missing sdk/key → agent-less mode
                self.log(f"[flow] LLM unavailable ({exc}); running worker-only")
        self.worker = Worker(cfg, self.telemetry, log=log)
        self.debugger = Debugger(cfg, self.llm, log=log) if self.llm else None
        self.optimizer = Optimizer(cfg, self.llm, log=log) if self.llm else None

    # ==================================================================
    # top level
    # ==================================================================
    def run(self, stages: list[str] | None = None) -> dict:
        order = stages or [s for s in STAGE_ORDER
                           if s in self.cfg.stage_names() or s == "macro_place"]
        self.log(f"[flow] stages: {order}")
        for name in order:
            self._banner(name)
            if name == "floorplan":
                self.run_floorplan()
            elif name == "macro_place":
                self.run_macro_place()
            else:
                self.run_stage(ALL_STAGES[name])
        report = self.write_final_report()
        return report

    def _banner(self, name: str) -> None:
        self.log("=" * 70)
        self.log(f"=== STAGE {name}  ({time.strftime('%F %T')})")
        self.log("=" * 70)

    # ==================================================================
    # stage 0: floorplan (worker only — predefined DEF)
    # ==================================================================
    def run_floorplan(self) -> None:
        spec = ALL_STAGES["floorplan"]
        canonical = self.cfg.save_dir / spec.output_odb
        res, m = self.worker.run(spec, {}, "floorplan", out_odb=canonical)
        # OL_FACT lines -> design context info
        facts: dict = {}
        for k, v in re.findall(r"OL_FACT (\S+) (.+)", res.log_text):
            facts[k] = v.strip()
        clocks = re.findall(r"OL_FACT clock (\S+) period_ns (\S+)",
                            res.log_text)
        info = {k: v for k, v in facts.items() if not k.startswith("clock")}
        if clocks:
            info["clocks"] = {c: p for c, p in clocks}
        bbox = re.findall(r"OL_FACT core_bbox_um (\S+) (\S+) (\S+) (\S+)",
                          res.log_text)
        if bbox:
            info["core_bbox_um"] = [float(v) for v in bbox[-1]]
        sizes = re.findall(r"^OL_MACRO (\S+) (\S+) (\S+)$", res.log_text,
                           re.MULTILINE)
        if sizes:
            info["macro_sizes_um"] = {n: [float(w), float(h)]
                                      for n, w, h in sizes}
        self.context.set_info(**info)
        status = "done" if res.ok else "failed"
        self.context.update_stage("floorplan", status=status, metrics=m,
                                  params={})
        if not res.ok:
            self.context.note("floorplan crashed — check "
                              + str(res.log_path))

    # ==================================================================
    # stage 1: macro placement optimization (merged macroflow)
    # ==================================================================
    def run_macro_place(self) -> None:
        cfg = self.cfg
        mp = cfg.macro
        n_macros = int(self.context.data["info"].get("num_macros", 0) or 0)
        enable = mp.get("enable", "auto")
        if enable is False or str(enable).lower() == "false" or \
                (str(enable).lower() == "auto" and n_macros == 0):
            self.log(f"[macro_place] skipped (num_macros={n_macros}, "
                     f"enable={enable})")
            # downstream reads the floorplan checkpoint directly
            src = cfg.save_dir / ALL_STAGES["floorplan"].output_odb
            dst = cfg.save_dir / ALL_STAGES["macro_place"].output_odb
            if src.exists():
                shutil.copyfile(src, dst)
            self.context.update_stage("macro_place", status="skipped")
            return

        work_dir = cfg.output_dir / "macro_place"
        t0 = time.monotonic()
        ok = self._run_macroflow(work_dir, mp)
        self.telemetry.add_tool_run(stage="macro_place",
                                    seconds=time.monotonic() - t0,
                                    label="macroflow")
        cands = self._load_candidates(work_dir)
        self.log(f"[macro_place] macroflow ok={ok}, "
                 f"{len(cands)} candidate(s) loaded")

        # PPA/DRC feedback: same simple downstream scripts for any candidate
        # that does not already carry routed metrics.
        eval_mode = str(mp.get("eval_downstream", "auto")).lower()
        need_eval = [c for c in cands if c.get("coords")
                     and (eval_mode == "always"
                          or (eval_mode == "auto"
                              and not c.get("routed_metrics")))]
        if need_eval and eval_mode != "never":
            self._eval_candidates_downstream(need_eval)

        # select the best candidate
        scored = []
        for c in cands:
            m = c.get("routed_metrics") or {}
            if not m and not c.get("coords"):
                continue
            scored.append((score_metrics(m, crashed=not bool(m),
                                         priority=cfg.goal_priority), c))
        if not scored:
            self.log("[macro_place] no usable candidate — falling back to "
                     "predefined floorplan macro positions")
            shutil.copyfile(cfg.save_dir / "00_floorplan.odb",
                            cfg.save_dir / "01_macro_placed.odb")
            self.context.update_stage("macro_place", status="fallback")
            return
        scored.sort(key=lambda t: t[0])
        best = scored[0][1]
        self.log(f"[macro_place] best candidate: {best['id']} "
                 f"metrics={json.dumps(best.get('routed_metrics', {}))[:200]}")

        # apply the best coordinates onto the pristine floorplan
        # (legalized first: overlapping FIXED macros make GPL diverge)
        spec = ALL_STAGES["macro_place"]
        best["coords"] = self._legalize_coords(best["coords"], best["id"])
        coords_tcl = self._coords_tcl(best["coords"])
        res, m = self.worker.run(
            spec, {"macro_coords": coords_tcl}, "macro_place",
            in_odb=cfg.save_dir / "00_floorplan.odb",
            out_odb=cfg.save_dir / spec.output_odb)
        self.context.update_stage(
            "macro_place",
            status="done" if res.ok else "failed",
            metrics=best.get("routed_metrics", {}),
            params={"candidate": best["id"],
                    "n_candidates": len(cands)})
        self.context.set_last_action(
            "macro_place",
            {"selected_candidate": best["id"], "n_candidates": len(cands)})
        self.context.record(
            "macro_place", "candidates_scored",
            [{"id": c["id"], "metrics": c.get("routed_metrics", {})}
             for _, c in scored])

        # merge macroflow telemetry
        tj = work_dir / "model_output" / "telemetry.json"
        if not tj.exists():
            hits = list(work_dir.glob("**/telemetry.json"))
            tj = hits[0] if hits else tj
        if tj.exists():
            try:
                self.telemetry.merge_external("macro_place_macroflow",
                                              json.loads(tj.read_text()))
            except (OSError, json.JSONDecodeError):
                pass

    # ------------------------------------------------------------------
    def _run_macroflow(self, work_dir: Path, mp: dict) -> bool:
        """Run the merged macro-placement project as a subprocess."""
        cfg = self.cfg
        cmd = [sys.executable, str(PROJECT_ROOT / "run_flow.py"),
               "--data-dir", str(cfg.data_dir),
               "--design", cfg.design,
               "--top", cfg.top,
               "--config", str(mp.get("model_config",
                                      "config_macro_gpt.yaml")),
               "--work-dir", str(work_dir),
               "--macro-candidates", str(mp.get("candidates", 3)),
               "--parallel-workers", str(mp.get("parallel_workers", 1)),
               "--gate-a-max-refine", str(mp.get("gate_a_max_refine", 2)),
               "--gate-b-max-refine", str(mp.get("gate_b_max_refine", 2)),
               "--post-route-opt-loops", "0",
               "--quiet"]
        if cfg.floorplan_def:
            cmd += ["--floorplan-def", cfg.floorplan_def]
        if cfg.tech_dir:
            cmd += ["--tech-dir", str(cfg.tech_dir)]
        if mp.get("route_mode"):
            cmd += ["--route-mode", str(mp["route_mode"])]
        if mp.get("stop_after"):
            cmd += ["--stop-after", str(mp["stop_after"])]
        log_path = cfg.log_dir / "macroflow.log"
        self.log(f"[macro_place] running macroflow -> {log_path}")
        self.log("    " + " ".join(cmd))
        import os as _os
        env = _os.environ.copy()
        # a stale env key must not shadow the configured api_key_file
        env.pop("OPENAI_API_KEY", None)
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                  cwd=str(PROJECT_ROOT), env=env,
                                  timeout=int(mp.get("timeout_s", 4 * 3600)))
        return proc.returncode == 0

    def _load_candidates(self, work_dir: Path) -> list[dict]:
        """Read macroflow's candidates ledger (coords + any routed metrics)."""
        out: list[dict] = []
        # prefer the merged top-level ledger; per-worker nested candidates
        # dirs would double-count the same candidates
        cand_dirs = sorted(work_dir.glob("candidates/*cand_*"))
        if not cand_dirs:
            cand_dirs = sorted(work_dir.glob("**/candidates/*cand_*"))
        for d in cand_dirs:
            if not d.is_dir():
                continue
            c: dict = {"id": d.name, "dir": str(d)}
            try:
                # note: names may contain literal "\/" (Verilog escaped
                # hierarchy) — they match the DB as-is; do not unescape
                c["coords"] = json.loads((d / "coords.json").read_text())
            except (OSError, json.JSONDecodeError):
                c["coords"] = []
            try:
                raw = json.loads((d / "metrics.json").read_text())
                c["routed_metrics"] = self._normalize_metrics(raw)
            except (OSError, json.JSONDecodeError):
                c["routed_metrics"] = {}
            out.append(c)
        return out

    @staticmethod
    def _normalize_metrics(raw: dict) -> dict:
        """Map macroflow metric names onto the flow's metric keys."""
        flat: dict = {}

        def walk(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(v, f"{prefix}{k}.")
                elif isinstance(v, (int, float)):
                    flat[f"{prefix}{k}".lower()] = v
        if isinstance(raw, dict):
            walk(raw)
        m: dict = {}
        for key, val in flat.items():
            base = key.rsplit(".", 1)[-1]
            if base in ("wns", "wns_setup", "worst_slack") and \
                    "wns_setup" not in m:
                m["wns_setup"] = float(val)
            elif base in ("wns_hold", "worst_slack_min") and \
                    "wns_hold" not in m:
                m["wns_hold"] = float(val)
            elif base == "tns" and "tns" not in m:
                m["tns"] = float(val)
            elif "drc" in base and "drc_violations" not in m:
                m["drc_violations"] = int(val)
            elif "wirelength" in base or "wire_length" in base:
                m.setdefault("wirelength_um", float(val))
            elif base in ("hpwl",):
                m.setdefault("hpwl", float(val))
        return m

    # ------------------------------------------------------------------
    @staticmethod
    def _coords_tcl(coords: list[dict]) -> str:
        return " ".join(
            "{%s %s %s %s}" % (e["name"], e["x"], e["y"],
                               e.get("orient", "R0"))
            for e in coords)

    def _macro_geometry(self) -> tuple[dict, list] | None:
        """(macro_sizes {name: (w, h)}, core bbox [x0 y0 x1 y1]) from the
        floorplan facts; queried from the floorplan checkpoint when missing
        (e.g. an output dir produced before the facts were exported)."""
        info = self.context.data.get("info", {})
        sizes = info.get("macro_sizes_um") or {}
        bbox = info.get("core_bbox_um")
        if sizes and bbox:
            return {k: tuple(v) for k, v in sizes.items()}, list(bbox)
        odb = self.cfg.save_dir / ALL_STAGES["floorplan"].output_odb
        if not odb.exists():
            return None
        from . import runner
        tag = "macro_geometry_query"
        script = self.cfg.gen_dir / f"{tag}.tcl"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(_MACRO_GEOMETRY_TCL.replace("@ODB@", str(odb)))
        res = runner.run_openroad(script, self.cfg.log_dir / f"{tag}.log",
                                  exe=self.cfg.openroad_exe, timeout_s=600,
                                  telemetry=self.telemetry, stage="macro_place")
        b = re.findall(r"OL_FACT core_bbox_um (\S+) (\S+) (\S+) (\S+)",
                       res.log_text)
        s = re.findall(r"^OL_MACRO (\S+) (\S+) (\S+)$", res.log_text,
                       re.MULTILINE)
        if not (b and s):
            return None
        bbox = [float(v) for v in b[-1]]
        sizes = {n: (float(w), float(h)) for n, w, h in s}
        self.context.set_info(core_bbox_um=bbox,
                              macro_sizes_um={k: list(v)
                                              for k, v in sizes.items()})
        return sizes, bbox

    def _legalize_coords(self, coords: list[dict], cand_id: str) -> list[dict]:
        """Resolve macro overlaps / out-of-core positions with the repo's
        force-directed legalizer (``legalize.py``) before the coordinates
        are applied to the floorplan. Never blocks the flow."""
        if not coords:
            return coords
        geom = self._macro_geometry()
        if geom is None:
            self.log(f"    [legalize] {cand_id}: no macro geometry available "
                     "— applying coordinates as-is")
            return coords
        sizes, (x0, y0, x1, y1) = geom

        def _norm(n: str) -> str:          # coords.json may escape '/'
            return n.replace("\\/", "/")
        sizes_n = {_norm(k): v for k, v in sizes.items()}
        missing = [e["name"] for e in coords
                   if _norm(e["name"]) not in sizes_n]
        if missing:
            self.log(f"    [legalize] {cand_id}: {len(missing)} macro(s) "
                     f"without size info (e.g. {missing[0][:60]})")
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from legalize import (Macro, _oriented_size, count_overlaps,
                                  legalize_plan_entries)
            ws = [v[0] for v in sizes_n.values()] or [77.71]
            hs = [v[1] for v in sizes_n.values()] or [40.60]
            entries = [{"name": _norm(e["name"]), "x": float(e["x"]),
                        "y": float(e["y"]), "orient": e.get("orient", "R0")}
                       for e in coords]
            macros = []
            for e in entries:
                w, h = sizes_n.get(e["name"], (max(ws), max(hs)))
                w, h = _oriented_size(w, h, e["orient"])
                macros.append(Macro(e["name"], e["x"], e["y"], w, h,
                                    e["orient"]))
            before = count_overlaps(macros)
            outside = sum(1 for m in macros if m.x < x0 or m.y < y0
                          or m.x2 > x1 or m.y2 > y1)
            if before == 0 and outside == 0:
                self.log(f"    [legalize] {cand_id}: 0 overlaps, all macros "
                         "inside core — nothing to fix")
                return coords
            self.log(f"    [legalize] {cand_id}: {before} overlap(s), "
                     f"{outside} outside core -> legalizing")
            # clamp out-of-core macros first (legalize_plan_entries only acts
            # on overlaps; clamping may create overlaps it then resolves)
            for e, m in zip(entries, macros):
                e["x"] = max(x0, min(e["x"], x1 - m.w))
                e["y"] = max(y0, min(e["y"], y1 - m.h))
            fixed = legalize_plan_entries(
                entries, sizes_n, x0, y0, x1, y1,
                default_w=max(ws), default_h=max(hs), gap=2.0)
            out = [{**e, "x": round(float(f["x"]), 3),
                    "y": round(float(f["y"]), 3)}
                   for e, f in zip(coords, fixed)]
            self.context.record("macro_place", "legalized",
                                {"candidate": cand_id,
                                 "overlaps_before": before,
                                 "outside_core_before": outside})
            return out
        except Exception as exc:  # noqa: BLE001 — never block the flow
            self.log(f"    [legalize] {cand_id}: legalizer failed ({exc}); "
                     "applying coordinates as-is")
            return coords

    # ------------------------------------------------------------------
    def _eval_candidates_downstream(self, cands: list[dict]) -> None:
        """PPA/DRC feedback for macro candidates: apply coords, then run the
        SAME simple downstream scripts (detail place -> cts -> light route)
        per candidate, in parallel."""
        cfg = self.cfg
        mp = cfg.macro
        light = {"route": {"droute_end_iter": int(mp.get("eval_droute_iter", 5)),
                           "congestion_iterations": 20}}
        workers = int(mp.get("eval_parallel", min(3, len(cands))) or 1)

        def _one(c):
            base = cfg.output_dir / "macro_place" / "eval" / c["id"]
            sub = {k: base / k for k in
                   ("saves_db", "reports", "logs", "generated_scripts",
                    "images")}
            for p in sub.values():
                p.mkdir(parents=True, exist_ok=True)
            common = {"save_dir": str(sub["saves_db"]),
                      "report_dir": str(sub["reports"]),
                      "image_dir": str(sub["images"])}
            try:
                c["coords"] = self._legalize_coords(c["coords"], c["id"])
                coords_tcl = self._coords_tcl(c["coords"])
                chain = [
                    ("macro_place", {"macro_coords": coords_tcl, **common}),
                    ("detail_place", dict(common)),
                    ("cts", dict(common)),
                    ("route", {**common, **light["route"]}),
                ]
                in_odb = cfg.save_dir / "00_floorplan.odb"
                m_last: dict = {}
                for name, params in chain:
                    spec = ALL_STAGES[name]
                    out_odb = sub["saves_db"] / spec.output_odb
                    tag = f"{c['id']}_{name}"
                    res, m_last = self.worker.run(
                        spec, params, tag, in_odb=in_odb, out_odb=out_odb,
                        gen_dir=sub["generated_scripts"], log_dir=sub["logs"])
                    drc = sub["reports"] / f"{tag}.drc.rpt"
                    if drc.exists():
                        n = M.count_drc_violations(drc)
                        if n is not None:
                            m_last["drc_violations"] = n
                    if not res.ok:
                        m_last["crashed"] = True
                        break
                    in_odb = out_odb
                c["routed_metrics"] = m_last
                (base / "eval_metrics.json").write_text(
                    json.dumps(m_last, indent=2))
            except Exception as exc:  # noqa: BLE001
                self.log(f"    [macro-eval] {c['id']} failed: {exc}")
                c["routed_metrics"] = {}

        self.log(f"[macro_place] evaluating {len(cands)} candidate(s) with "
                 f"the simple downstream flow ({workers} parallel)")
        if workers > 1 and len(cands) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_one, cands))
        else:
            for c in cands:
                _one(c)

    # ==================================================================
    # generic agent stage
    # ==================================================================
    def run_stage(self, spec: StageSpec, *, params: dict | None = None,
                  agents: bool = True) -> dict:
        cfg = self.cfg
        budget = cfg.budget_for(spec.name)
        agents = agents and self.llm is not None
        opt_budget = budget.optimizer_budget if agents else 0
        debug_budget = (opt_budget + cfg.debugger_extra_budget) if agents \
            else 0
        in_odb = None
        if spec.input_stage:
            in_odb = cfg.save_dir / ALL_STAGES[spec.input_stage].output_odb
        canonical = cfg.save_dir / spec.output_odb

        p0 = dict(spec.default_params)
        p0.update(params or {})
        res, m = self.worker.run(spec, p0, spec.name, in_odb=in_odb,
                                 out_odb=canonical)
        best = {"tag": spec.name, "params": p0, "ok": res.ok, "metrics": m,
                "odb": str(canonical), "res": res, "best": True}
        history = [best]
        self.context.update_stage(spec.name, status="running", metrics=m,
                                  params=p0)
        self.context.record(spec.name, "attempt",
                            {"tag": spec.name, "params": p0, "ok": res.ok,
                             "metrics": m})

        dbg = {"verdict": "pass", "issues": []}
        if debug_budget > 0:
            dbg = self.debugger.check(
                spec, params=best["params"], m=best["metrics"],
                res=best["res"],
                context_summary=self.context.summary_for_prompt(),
                attempt_no=1, odb=Path(best["odb"]), tag=best["tag"])
            debug_budget -= 1
            self.context.set_issues(spec.name, dbg.get("issues", []))

        round_no = 0
        while (dbg.get("verdict") != "pass" and opt_budget > 0
               and debug_budget > 0):
            round_no += 1
            self.log(f"  [stage {spec.name}] optimize round {round_no} "
                     f"(opt budget left {opt_budget}, "
                     f"debug budget left {debug_budget})")
            proposal = self.optimizer.propose(
                spec, history=[{k: v for k, v in h.items() if k != "res"}
                               for h in history],
                debug_report={k: v for k, v in dbg.items()},
                context_summary=self.context.summary_for_prompt(),
                n_suggestions=cfg.optimizer_suggestions,
                odb=Path(best["odb"]), tag=f"{spec.name}_r{round_no}",
                own_history=self.context.stage_history(spec.name))
            opt_budget -= 1
            suggestions = proposal.get("suggestions", [])
            if proposal.get("stop") or not suggestions:
                self.context.set_last_action(
                    spec.name, {"round": round_no, "optimizer_stop": True,
                                "reason": str(proposal.get("reason", ""))[:200]})
                break

            if spec.name == "finish":
                attempts = self._run_finish_suggestions(
                    spec, best["params"], suggestions, round_no)
            else:
                attempts = self.optimizer.run_suggestions(
                    self.worker, spec, best["params"], suggestions,
                    round_no, in_odb, cfg.parallel_eval)
            history.extend(attempts)
            for a in attempts:
                self.context.record(
                    spec.name, "attempt",
                    {"tag": a["tag"], "changed": a.get("changed"),
                     "rationale": a.get("rationale", ""),
                     "ok": a["ok"], "metrics": a["metrics"]})

            cand = min(history,
                       key=lambda h: score_metrics(
                           h["metrics"], crashed=not h["ok"],
                           priority=cfg.goal_priority))
            for h in history:
                h["best"] = h is cand
            if cand is not best:
                best = cand
                promote(best, canonical, log=self.log)
            self.context.update_stage(spec.name, metrics=best["metrics"],
                                      params=best["params"])
            self.context.set_last_action(
                spec.name,
                {"round": round_no,
                 "changed": [s["params"] for s in suggestions],
                 "selected": best["tag"]})

            dbg = self.debugger.check(
                spec, params=best["params"], m=best["metrics"],
                res=best["res"],
                context_summary=self.context.summary_for_prompt(),
                attempt_no=1 + len(history),
                odb=Path(best["odb"]), tag=best["tag"])
            debug_budget -= 1
            self.context.set_issues(spec.name, dbg.get("issues", []))

        if not best["ok"]:
            status = "failed"
        elif dbg.get("verdict") == "pass":
            status = "done"
        else:
            status = "done_with_issues"
        self.context.update_stage(spec.name, status=status,
                                  metrics=best["metrics"],
                                  params=best["params"])
        self.log(f"  [stage {spec.name}] -> {status}")
        return best

    # ------------------------------------------------------------------
    def _run_finish_suggestions(self, spec, base_params: dict,
                                suggestions: list[dict],
                                round_no: int) -> list[dict]:
        """Finish-stage suggestions may touch upstream parameters; those
        re-run the flow from the owning stage (reusing the previously
        successful scripts) before re-running the finish checks."""
        cfg = self.cfg
        attempts = []
        for k, s in enumerate(suggestions):
            upstream = {p: v for p, v in s["params"].items()
                        if p in PARAM_OWNER}
            local = {p: v for p, v in s["params"].items()
                     if p not in PARAM_OWNER}
            tag = f"finish_opt{round_no}_{k}"
            in_odb = cfg.save_dir / ALL_STAGES["route"].output_odb
            ok_chain = True
            if upstream:
                start = min((STAGE_ORDER.index(PARAM_OWNER[p])
                             for p in upstream), default=len(STAGE_ORDER))
                chain = [n for n in STAGE_ORDER[start:] if n in
                         ("detail_place", "cts", "route")]
                self.log(f"    [finish] suggestion {k}: rerun {chain} "
                         f"with {upstream}")
                prev_odb = None
                for name in chain:
                    up_spec = ALL_STAGES[name]
                    up_params = dict(
                        self.context.stage_record(name).get("params") or
                        up_spec.default_params)
                    up_params.update({p: v for p, v in upstream.items()
                                      if PARAM_OWNER[p] == name})
                    if prev_odb is None and up_spec.input_stage:
                        prev_odb = cfg.save_dir / \
                            ALL_STAGES[up_spec.input_stage].output_odb
                    out_odb = cfg.save_dir / f"{tag}_{name}.odb"
                    res, m_up = self.worker.run(
                        up_spec, up_params, f"{tag}_{name}",
                        in_odb=prev_odb, out_odb=out_odb)
                    if not res.ok:
                        ok_chain = False
                        break
                    prev_odb = out_odb
                if ok_chain and prev_odb is not None:
                    in_odb = prev_odb
            p = dict(base_params)
            p.update(local)
            out_odb = cfg.save_dir / f"{tag}.odb"
            if ok_chain:
                res, m = self.worker.run(spec, p, tag, in_odb=in_odb,
                                         out_odb=out_odb)
                m.update(self._parse_ir_reports(tag))
                # carry DRC count from the rerun route if present
                drc = M.count_drc_violations(
                    cfg.report_dir / f"{tag}_route.drc.rpt")
                if drc is not None:
                    m["drc_violations"] = drc
                ok = res.ok
            else:
                res, m, ok = None, {"crashed": True}, False
            attempts.append({"tag": tag, "params": p,
                             "changed": s["params"],
                             "rationale": s["rationale"], "ok": ok,
                             "metrics": m, "odb": str(out_odb),
                             "res": res or _NullRes()})
        return attempts

    # ------------------------------------------------------------------
    def _parse_ir_reports(self, tag: str) -> dict:
        """Compute worst IR drop from the psm voltage files."""
        out: dict = {}
        rd = self.cfg.report_dir
        vdd_nom = float(ALL_STAGES["finish"].default_params.get(
            "vdd_voltage", 1.1))
        vf = rd / f"{tag}.vdd_voltage.rpt"
        if vf.exists():
            vs = _voltages(vf)
            if vs:
                out["ir_vdd_min_v"] = min(vs)
                out["ir_worst_drop_v"] = round(vdd_nom - min(vs), 6)
        vf = rd / f"{tag}.vss_voltage.rpt"
        if vf.exists():
            vs = _voltages(vf)
            if vs:
                out["ir_vss_max_v"] = max(vs)
        return out

    # ==================================================================
    # final report
    # ==================================================================
    def write_final_report(self) -> dict:
        cfg = self.cfg
        stages_rec = self.context.data.get("stages", {})
        final: dict = {}
        for name in ("route", "finish"):
            final.update(stages_rec.get(name, {}).get("metrics", {}))
        final.update(self._parse_ir_reports("finish"))
        ok, fails = goal_check(final, cfg.goal)
        tot = self.telemetry.totals()
        payload = {
            "design": cfg.design,
            "goal": cfg.goal,
            "goal_met": ok,
            "goal_fails": fails,
            "final_metrics": final,
            "per_stage": {n: {"status": r.get("status"),
                              "metrics": r.get("metrics"),
                              "params": r.get("params")}
                          for n, r in stages_rec.items()},
            "usage": {
                "prompt_tokens": tot["prompt_tokens"],
                "completion_tokens": tot["completion_tokens"],
                "llm_calls": tot["llm_calls"],
                "model_time_s": round(tot["model_time_s"], 1),
                "tool_time_s": round(tot["tool_time_s"], 1),
                "wall_time_s": round(tot["wall_time_s"], 1),
            },
        }
        (cfg.output_dir / "final_metrics.json").write_text(
            json.dumps(payload, indent=2))

        lines = [f"# OpenLayout final report — {cfg.design}", "",
                 f"goal: `{json.dumps(cfg.goal)}`",
                 f"**goal met: {ok}**" +
                 (f" — fails: {'; '.join(fails)}" if fails else ""), "",
                 "## Final metrics", ""]
        for k in sorted(final):
            lines.append(f"- {k}: {final[k]}")
        lines += ["", "## Stages", ""]
        for n, r in stages_rec.items():
            lines.append(f"### {n} — {r.get('status')}")
            lines.append(f"- params: `{json.dumps(r.get('params', {}))}`")
            lines.append(f"- metrics: `{json.dumps(r.get('metrics', {}))}`")
            if r.get("key_issues"):
                lines.append("- issues: "
                             + "; ".join(map(str, r["key_issues"])))
        u = payload["usage"]
        lines += ["", "## Usage", "",
                  f"- LLM calls: {u['llm_calls']}, tokens: "
                  f"{u['prompt_tokens']} prompt + {u['completion_tokens']} "
                  f"completion",
                  f"- model runtime: {u['model_time_s']}s, "
                  f"tool runtime: {u['tool_time_s']}s, "
                  f"wall: {u['wall_time_s']}s"]
        (cfg.output_dir / "report.md").write_text("\n".join(lines) + "\n")
        self.log(f"[flow] final report -> {cfg.output_dir}/report.md  "
                 f"goal_met={ok}")
        return payload


_MACRO_GEOMETRY_TCL = r"""read_db @ODB@
set _blk [ord::get_db_block]
set _dbu [[ord::get_db_tech] getDbUnitsPerMicron]
set _core [$_blk getCoreArea]
puts "OL_FACT core_bbox_um [format %.3f [expr {[$_core xMin]/double($_dbu)}]] [format %.3f [expr {[$_core yMin]/double($_dbu)}]] [format %.3f [expr {[$_core xMax]/double($_dbu)}]] [format %.3f [expr {[$_core yMax]/double($_dbu)}]]"
foreach _i [$_blk getInsts] {
    set _m [$_i getMaster]
    if {[$_m isBlock]} {
        puts "OL_MACRO [$_i getName] [format %.3f [expr {[$_m getWidth]/double($_dbu)}]] [format %.3f [expr {[$_m getHeight]/double($_dbu)}]]"
    }
}
"""


class _NullRes:
    ok = False
    seconds = 0.0
    log_text = ""
    log_path = Path("/dev/null")


def _voltages(path: Path) -> list[float]:
    """Tolerant psm voltage-file parse: last float per line, plausible volts."""
    vs: list[float] = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            nums = re.findall(r"-?\d+\.\d+(?:e[+-]?\d+)?", line)
            if nums:
                v = float(nums[-1])
                if -0.5 <= v <= 2.5:
                    vs.append(v)
    except OSError:
        pass
    return vs
