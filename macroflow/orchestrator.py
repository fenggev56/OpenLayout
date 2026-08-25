"""Orchestrator — coordinates Planner / Placer / Checker agents across the
skill-driven macro-placement flow.

Control flow (the validated skeleton; the *decisions* inside it are the LLM's):

    init                                   (Python: read/floorplan)
    Planner.strategy                       (LLM: strategy from image + pins)
    Placer.place                           (LLM: coordinates or region counts)
    Gate-A loop                            (LLM checker; Planner revises on fail;
                                             continue with latest placement on limit)
    baseline fast-place                    (Python: Gate-B proxy image)
    Gate-B loop (<= max_refine):
        Checker Gate-B -> proceed?         (LLM: judge density/tightness)
        Planner.strategy                   (LLM: revise shape/regions on fail)
        Placer.place                       (LLM: re-place from revised strategy)
        stop on checker-proceed or try limit, keeping latest placement
    detailed placement                     (Python: skill template)
    CTS + route                            (Python: skill template)
    save candidate                         (coords / density / timing / images / ODBs)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from pathlib import Path

from .agents import PlacementIncomplete, Specialists
from .candidates import MacroCandidateStore
from .config import FlowConfig
from .context import FlowContext
from .operations import FlowOps
from .skills import SkillLibrary
from .snapshots import SnapshotRegistry


_STOP_ORDER = ["placement", "gateb", "route"]

DESIGN_TNS_GOALS = {
    "bp_fe": -15.0,
    "bp_be": -20.0,
    "VeriGPU": -10.0,
    "verigpu": -10.0,
    "swerv_wrapper": -300.0,
    "serv_wrapper": -300.0,
    "ariane133": -1.0,
    "isa_npu": -1.0,
}


class _NoOpVLM:
    """Placeholder VLM for deterministic workers that never call agents."""

    def chat(self, *args, **kwargs):
        raise RuntimeError("VLM is disabled for deterministic post-route refinement")


def _visible_gpu_slots() -> list[str]:
    """Return host GPU IDs that workers can be pinned to.

    If CUDA_VISIBLE_DEVICES is already constrained, preserve those logical slots.
    Otherwise ask PyTorch for the visible device count without loading the model.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        slots = [s.strip() for s in env.split(",") if s.strip()]
        if slots:
            return slots
    try:
        import torch
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    return [str(i) for i in range(n)]


def _run_parallel_candidate_worker(payload: dict) -> dict:
    """Run one macro candidate in an isolated process/work directory.

    Each worker owns its own VLM, OpenROAD session, generated scripts and
    artifacts. The parent process later copies the candidate audit files into
    the main candidate ledger.
    """
    start = time.time()
    try:
        if payload.get("mode") == "post_route_refine":
            return _run_post_route_refine_worker(payload, start)
        gpu = payload.get("cuda_visible_devices")
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        cfg = FlowConfig(data_dir=payload["data_dir"],
                         design=payload["design"],
                         top=payload["top"],
                         model_config=payload["model_config"],
                         work_dir=payload["work_dir"],
                         route_mode=payload.get("route_mode"),
                         floorplan_def=payload.get("floorplan_def",
                                                   "def/2_1_floorplan.def"))
        orch = MacroFlowOrchestrator(
            cfg,
            max_refine=payload["gate_b_max_refine"],
            verbose=payload["verbose"],
            stop_after=payload["stop_after"],
            gui=False,
            gate_a_max_refine=payload["gate_a_max_refine"],
            macro_candidates=1,
            parallel_workers=1,
            candidate_offset=payload["candidate_index"],
            candidate_total=payload["candidate_total"],
            initial_strategies=[payload.get("strategy", "")],
            post_route_opt_loops=0,
        )
        metrics = orch.run()
        telemetry = orch.ctx.telemetry.report() if orch.ctx is not None else {}
        candidates = []
        for c in orch.macro_store.all() if orch.macro_store else []:
            candidates.append({
                "strategy": c.strategy,
                "coords": c.coords,
                "metrics": c.metrics,
                "gate_a": c.gate_a,
                "gate_b": c.gate_b,
                "images": c.images,
                "odbs": c.odbs,
                "worker_candidate_id": c.id,
                "worker_dir": c.dir,
            })
        return {
            "ok": True,
            "candidate_index": payload["candidate_index"],
            "work_dir": payload["work_dir"],
            "metrics": metrics,
            "telemetry": telemetry,
            "candidates": candidates,
            "elapsed": time.time() - start,
        }
    except Exception as exc:
        return {
            "ok": False,
            "candidate_index": payload.get("candidate_index"),
            "work_dir": payload.get("work_dir"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed": time.time() - start,
        }


def _run_post_route_refine_worker(payload: dict, start: float | None = None) -> dict:
    """Run one deterministic post-route refinement candidate.

    The worker does not load Qwen. It initializes OpenROAD, adopts the current
    global-best macro coordinates from the parent, applies one local geometric
    variant, then runs detailed placement + CTS/route for routed feedback.
    """
    start = start or time.time()
    try:
        cfg = FlowConfig(data_dir=payload["data_dir"],
                         design=payload["design"],
                         top=payload["top"],
                         model_config=payload["model_config"],
                         work_dir=payload["work_dir"],
                         route_mode=payload.get("route_mode"),
                         floorplan_def=payload.get("floorplan_def",
                                                   "def/2_1_floorplan.def"))
        orch = MacroFlowOrchestrator(
            cfg,
            max_refine=0,
            verbose=payload["verbose"],
            stop_after=None,
            gui=False,
            gate_a_max_refine=0,
            macro_candidates=1,
            parallel_workers=1,
            candidate_offset=payload["candidate_index"],
            candidate_total=payload["candidate_total"],
            post_route_opt_loops=0,
        )
        orch.setup(vlm=_NoOpVLM())
        try:
            metrics = orch.run_post_route_refinement_candidate(
                base_coords=payload["base_coords"],
                loop_index=payload["loop_index"],
                variant_index=payload["variant_index"],
                total_variants=payload["candidate_total"],
                parent_candidate_id=payload["parent_candidate_id"],
            )
            telemetry = orch.ctx.telemetry.report() if orch.ctx is not None else {}
            candidates = []
            for c in orch.macro_store.all() if orch.macro_store else []:
                candidates.append({
                    "strategy": c.strategy,
                    "coords": c.coords,
                    "metrics": c.metrics,
                    "gate_a": c.gate_a,
                    "gate_b": c.gate_b,
                    "images": c.images,
                    "odbs": c.odbs,
                    "worker_candidate_id": c.id,
                    "worker_dir": c.dir,
                })
        finally:
            try:
                orch.ctx.session.close()
            except Exception:
                pass
        return {
            "ok": True,
            "mode": "post_route_refine",
            "loop_index": payload["loop_index"],
            "candidate_index": payload["candidate_index"],
            "variant_index": payload["variant_index"],
            "parent_candidate_id": payload["parent_candidate_id"],
            "work_dir": payload["work_dir"],
            "metrics": metrics,
            "telemetry": telemetry,
            "candidates": candidates,
            "elapsed": time.time() - start,
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "post_route_refine",
            "loop_index": payload.get("loop_index"),
            "candidate_index": payload.get("candidate_index"),
            "variant_index": payload.get("variant_index"),
            "parent_candidate_id": payload.get("parent_candidate_id"),
            "work_dir": payload.get("work_dir"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed": time.time() - start,
        }


class MacroFlowOrchestrator:
    def __init__(self, cfg: FlowConfig, max_refine: int | None = None,
                 verbose: bool = True,
                 stop_after: str | None = None, gui: bool = False,
                 goal_file: str | None = None, max_tries: int | None = None,
                 gate_a_max_refine: int | None = None,
                 macro_candidates: int | None = None,
                 parallel_workers: int | None = None,
                 candidate_offset: int = 0,
                 candidate_total: int | None = None,
                 initial_strategies: list[str] | None = None,
                 post_route_opt_loops: int | None = None,
                 optimizer_variants: int | None = None):
        self.cfg = cfg
        self.max_refine = max_refine if max_refine is not None else cfg.gate_b_max_refine
        self.gate_a_max_refine = gate_a_max_refine if gate_a_max_refine is not None \
            else cfg.gate_a_max_refine
        self.macro_candidates = macro_candidates if macro_candidates is not None \
            else cfg.macro_candidates
        self.parallel_workers = parallel_workers if parallel_workers is not None \
            else cfg.parallel_workers
        self.post_route_opt_loops = post_route_opt_loops if post_route_opt_loops is not None \
            else cfg.post_route_opt_loops
        self.optimizer_variants = optimizer_variants if optimizer_variants is not None \
            else cfg.optimizer_variants
        self.verbose = verbose
        self.stop_after = stop_after          # placement|gateb|detail|route|None
        self.gui = gui                        # render std-cell placement live (--gui)
        self.goal_file = goal_file            # --goal-file (else <data_dir>/goal.yaml ...)
        self.max_tries = max_tries            # --max-tries (else goal.max_tries)
        self.candidate_offset = max(0, int(candidate_offset))
        self.candidate_total = candidate_total
        self.initial_strategies = list(initial_strategies or [])
        self.ctx: FlowContext | None = None
        self.ops: FlowOps | None = None
        self.specialists: Specialists | None = None
        self.macro_store: MacroCandidateStore | None = None
        self.parallel_results: list[dict] = []

    def _stop(self, point: str) -> bool:
        if not self.stop_after or self.stop_after not in _STOP_ORDER:
            return False
        if _STOP_ORDER.index(point) >= _STOP_ORDER.index(self.stop_after):
            self.ctx.note(f"[stop-after={self.stop_after}] halting at '{point}'")
            return True
        return False

    # ------------------------------------------------------------------
    def _banner(self, msg: str):
        if self.verbose:
            print(f"\n{'=' * 72}\n  {msg}\n{'=' * 72}", flush=True)

    def setup(self, vlm=None, session=None):
        """Load the VLM + start OpenROAD (or accept injected fakes for testing)."""
        self._banner(f"MACROFLOW — multi-agent macro placement ({self.cfg.design})")
        injected_runtime = vlm is not None or session is not None
        if vlm is None:
            from .llm import build_vlm
            print("[setup] loading VLM ...", flush=True)
            vlm = build_vlm(self.cfg.model_config_path)
        if session is None:
            from .openroad import ORSession
            session = ORSession(
                self.cfg.session_dir,
                save_image_width=self.cfg.save_image_width,
                save_image_resolution=self.cfg.save_image_resolution,
            )
            print("[setup] starting OpenROAD ...", flush=True)
            session.start()
        registry = SnapshotRegistry()
        skills = SkillLibrary()
        self.ctx = FlowContext(self.cfg, session, vlm, skills, registry)
        self.ctx.gui = self.gui
        self.ops = FlowOps(self.ctx)
        self.specialists = Specialists(self.ctx, self.ops, verbose=self.verbose)

        self.macro_store = MacroCandidateStore(self.cfg.candidates_dir)
        self.ctx.note("[macro-loop] "
                      f"candidates={self.macro_candidates}, "
                      f"GateA tries={self.gate_a_max_refine}, "
                      f"GateB tries={self.max_refine}, "
                      f"parallel_workers={self.parallel_workers}, "
                      f"post_route_loops={self.post_route_opt_loops}, "
                      f"route_mode={self.cfg.route_mode_effective}")
        if self.parallel_workers > 1 and injected_runtime:
            self.ctx.note("[macro-loop] injected runtime detected; using the "
                          "single-session path despite parallel_workers > 1.")

    # ------------------------------------------------------------------
    def run(self, vlm=None, session=None) -> dict:
        if self._use_parallel_runtime(vlm, session):
            t0 = time.time()
            self._run_parallel_flow()
            metrics = self._finalize()
            self._save_parallel_summary(elapsed=time.time() - t0)
            return metrics

        self.setup(vlm=vlm, session=session)
        t0 = time.time()
        try:
            self._run_flow()
            if self.post_route_opt_loops > 0:
                try:
                    self.ctx.session.close()
                except Exception:
                    pass
                workers = min(self.parallel_workers, self.macro_candidates)
                self._run_post_route_optimization_loops(_visible_gpu_slots(), workers)
        finally:
            metrics = self._finalize()
            self._save_summary(elapsed=time.time() - t0)
            try:
                self.ctx.session.close()
            except Exception:
                pass
        return metrics

    def _use_parallel_runtime(self, vlm, session) -> bool:
        return (self.parallel_workers > 1 and self.macro_candidates > 1
                and vlm is None and session is None)

    def _run_parallel_flow(self):
        """Process macro candidates concurrently in isolated worker processes."""
        self._banner(f"MACROFLOW — parallel macro candidates ({self.cfg.design})")
        self.macro_store = MacroCandidateStore(self.cfg.candidates_dir)
        workers = min(self.parallel_workers, self.macro_candidates)
        if self.verbose:
            print(f"  [parallel] candidates={self.macro_candidates}, "
                  f"workers={workers}, GateA={self.gate_a_max_refine}, "
                  f"GateB={self.max_refine}, "
                  f"post-route loops={self.post_route_opt_loops}, "
                  f"route_mode={self.cfg.route_mode_effective}", flush=True)

        strategies = self._plan_parallel_candidate_strategies()
        gpu_slots = _visible_gpu_slots()
        if self.verbose and gpu_slots:
            print(f"  [parallel] worker GPU assignment: "
                  f"{gpu_slots[:workers]}", flush=True)

        worker_root = self.cfg.output_root / "candidate_workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        payloads = []
        for i in range(self.macro_candidates):
            gpu = gpu_slots[i % len(gpu_slots)] if gpu_slots else None
            payloads.append({
                "candidate_index": i,
                "candidate_total": self.macro_candidates,
                "strategy": strategies[i],
                "cuda_visible_devices": gpu,
                "data_dir": str(self.cfg.data_dir),
                "design": self.cfg.design,
                "top": self.cfg.top,
                "model_config": self.cfg.model_config,
                "floorplan_def": self.cfg.floorplan_def,
                "route_mode": self.cfg.route_mode_effective,
                "work_dir": str(worker_root / f"cand_{i:02d}"),
                "gate_a_max_refine": self.gate_a_max_refine,
                "gate_b_max_refine": self.max_refine,
                "stop_after": self.stop_after,
                "verbose": self.verbose,
            })

        results: list[dict] = []
        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn_ctx) as ex:
            future_map = {ex.submit(_run_parallel_candidate_worker, p): p
                          for p in payloads}
            for fut in as_completed(future_map):
                p = future_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "candidate_index": p["candidate_index"],
                        "work_dir": p["work_dir"],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                results.append(result)
                status = "done" if result.get("ok") else "failed"
                if self.verbose:
                    print(f"  [parallel] candidate {p['candidate_index'] + 1}/"
                          f"{self.macro_candidates} {status} "
                          f"({result.get('elapsed', 0):.0f}s)", flush=True)

        self.parallel_results = sorted(results, key=lambda r: r.get("candidate_index", -1))
        for result in self.parallel_results:
            if result.get("ok"):
                self._merge_parallel_result(result)
            else:
                self._record_parallel_failure(result)

        self._run_post_route_optimization_loops(gpu_slots, workers)
        self._prune_parallel_worker_odbs()

    def _best_images(self, best) -> list[str]:
        """Image paths for the VLM optimizer: pin-side (most informative for
        placement strategy) first, then placement/routed snapshots."""
        imgs: list[str] = []
        try:
            pin = sorted(self.cfg.image_dir.glob(
                f"macro_pin_sides/{best.tag}_*pin_sides.png"))
            if pin:
                imgs.append(str(pin[-1]))
        except Exception:
            pass
        if isinstance(best.images, dict):
            for key in ("macro", "routed", "detail"):
                p = best.images.get(key)
                if p and Path(p).is_file() and str(p) not in imgs:
                    imgs.append(str(p))
        return imgs[:2]

    @staticmethod
    def _parse_numbered_strategies(text: str, n: int) -> list[str]:
        """Split a numbered VLM reply ("1. ... 2. ...") into strategy strings."""
        import re
        if not text:
            return []
        parts = re.split(r"(?mi)^\s*(?:strategy|variant|option)?\s*\d+\s*[.):\-]\s+",
                         text)
        parts = [" ".join(p.split()) for p in parts if p and p.strip()]
        # Drop a leading preamble if the model produced one before "1.".
        if len(parts) > n:
            parts = parts[-n:]
        return [p[:700] for p in parts if len(p) > 20][:n]

    def _plan_optimizer_strategies(self, best, loop_index: int, n: int) -> list[str]:
        """Ask the parent VLM (still loaded after the flow) for ``n`` diverse,
        timing-targeted macro-placement strategies, grounded in the optimizer
        skill + the best candidate's routed image and PPA. Returns [] (→ all
        geometric) if the VLM is unavailable or yields nothing."""
        if n <= 0 or self.ctx is None or getattr(self.ctx, "vlm", None) is None:
            return []
        try:
            skill = self.ctx.skills.skill_text("optimizer")
        except Exception:
            skill = ""
        routed = best.metrics.get("routed", {}) if isinstance(best.metrics, dict) else {}
        wns, tns, drc = routed.get("wns"), routed.get("tns"), routed.get("drc")
        n_macros = len(best.coords) if best.coords else "?"
        images = self._best_images(best)
        text = (
            f"Design {self.cfg.design}: {n_macros} macros. The current best macro "
            f"placement routed with post-route timing WNS={wns} ns, TNS={tns} ns, "
            f"DRC={drc}. Attached image(s): the placement with each macro's pin "
            f"side drawn as a red bar, and/or the routed result.\n\n"
            f"Propose exactly {n} DISTINCT macro-placement strategies that should "
            f"reduce TNS (shorten critical paths) while staying routable and "
            f"DRC-clean. Make them genuinely different from each other and from "
            f"the current layout — e.g. spread macros across three sides leaving "
            f"the pin side clear, cluster same-cluster / timing-critical macros "
            f"adjacent, widen channels that force detours, avoid corner channels "
            f"and deep stacks (>3). Each strategy is ONE short paragraph naming "
            f"the sides/topology. Number them 1. to {n}."
        )
        content = [{"type": "image", "image": f"file://{p}"} for p in images]
        content.append({"type": "text", "text": text})
        try:
            resp = self.ctx.vlm.chat(
                [{"role": "user", "content": content}],
                system=skill or "You optimize macro placement for timing/PPA.",
                max_tokens=2048)
        except Exception as exc:
            if self.verbose:
                print(f"  [post-route] VLM strategy planning failed ({exc}); "
                      "geometric variants only", flush=True)
            return []
        strategies = self._parse_numbered_strategies(getattr(resp, "text", ""), n)
        try:
            out = self.cfg.stage_model_dir("optimizer") / f"loop_{loop_index:02d}_strategies.json"
            out.write_text(json.dumps(strategies, indent=2))
        except Exception:
            pass
        return strategies

    def _run_post_route_optimization_loops(self, gpu_slots: list[str],
                                           workers: int) -> None:
        """After the first routed batch, iteratively refine the current best.

        Each loop selects the global best candidate so far, creates
        ``macro_candidates`` deterministic local macro variants from its saved
        coordinates, runs detail/route on those variants in parallel, merges the
        results, and then repeats with the new global best.
        """
        if self.post_route_opt_loops <= 0:
            return
        if self.stop_after in {"placement", "gateb"}:
            if self.verbose:
                print("  [post-route] skipped because stop-after is before route",
                      flush=True)
            return
        if self.macro_store is None:
            return

        worker_root = self.cfg.output_root / "post_route_workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        for loop_index in range(1, self.post_route_opt_loops + 1):
            best = self.macro_store.best()
            if best is None or not best.coords:
                if self.verbose:
                    print("  [post-route] no routed best coordinates available; "
                          "stopping optimization loop", flush=True)
                break
            if self._best_meets_tns_goal(best):
                if self.verbose:
                    routed = best.metrics.get("routed", {})
                    print("  [post-route] current best already meets TNS goal; "
                          f"skipping remaining loops: #{best.id} {routed}",
                          flush=True)
                break
            before_best_id = best.id
            self._banner("Post-route optimization loop "
                         f"{loop_index}/{self.post_route_opt_loops} — "
                         f"refine best candidate #{best.id}")
            n_var = max(1, self.optimizer_variants)
            # VLM-guided strategy variants (genuinely different topologies aimed
            # at timing) + geometric local-search variants (gentle moves around
            # the current best). Reserve >=1 geometric for robustness; if the VLM
            # is unavailable or returns nothing, the loop is all-geometric.
            n_strat_target = n_var - 1 if n_var > 1 else 0
            strategies = self._plan_optimizer_strategies(best, loop_index,
                                                         n_strat_target)
            n_geo = n_var - len(strategies)
            if self.verbose:
                print(f"  [post-route] loop {loop_index}: "
                      f"{len(strategies)} VLM-strategy + {n_geo} geometric "
                      f"variant(s)", flush=True)
            payloads = []
            idx = 0
            for strat in strategies:
                gpu = gpu_slots[idx % len(gpu_slots)] if gpu_slots else None
                payloads.append({
                    "candidate_index": idx,
                    "candidate_total": n_var,
                    "strategy": strat,
                    "cuda_visible_devices": gpu,
                    "data_dir": str(self.cfg.data_dir),
                    "design": self.cfg.design,
                    "top": self.cfg.top,
                    "model_config": self.cfg.model_config,
                    "floorplan_def": self.cfg.floorplan_def,
                    "route_mode": self.cfg.route_mode_effective,
                    "work_dir": str(worker_root /
                                    f"loop_{loop_index:02d}" / f"cand_{idx:02d}"),
                    "gate_a_max_refine": self.gate_a_max_refine,
                    "gate_b_max_refine": self.max_refine,
                    "stop_after": self.stop_after,
                    "verbose": self.verbose,
                })
                idx += 1
            for g in range(n_geo):
                gpu = gpu_slots[idx % len(gpu_slots)] if gpu_slots else None
                payloads.append({
                    "mode": "post_route_refine",
                    "loop_index": loop_index,
                    "variant_index": g,
                    "candidate_index": idx,
                    "candidate_total": n_var,
                    "parent_candidate_id": best.id,
                    "base_coords": best.coords,
                    "cuda_visible_devices": gpu,
                    "data_dir": str(self.cfg.data_dir),
                    "design": self.cfg.design,
                    "top": self.cfg.top,
                    "model_config": self.cfg.model_config,
                    "floorplan_def": self.cfg.floorplan_def,
                    "route_mode": self.cfg.route_mode_effective,
                    "work_dir": str(worker_root /
                                    f"loop_{loop_index:02d}" / f"cand_{idx:02d}"),
                    "verbose": self.verbose,
                })
                idx += 1

            # Strategy variants each load the VLM, so cap concurrency by the
            # number of visible GPU slots (geometric variants are CPU/OpenROAD).
            opt_workers = max(1, min(n_var, len(gpu_slots) or 1, max(workers, len(gpu_slots) or 1)))
            results: list[dict] = []
            spawn_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=opt_workers, mp_context=spawn_ctx) as ex:
                future_map = {ex.submit(_run_parallel_candidate_worker, p): p
                              for p in payloads}
                for fut in as_completed(future_map):
                    p = future_map[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {
                            "ok": False,
                            "mode": "post_route_refine",
                            "loop_index": loop_index,
                            "candidate_index": p["candidate_index"],
                            "variant_index": p["variant_index"],
                            "parent_candidate_id": p["parent_candidate_id"],
                            "work_dir": p["work_dir"],
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    results.append(result)
                    status = "done" if result.get("ok") else "failed"
                    if self.verbose:
                        print("  [post-route] "
                              f"loop {loop_index}/{self.post_route_opt_loops} "
                              f"variant {p['candidate_index'] + 1}/"
                              f"{n_var} {status} "
                              f"({result.get('elapsed', 0):.0f}s)", flush=True)

            sorted_results = sorted(results,
                                    key=lambda r: r.get("candidate_index", -1))
            self.parallel_results.extend(sorted_results)
            for result in sorted_results:
                if result.get("ok"):
                    self._merge_parallel_result(result)
                else:
                    self._record_parallel_failure(result)
            best_after = self.macro_store.best()
            if self.verbose and best_after is not None:
                routed = best_after.metrics.get("routed", {})
                print("  [post-route] current best after loop "
                      f"{loop_index}: #{best_after.id} {routed}", flush=True)
            if best_after is not None and self._best_meets_tns_goal(best_after):
                if self.verbose:
                    print("  [post-route] TNS goal met; stopping PPA loop",
                          flush=True)
                break
            if best_after is not None and best_after.id == before_best_id:
                if self.verbose:
                    print("  [post-route] no PPA improvement in loop; "
                          "stopping optimization loop", flush=True)
                break

    def _best_meets_tns_goal(self, cand) -> bool:
        target = DESIGN_TNS_GOALS.get(self.cfg.design)
        if target is None:
            return False
        routed = cand.metrics.get("routed") or {}
        if not isinstance(routed, dict):
            routed = {}
        try:
            return float(routed.get("tns")) >= float(target)
        except (TypeError, ValueError):
            return False

    def _plan_parallel_candidate_strategies(self) -> list[str]:
        """Generate all candidate macro-placement strategies before fanout.

        The parent process owns this planning pass so ``macro_candidates=n`` means
        isolated workers start from n distinct shapes/topologies, then each worker
        runs its own place/check/refine trajectory in parallel.
        """
        self._banner("Stage 0/1 — parent seeds distinct candidate shapes")
        strategies = [
            self._parallel_shape_strategy(i)
            for i in range(self.macro_candidates)
        ]
        if self.verbose:
            for i, strategy in enumerate(strategies):
                print(f"  [Planner:parallel cand{i:02d}] seeded strategy: "
                      f"{strategy[:220]}", flush=True)

        out = self.cfg.stage_model_dir("planning") / "candidate_strategies.json"
        out.write_text(json.dumps([
            {"candidate_index": i, "strategy": s}
            for i, s in enumerate(strategies)
        ], indent=2))
        return strategies

    @staticmethod
    def _parallel_shape_strategy(index: int) -> str:
        """Deterministic initial topology families for parallel exploration.

        LLM-only diversity tended to collapse into repeated top-edge C-shapes.
        These seeds give each worker a genuinely different starting outline while
        leaving Gate-B refinement free to optimize the trajectory.
        """
        families = [
            ("TOP_BAND", "top edge"),
            ("RIGHT_BAND", "right edge"),
            ("BOTTOM_BAND", "bottom edge"),
            ("LEFT_BAND", "left edge"),
            ("TOP_RIGHT_L", "top and right edges"),
            ("BOTTOM_RIGHT_L", "bottom and right edges"),
            ("BOTTOM_LEFT_L", "bottom and left edges"),
            ("TOP_LEFT_L", "top and left edges"),
            ("RING", "top, right, bottom, and left edges"),
            ("VERTICAL_SPLIT", "left and right edges"),
        ]
        name, edges = families[index % len(families)]
        return (
            f"Required shape family {name}: place macro clusters in compact "
            f"side bands along the {edges}. Multiple legal lanes near those "
            "edges are allowed; do not require a single row or a literal C-notch. "
            "Use R0 orientation unless legalization requires otherwise. Preserve "
            "dataflow locality by keeping macros from the same cluster adjacent, "
            "and keep the standard-cell area contiguous and routable."
        )

    def _merge_parallel_result(self, result: dict):
        for data in result.get("candidates", []):
            cand = self.macro_store.new(strategy=data.get("strategy", ""))
            cand.strategy = data.get("strategy", "")
            metrics = dict(data.get("metrics") or {})
            metrics["worker_candidate_index"] = result.get("candidate_index")
            metrics["worker_elapsed_s"] = round(float(result.get("elapsed", 0.0)), 3)
            is_best = self.macro_store.commit(
                cand,
                data.get("coords") or [],
                metrics,
                data.get("gate_a") or {},
                data.get("gate_b") or {},
                images=data.get("images") or {},
                odbs=data.get("odbs") or {},
            )
            if self.verbose:
                print(f"  [candidate:{cand.tag}] merged from {data.get('worker_dir')}"
                      f"{' <- BEST' if is_best else ''}", flush=True)
            self._export_parallel_candidate_images(cand)

    def _prune_parallel_worker_odbs(self) -> None:
        """Remove worker-local OpenROAD checkpoints after parent merge.

        The parent candidate ledger keeps the winning routed ODB. Worker
        directories are only staging areas, so their ODBs duplicate data and can
        dominate disk usage on multi-candidate runs.
        """
        removed = 0
        for dirname in ("candidate_workers", "post_route_workers"):
            root = self.cfg.output_root / dirname
            if not root.exists():
                continue
            for path in root.rglob("*.odb"):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        if self.verbose and removed:
            print(f"  [cleanup] pruned {removed} worker ODB file(s)", flush=True)

    def _export_parallel_candidate_images(self, cand):
        """Put merged worker images in the parent images tree as well.

        The canonical audit trail is ``candidates/macro_cand_##/{fast,detail,routed}.png``.
        Parallel runs used to leave the parent ``images/`` tree empty, which made
        the usual browsing path confusing. Keep both layouts.
        """
        stage_by_key = {
            "macro": "macro_placed",
            "pin_sides": "macro_pin_sides",
            "fast": "fast_placed",
            "detail": "detail_placed",
            "routed": "routed",
        }
        for key, src in (cand.images or {}).items():
            stage = stage_by_key.get(key)
            if not stage:
                continue
            try:
                src_path = Path(src)
                if not src_path.exists():
                    continue
                dst = self.cfg.stage_image_dir(stage) / f"{cand.tag}_{key}.png"
                shutil.copy2(src_path, dst)
            except OSError:
                pass

    def _record_candidate_exception(self, index: int, exc: Exception):
        """Record a sequential-mode candidate failure the way the parallel path
        records a dead worker, then let the batch continue."""
        out = (self.cfg.stage_model_dir("candidates", "failures") /
               f"candidate_{index}_failed.json")
        try:
            out.write_text(json.dumps(
                {"ok": False, "candidate_index": index, "mode": "sequential",
                 "error": str(exc), "error_type": type(exc).__name__},
                indent=2, default=str))
        except OSError:
            pass
        if self.verbose:
            print(f"  [candidate:{index}] FAILED ({type(exc).__name__}): {exc}",
                  flush=True)
            print(f"  [candidate:{index}] continuing with remaining candidates; "
                  f"details: {out}", flush=True)

    def _record_parallel_failure(self, result: dict):
        prefix = "candidate"
        if result.get("mode") == "post_route_refine":
            prefix = f"post_loop_{int(result.get('loop_index') or 0):02d}_variant"
        out = (self.cfg.stage_model_dir("candidates", "failures") /
               f"{prefix}_{result.get('candidate_index')}_failed.json")
        out.write_text(json.dumps(result, indent=2, default=str))
        if self.verbose:
            print(f"  [parallel] failure details: {out}", flush=True)

    def _run_flow(self):
        ops, sp, ctx = self.ops, self.specialists, self.ctx
        tel = ctx.telemetry

        self._banner("Stage 0 — init")
        with tel.stage("init"):
            ops.init_design()

        self._banner("Stage 1/2 — macro candidate batch")
        with tel.stage("macro_candidates"):
            for i in range(self.macro_candidates):
                try:
                    self._run_macro_candidate(i)
                except PlacementIncomplete as exc:
                    # Pure-model mode: this candidate is failed because the
                    # model never produced coordinates for every macro. That
                    # fails THE CANDIDATE, not the run — the parallel path
                    # already records it as {"ok": False} and carries on, and
                    # letting it propagate here would discard sibling
                    # candidates that were fully model-placed and routed.
                    self._record_candidate_exception(i, exc)
                if self.stop_after == "placement":
                    break
        if self._stop("route"):
            return

    # ------------------------------------------------------------------
    def _finalize(self) -> dict:
        """Final metrics from the best macro candidate.

        When the flow is not stopped at Gate B, candidates include post-route
        timing parsed after detailed placement, CTS and routing.
        """
        best = self.macro_store.best() if self.macro_store else None
        if best is not None:
            return {"best_density": best.metrics.get("best_density"),
                    "best_candidate": best.id,
                    "gate_a": best.gate_a.get("verdict"),
                    "gate_b": best.gate_b.get("verdict"),
                    "routed": best.metrics.get("routed", {})}
        ctx, ops = self.ctx, self.ops
        if ctx is None or ops is None:
            return {}
        if best is None:
            return {"best_density": ctx.state.best_density}
        return {}

    # ------------------------------------------------------------------
    def _reset_candidate_state(self, tag: str):
        st = self.ctx.state
        self.ops.reload_clean()
        st.try_tag = tag
        st.entries = []
        st.last_density = None
        st.best_density = 0.0
        st.best_entries = []
        st.best_detail_db = None
        st.routed_metrics = None
        st.last_macro_db = None
        st.last_fast_db = None
        st.macro_pin_sides = {}
        st.compaction_round = 0
        st.history = []

    def _run_macro_candidate(self, index: int):
        ops, sp, ctx = self.ops, self.specialists, self.ctx
        cand = self.macro_store.new()
        display_index = self.candidate_offset + index
        display_total = self.candidate_total or self.macro_candidates
        self._banner(f"Candidate {display_index + 1}/{display_total} — {cand.tag}")
        self._reset_candidate_state(cand.tag)

        strategy = self.initial_strategies[index] if index < len(self.initial_strategies) else ""
        if not strategy:
            note = (f"Candidate {display_index + 1} of {display_total}. Generate a "
                    "distinct macro-placement shape/topology for this candidate; do "
                    "not merely repeat earlier candidates.")
            strategy = sp.strategy(candidate_note=note, context="init")
        else:
            ctx.note(f"[Planner:{cand.tag}] using pre-planned strategy")
        cand.strategy = strategy
        Path(cand.dir, "strategy.txt").write_text(strategy or "")
        ctx.note(f"[Planner:{cand.tag}] strategy: {strategy[:220]}")
        sp.place(strategy)

        gate_a, strategy = self._gate_a_loop(strategy)
        if self._stop("placement"):
            self._save_macro_candidate(cand, strategy, gate_a, {})
            return

        # New workflow: after Gate A (PASS or refinement budget exhausted), go
        # straight to detailed placement + CTS/route. The pre-route fast-place
        # "Gate B" density proxy was removed; timing/PPA is judged after route
        # (Gate B / PPA check) and drives the optimizer loop downstream.
        if not self._gate_allows_next(gate_a):
            ctx.note(f"[GateA:{cand.tag}] failed after {self.gate_a_max_refine} "
                     "attempt(s); proceeding to backend with latest placement")

        gate_b = {}
        backend = {}
        if not self._stop("gateb"):
            backend = self._run_backend_for_candidate(cand)

        self._save_macro_candidate(cand, strategy, gate_a, gate_b, backend)

    def run_post_route_refinement_candidate(self, base_coords: list,
                                            loop_index: int,
                                            variant_index: int,
                                            total_variants: int,
                                            parent_candidate_id: int | None) -> dict:
        """Worker entry for one post-route local refinement candidate."""
        ops, ctx = self.ops, self.ctx
        self._banner("Post-route refinement candidate "
                     f"{variant_index + 1}/{total_variants} "
                     f"(loop {loop_index}, parent #{parent_candidate_id})")
        with ctx.telemetry.stage("init"):
            ops.init_design()
        cand = self.macro_store.new()
        strategy = (
            f"Post-route local refinement loop {loop_index}, variant "
            f"{variant_index + 1}/{total_variants}, derived from parent "
            f"candidate #{parent_candidate_id}."
        )
        cand.strategy = strategy
        Path(cand.dir, "strategy.txt").write_text(strategy)
        self._reset_candidate_state(cand.tag)
        ops.load_entries(base_coords)
        variant = ops.post_route_refine_variant(
            variant_index=variant_index,
            total_variants=total_variants,
            loop_index=loop_index,
            tag=cand.tag,
        )
        backend = self._run_backend_for_candidate(cand)
        backend["metrics"] = {
            "post_route_loop": loop_index,
            "post_route_variant": variant_index,
            "post_route_parent": parent_candidate_id,
            "post_route_anchor": variant.get("anchor"),
            "post_route_strength": variant.get("strength"),
        }
        gate_a = {
            "verdict": "SKIP",
            "feedback": "post-route local refinement starts from routed best",
            "proceed": True,
        }
        gate_b = {
            "verdict": "SKIP",
            "feedback": "post-route feedback comes from CTS/route metrics",
            "proceed": True,
        }
        self._save_macro_candidate(cand, strategy, gate_a, gate_b, backend)
        return self._finalize()

    def _gate_a_loop(self, strategy: str) -> tuple[dict, str]:
        ops, sp, ctx = self.ops, self.specialists, self.ctx
        verdict = {}
        current_strategy = strategy
        for attempt in range(1, self.gate_a_max_refine + 1):
            ops.highlight_macro_pin_sides(
                tag=f"{ctx.state.try_tag}_gate_a{attempt}")
            verdict = sp.check("Gate_A", "gate_a", strategy=current_strategy)
            ctx.note(f"[Checker:GateA a{attempt}] {verdict['verdict']} "
                     f"proceed={verdict['proceed']} — {verdict['feedback'][:140]}")
            if verdict["proceed"]:
                return verdict, current_strategy
            if attempt >= self.gate_a_max_refine:
                verdict = self._mark_limit_exhausted(verdict, "GateA", attempt)
                ctx.note("[GateA] try limit exhausted — continuing to Gate-B with "
                         "latest macro placement")
                break
            ctx.note("[Placer:GateA] refining current placement to match the "
                     "original strategy; Planner is not re-run")
            sp.refine_gate_a(current_strategy, verdict["feedback"])
        return verdict, current_strategy

    def _gate_b_loop(self, strategy: str) -> tuple[dict, str]:
        ops, sp, ctx, tel = self.ops, self.specialists, self.ctx, self.ctx.telemetry
        sp.baseline_fast()
        ctx.note(f"[GateB] baseline density = {ctx.state.best_density:.4f}")
        verdict = {}
        current_strategy = strategy
        for attempt in range(1, self.max_refine + 1):
            verdict = sp.check("Gate_B", "gate_b")
            ctx.note(f"[Checker:GateB b{attempt}] {verdict['verdict']} "
                     f"proceed={verdict['proceed']} — {verdict['feedback'][:140]}")
            if verdict["proceed"]:
                ctx.note("[GateB] checker says proceed — stopping refinement")
                return verdict, current_strategy
            if attempt >= self.max_refine:
                verdict = self._mark_limit_exhausted(verdict, "GateB", attempt)
                ctx.note("[GateB] try limit exhausted — keeping latest macro "
                         "placement for candidate save/ranking")
                break
            tel.incr("gate_b_rounds")
            before = ctx.state.best_density
            current_strategy = sp.strategy(
                candidate_note=f"Gate-B refinement attempt {attempt + 1}.",
                feedback=verdict["feedback"],
                previous_strategy=current_strategy,
                context="gate_b")
            ctx.note(f"[Planner:GateB] revised strategy: {current_strategy[:220]}")
            sp.replan_from_strategy(current_strategy, verdict["feedback"],
                                    context="gate_b",
                                    fallback_anchor=verdict.get("anchor"),
                                    fallback_strength=verdict.get("strength"))
            with tel.op("fast_place"):
                ops.fast_place(f"r{ctx.state.compaction_round}")
            gain = ctx.state.best_density - before
            ctx.note(f"[GateB] density gain after Planner revision: {gain:+.4f}")
        ctx.note(f"[GateB] best density = {ctx.state.best_density:.4f} "
                 f"after {ctx.state.compaction_round} refinement round(s)")
        return verdict, current_strategy

    def _run_backend_for_candidate(self, cand) -> dict:
        """Run detailed placement, CTS and route for the latest candidate.

        The candidate uses the best fast-placement macro entries if Gate B found
        an improvement; otherwise it uses the latest macro placement. This is the
        timing-bearing backend path, so failures are recorded in candidate
        metrics instead of hiding the candidate.
        """
        ops, ctx, tel = self.ops, self.ctx, self.ctx.telemetry
        tag = cand.tag
        entries = ctx.state.best_entries or ctx.state.entries
        result = {}
        try:
            with tel.stage(f"{tag}_detail_place"):
                detail = ops.detail_place(entries=entries, tag=tag)
            result["detail"] = detail
            if isinstance(detail, dict) and detail.get("density") is not None:
                try:
                    density = float(detail["density"])
                    ctx.state.last_density = density
                    if density > ctx.state.best_density:
                        ctx.state.best_density = density
                except (TypeError, ValueError):
                    pass
            detail_error = detail.get("error") if isinstance(detail, dict) else None
            if detail_error:
                result["error"] = detail_error
                ctx.note(f"[backend:{tag}] detail placement failed: {detail_error}")
                return result

            routed_odb = self.cfg.save_dir / f"{tag}_routed.odb"
            with tel.stage(f"{tag}_cts_route"):
                route = ops.route_finalize(tag=tag, out_odb=routed_odb)
            result["route"] = route
            route_error = route.get("error") if isinstance(route, dict) else None
            if route_error:
                result["error"] = route_error
                ctx.note(f"[backend:{tag}] CTS/route failed: {route_error}")
            return result
        except Exception as exc:
            result["error"] = str(exc)
            ctx.note(f"[backend:{tag}] failed: {exc}")
            return result

    @staticmethod
    def _gate_allows_next(verdict: dict) -> bool:
        """Whether flow control should advance past a gate.

        ``proceed`` remains the checker's quality verdict. ``continue_after_limit``
        records the policy choice to keep going once refinement budget is spent.
        """
        return bool(verdict.get("proceed") or verdict.get("continue_after_limit"))

    @staticmethod
    def _mark_limit_exhausted(verdict: dict, gate: str, attempts: int) -> dict:
        out = dict(verdict or {})
        out.setdefault("verdict", "FAIL")
        out.setdefault("feedback", "")
        out.setdefault("proceed", False)
        out["limit_exhausted"] = True
        out["continue_after_limit"] = True
        out["attempts"] = attempts
        out["advance_reason"] = (
            f"{gate} refinement limit exhausted; using latest macro placement.")
        return out

    def _save_macro_candidate(self, cand, strategy: str, gate_a: dict, gate_b: dict,
                              backend: dict | None = None):
        ctx, st = self.ctx, self.ctx.state
        cand.strategy = strategy
        coords = st.best_entries or st.entries
        overlaps = self.ops.check_overlaps()
        outside_core = self.ops.check_outside_core()
        backend = backend or {}
        route = backend.get("route") if isinstance(backend.get("route"), dict) else {}
        detail = backend.get("detail") if isinstance(backend.get("detail"), dict) else {}
        routed_metrics = route.get("metrics") or st.routed_metrics or {}
        metrics = {
            "density": st.last_density,
            "best_density": st.best_density,
            "overlaps": overlaps,
            "outside_core": outside_core,
            "compaction_rounds": st.compaction_round,
            "routed": routed_metrics,
        }
        if backend.get("error"):
            metrics["backend_error"] = backend["error"]
        if isinstance(backend.get("metrics"), dict):
            metrics.update(backend["metrics"])
        metrics.update({f"routed_{k}": v for k, v in routed_metrics.items()})
        macro_snap = ctx.registry.latest("macro_placed")
        pin_side_snap = ctx.registry.latest("macro_pin_sides")
        fast_snap = ctx.registry.latest("fast_placed")
        detail_snap = ctx.registry.latest("detail_placed")
        routed_snap = ctx.registry.latest("routed")
        images = {
            "macro": macro_snap.path if macro_snap else None,
            "pin_sides": pin_side_snap.path if pin_side_snap else None,
            "fast": fast_snap.path if fast_snap else None,
            "detail": detail_snap.path if detail_snap else detail.get("detail_image"),
            "routed": routed_snap.path if routed_snap else route.get("routed_image"),
        }
        odbs = {
            "macro": st.last_macro_db,
            "fast": st.last_fast_db,
            "detail": detail.get("detail_db") or st.best_detail_db,
            "routed": route.get("routed_odb") or route.get("deliverable"),
        }
        is_best = self.macro_store.commit(cand, coords, metrics, gate_a, gate_b,
                                          images=images, odbs=odbs)
        ctx.note(f"[candidate:{cand.tag}] saved to {cand.dir}"
                 f"{' <- BEST' if is_best else ''}")

    def _save_summary(self, elapsed: float):
        ctx = self.ctx
        if ctx is None:
            return
        st = ctx.state
        tel = ctx.telemetry
        tel.set("macros", st.n_macros)
        tel.set("compaction_rounds", st.compaction_round)
        lines = [
            "=" * 72, f"MACROFLOW SUMMARY — {self.cfg.design}", "=" * 72,
            f"elapsed:        {elapsed:.0f}s",
            f"final stage:    {st.stage}",
            f"macros:         {st.n_macros}",
            f"best density:   {st.best_density:.4f}",
            f"compaction:     {st.compaction_round} round(s)",
            f"post-route loops: {self.post_route_opt_loops}",
            f"route mode:     {self.cfg.route_mode_effective}",
            f"macro candidates: {len(self.macro_store.all()) if self.macro_store else 0}",
            "",
            "compaction history:",
        ]
        for h in st.history:
            lines.append(f"  round {h['round']}: strength={h['strength']} "
                         f"anchor={h['anchor']} density={h['density']}")
        if self.macro_store is not None and self.macro_store.all():
            best = self.macro_store.best()
            lines += ["", self.macro_store.render_table(),
                      "", f"winning macro candidate: "
                          f"#{best.id if best else '?'} -> {best.dir if best else '?'}"]
            if best:
                lines.append(f"winning routed metrics: {best.metrics.get('routed', {})}")
                lines.append(f"winning routed odb: {best.odbs.get('routed', '')}")
        lines += ["", tel.render(), "", "snapshots:"]
        for s in ctx.registry.all():
            lines.append(f"  #{s.seq} [{s.stage}] {s.name} {s.meta or ''}")
        lines += ["", "operation log:"] + [f"  {t}" for t in ctx.transcript]
        run_out = self.cfg.stage_model_dir("run")
        out = run_out / "summary.txt"
        out.write_text("\n".join(lines))
        tel.write(run_out / "telemetry.json")
        if self._should_export_results():
            self._export_output_result(elapsed=elapsed, telemetry=tel.report())
        self._banner("DONE")
        if self.verbose:
            best = self.macro_store.best() if self.macro_store else None
            print(f"  best density: {st.best_density:.4f} | "
                  f"macro candidates: "
                  f"{len(self.macro_store.all()) if self.macro_store else 0}", flush=True)
            if best is not None:
                print(f"  best candidate: {best.dir}", flush=True)
                print(f"  routed metrics: {best.metrics.get('routed', {})}", flush=True)
                if best.odbs.get("routed"):
                    print(f"  routed odb:    {best.odbs['routed']}", flush=True)
            print(f"  summary:      {out}", flush=True)
            print(f"  images:       {self.cfg.image_dir}", flush=True)
            r = tel.report()
            print(f"  telemetry:    runtime {r['runtime_s']}s (llm {r['llm_time_s']}s"
                  f"/system {r['system_time_s']}s), {r['llm_calls']} llm calls, "
                  f"{r['tokens']['total']} tokens -> telemetry.json", flush=True)

    def _save_parallel_summary(self, elapsed: float):
        lines = [
            "=" * 72, f"MACROFLOW SUMMARY — {self.cfg.design}", "=" * 72,
            f"elapsed:          {elapsed:.0f}s",
            f"mode:             parallel macro candidates",
            f"workers:          {min(self.parallel_workers, self.macro_candidates)}",
            f"post-route loops: {self.post_route_opt_loops}",
            f"route mode:       {self.cfg.route_mode_effective}",
            f"macro candidates: {len(self.macro_store.all()) if self.macro_store else 0}",
        ]
        if self.macro_store is not None and self.macro_store.all():
            best = self.macro_store.best()
            lines += ["", self.macro_store.render_table(),
                      "", f"winning macro candidate: "
                          f"#{best.id if best else '?'} -> {best.dir if best else '?'}"]
            if best:
                lines.append(f"winning routed metrics: {best.metrics.get('routed', {})}")
                lines.append(f"winning routed odb: {best.odbs.get('routed', '')}")
        failures = [r for r in self.parallel_results if not r.get("ok")]
        if failures:
            lines += ["", "failed workers:"]
            for r in failures:
                lines.append(f"  candidate {r.get('candidate_index')}: {r.get('error')}")
        lines += ["", "worker outputs:"]
        for r in self.parallel_results:
            if r.get("mode") == "post_route_refine":
                label = (f"post-loop {r.get('loop_index')} "
                         f"variant {r.get('candidate_index')} "
                         f"parent #{r.get('parent_candidate_id')}")
            else:
                label = f"candidate {r.get('candidate_index')}"
            lines.append(f"  {label}: {'ok' if r.get('ok') else 'failed'} "
                         f"-> {r.get('work_dir')}")
        run_out = self.cfg.stage_model_dir("run")
        out = run_out / "summary.txt"
        out.write_text("\n".join(lines))
        (run_out / "telemetry.json").write_text(json.dumps({
            "mode": "parallel_macro_candidates",
            "elapsed_s": round(elapsed, 3),
            "workers": min(self.parallel_workers, self.macro_candidates),
            "post_route_opt_loops": self.post_route_opt_loops,
            "route_mode": self.cfg.route_mode_effective,
            "results": self.parallel_results,
        }, indent=2, default=str))
        self._export_output_result(
            elapsed=elapsed,
            telemetry=self._aggregate_parallel_telemetry(elapsed),
        )
        self._banner("DONE")
        if self.verbose:
            print(f"  summary:      {out}", flush=True)
            print(f"  candidates:   {self.cfg.candidates_dir}", flush=True)

    def _should_export_results(self) -> bool:
        """Only top-level user runs should update project/output.

        Parallel candidate and post-route refinement workers write their own
        worker summaries; the parent run exports the aggregate result.
        """
        parts = set(self.cfg.output_root.parts)
        return "candidate_workers" not in parts and "post_route_workers" not in parts

    def _aggregate_parallel_telemetry(self, elapsed: float) -> dict:
        prompt = completion = calls = 0
        llm_time_s = system_time_s = 0.0
        by_agent: dict[str, dict] = {}
        for result in self.parallel_results:
            tel = result.get("telemetry") or {}
            tokens = tel.get("tokens") or {}
            prompt += int(tokens.get("prompt") or 0)
            completion += int(tokens.get("completion") or 0)
            calls += int(tel.get("llm_calls") or 0)
            # Sum per-worker model-driver time and OpenROAD/geometry time so the
            # aggregate carries them too (single-run report() has these; the
            # parallel aggregate previously dropped them). Workers run
            # concurrently, so these are total *compute* across workers, not
            # wall-clock; wall-clock is `runtime_s` below.
            llm_time_s += float(tel.get("llm_time_s") or 0.0)
            system_time_s += float(tel.get("system_time_s") or 0.0)
            for agent, usage in (tel.get("by_agent") or {}).items():
                slot = by_agent.setdefault(agent, {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "seconds": 0.0,
                })
                slot["calls"] += int(usage.get("calls") or 0)
                slot["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                slot["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                slot["total_tokens"] += int(usage.get("total_tokens") or 0)
                slot["seconds"] += float(usage.get("seconds") or 0.0)
        return {
            "runtime_s": round(float(elapsed), 1),
            "llm_time_s": round(llm_time_s, 1),         # total model-driver compute
            "system_time_s": round(system_time_s, 1),   # total OpenROAD + geometry compute
            "tokens": {
                "prompt": prompt,
                "completion": completion,
                "total": prompt + completion,
            },
            "llm_calls": calls,
            "by_agent": by_agent,
        }

    def _export_output_result(self, elapsed: float, telemetry: dict | None = None) -> None:
        """Write compact benchmark result artifacts under project/output."""
        out_dir = self.cfg.project_root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        best = self.macro_store.best() if self.macro_store else None
        routed = {}
        if best is not None:
            routed = best.metrics.get("routed") or {}
            if not isinstance(routed, dict):
                routed = {}
        telemetry = telemetry or {}
        tokens = telemetry.get("tokens") or {}
        tns_goal = DESIGN_TNS_GOALS.get(self.cfg.design)
        tns_value = routed.get("tns")
        goal_met = None
        if tns_goal is not None and tns_value is not None:
            try:
                goal_met = float(tns_value) >= float(tns_goal)
            except (TypeError, ValueError):
                goal_met = None
        payload = {
            "design": self.cfg.design,
            "top": self.cfg.top,
            "data_dir": str(self.cfg.data_dir),
            "run_dir": str(self.cfg.output_root),
            "route_mode": self.cfg.route_mode_effective,
            "macro_candidates": self.macro_candidates,
            "parallel_workers": self.parallel_workers,
            "post_route_opt_loops": self.post_route_opt_loops,
            "elapsed_s": round(float(elapsed), 3),
            "best_candidate": best.id if best is not None else None,
            "tns_goal_ns": tns_goal,
            "goal_met": goal_met,
            "timing": {
                "wns": routed.get("wns"),
                "tns": routed.get("tns"),
            },
            "ppa": {
                "wns": routed.get("wns"),
                "tns": routed.get("tns"),
                "power": routed.get("power", routed.get("power_total")),
                "power_internal": routed.get("power_internal"),
                "power_switching": routed.get("power_switching"),
                "power_leakage": routed.get("power_leakage"),
                "drc": routed.get("drc"),
                "wirelength": routed.get("wirelength"),
                "area": routed.get("area"),
            },
            "token_usage": {
                "prompt": tokens.get("prompt", 0),
                "completion": tokens.get("completion", 0),
                "total": tokens.get("total", 0),
            },
            "llm_calls": telemetry.get("llm_calls", 0),
            "telemetry": telemetry,
            "summary": str(self.cfg.stage_model_dir("run") / "summary.txt"),
            "candidates_index": str(self.cfg.candidates_dir / "index.json"),
        }
        stamp = Path(self.cfg.output_root).name
        safe_design = "".join(c if c.isalnum() or c in "._-" else "_"
                              for c in self.cfg.design)
        result_path = out_dir / f"{stamp}_{safe_design}_result.json"
        latest_path = out_dir / f"latest_{safe_design}_result.json"
        result_path.write_text(json.dumps(payload, indent=2, default=str))
        latest_path.write_text(json.dumps(payload, indent=2, default=str))
        with (out_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
