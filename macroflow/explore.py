"""Goal-driven exploration controller — the coarse/fine optimization loop.

Wraps the per-try pipeline ``place a macro arrangement → detail-place → route →
read PPA → check the goal`` in a **deterministic** policy that explores *different*
placements until the goal is met or the try budget is spent:

  * **Try 0** is the agentic placement the orchestrator already produced
    (Planner / Placer / Checker + the Gate-B density loop).
  * Then, deterministically, per try:
      - **fine-grain** (default): reload the best candidate's coords + image and
        let the Refiner make small, targeted moves (close a thin edge channel,
        pull a macro toward its pins …);
      - **coarse-grain**: when a try fails to improve the goal, place a new preset
        OUTLINE (ring / C / U / corner / split) to explore a different shape.
    A try that sets a new best returns the loop to fine refinement; coarse shapes
    advance only when nothing improved.

Each try runs a full detail-place + route because post-route timing is the only
honest PPA signal (``5.PPA_checker/SKILL.md``) — so the budget is small by design.
Candidates are persisted by :class:`~macroflow.candidates.CandidateStore`; the
goal-best routed odb is promoted to ``best_overall.odb``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import geometry as geo
from .goal import Decision


class ExplorationController:
    def __init__(self, ctx, ops, specialists, goal, store, ppacheck,
                 verbose: bool = True):
        self.ctx = ctx
        self.ops = ops
        self.sp = specialists
        self.goal = goal
        self.store = store
        self.ppacheck = ppacheck
        self.tel = ctx.telemetry
        self.verbose = verbose
        self.max_tries = 1

    # ------------------------------------------------------------------
    def run(self, seed_coords: list, max_tries: int, route: bool = True,
            explore: bool = True) -> None:
        self.max_tries = max(1, int(max_tries))
        ctx, ops = self.ctx, self.ops

        # --- Try 0: the agentic placement, taken to a routed candidate -----
        decision, _ = self._evaluate(0, kind="seed", coords=seed_coords,
                                     shape="agentic", route=route)
        if not route:                               # --stop-after detail
            ctx.note("[explore] detail-only (stop-after detail) — no exploration")
            return
        if decision == Decision.STOP_GOAL or not explore:
            self._finalize_best()
            return

        # --- Exploration loop: deterministic coarse/fine policy ------------
        mode = "fine"
        shapes = list(geo.COARSE_SHAPE_ORDER)
        shape_i = 0
        for t in range(1, self.max_tries):
            ctx.state.try_tag = f"t{t:02d}"        # tag refine/place images per try
            best = self.store.best()
            use_fine = (mode == "fine" and best is not None)
            if not use_fine and shape_i >= len(shapes):
                use_fine = best is not None         # coarse shapes exhausted -> fine
                mode = "fine"

            if use_fine:
                ctx.note(f"[explore] try {t}: FINE-grain refine of candidate "
                         f"#{best.id} (shape '{best.shape}')")
                self.tel.incr("fine_tries")
                ops.reload_clean()
                ops.load_entries(best.coords)
                self.sp.refine(best)
                decision, is_best = self._evaluate(
                    t, kind="fine", coords=ctx.state.entries, shape=best.shape,
                    parent=best.id, route=True)
            else:
                shape = shapes[shape_i]
                shape_i += 1
                ctx.note(f"[explore] try {t}: COARSE-grain new shape '{shape}'")
                self.tel.incr("coarse_tries")
                ops.reload_clean()
                ops.place_shape(shape)
                decision, is_best = self._evaluate(
                    t, kind="coarse", coords=ctx.state.entries, shape=shape,
                    route=True)

            if decision == Decision.STOP_GOAL:
                break
            # Policy: refine a fresh best further; otherwise explore a new shape.
            mode = "fine" if is_best else "coarse"

        self._finalize_best()

    # ------------------------------------------------------------------
    def _evaluate(self, t: int, kind: str, coords: list, shape: str,
                  route: bool = True, parent: int | None = None):
        """Detail-place + route one candidate, read PPA, run the goal/PPA check,
        persist it. Returns ``(Decision, is_new_best)``."""
        ctx, ops = self.ctx, self.ops
        tag = f"t{t:02d}"
        ctx.state.try_tag = tag
        cand = self.store.new(kind=kind, coords=[dict(e) for e in coords],
                              shape=shape, parent=parent)

        with self.tel.op("detail"):
            dp = ops.detail_place(entries=coords, tag=tag)
        detail_db = dp.get("detail_db")

        if not route:
            return Decision.CONTINUE, False

        routed_odb = ctx.cfg.save_dir / f"{tag}_routed.odb"
        with self.tel.op("route"):
            rr = ops.route_finalize(tag=tag, out_odb=routed_odb)
        metrics = rr.get("metrics", {}) if isinstance(rr, dict) else {}

        decision, status, report = self.ppacheck.decide(t, self.max_tries, metrics)
        images = {"macro": str(ctx.img(f"{tag}_detail_macro_placed.png",
                                        stage="macro_placed")),
                  "detail": str(ctx.img(f"{tag}_detail_placed.png",
                                         stage="detail_placed")),
                  "routed": str(ctx.img(f"{tag}_routed.png", stage="routed"))}
        is_best = self.store.commit(cand, metrics, status, decision.value,
                                    routed_odb=str(routed_odb), images=images)
        # Per-try detail / macro odbs are dead weight once routed (the best's
        # routed odb is kept by the store).
        self._prune(detail_db)
        self._prune(ctx.cfg.save_dir / f"{tag}_detail_macro_placed.odb")

        self.tel.incr("tries")
        ctx.note("[PPA-check]\n" + report)
        return decision, is_best

    # ------------------------------------------------------------------
    def _finalize_best(self) -> None:
        ctx, cfg = self.ctx, self.ctx.cfg
        best = self.store.best()
        if best is None:
            ctx.note("[explore] no routed candidate to finalize")
            return
        if best.routed_odb and Path(best.routed_odb).exists():
            try:
                shutil.move(best.routed_odb, str(cfg.best_overall_odb))
                best.routed_odb = str(cfg.best_overall_odb)
                self.store.write_index()
            except OSError as e:
                ctx.note(f"[explore] could not promote best odb: {e}")
        ctx.state.routed_metrics = best.metrics or ctx.state.routed_metrics
        ctx.note(f"[explore] BEST = candidate #{best.id} (shape '{best.shape}', "
                 f"kind {best.kind}) metrics={best.metrics}")
        ctx.note(self.store.render_table())

    @staticmethod
    def _prune(path) -> None:
        try:
            p = Path(path)
            if path and p.exists():
                p.unlink()
        except (OSError, TypeError):
            pass
