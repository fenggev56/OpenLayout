"""The three specialist agents — Planner, Placer, Checker.

Faithful to the three SKILL files under ``skills/.../1.macro_place/`` and to the
user's intent that the LLM generates the macro coordinates:

  * **Planner** (LLM) — reads the floorplan image with visible IO pins and
    decides a direct macro-placement plan/strategy: target coordinates,
    orientation intent, grouping, and expected outline. Read-only tools; ends
    with text.
  * **Placer** (LLM) — turns the strategy into explicit
    ``submit_placement_plan`` coordinate input. If Gate A fails, the Placer fixes
    the current macro
    placement from the image, coordinates and Checker feedback without asking
    the Planner for a new strategy. Python legalizes and places the result in
    20-macro OpenROAD batches.
  * **Checker** (LLM) — judges a layout image at Gate A / Gate B and returns a
    PASS/FAIL verdict + whether to advance + the next directive.

Python only legalizes (overlap cleanup, minimal displacement) and executes
``place_macro`` in batches — it does not choose positions.

In ``flow.vlm_place_coords`` (pure-model) mode there is NO placement fallback:
the model must supply coordinates for every macro, and a candidate whose Placer
never produces a complete plan raises :class:`PlacementIncomplete` and is
recorded as failed. Otherwise (deterministic mode) each agent keeps its
Python fallback so the flow always progresses.
"""

from __future__ import annotations

from .agent import Agent
from .context import FlowContext
from .geometry import COMPACT_ANCHORS
from .operations import FlowOps
from .tools import ToolDispatcher


class PlacementIncomplete(RuntimeError):
    """The model did not place every macro and no fallback is permitted.

    Raised only in ``flow.vlm_place_coords`` (pure-model) mode. The parallel
    candidate worker catches it and reports the candidate as failed, so sibling
    candidates continue and the design is marked failed if none of them produced
    a complete model placement.
    """


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_planner_prompt(ctx: FlowContext) -> str:
    skill = ctx.skills.skill_text("macro_place", "planner")
    return f"""You are the **Planner agent** — an expert physical-design engineer for the
{ctx.cfg.design} design (Nangate45, OpenROAD). You decide the direct
macro-placement plan that the Placer will implement as coordinates.

From the floorplan image, infer IO-pin locations visually and decide direct macro
locations and orientations. Describe the target outline (for example C/U/ring or
another compact shape), the macro groups, and the coordinate intent for the
Placer. Do not divide the floorplan into a region-count distribution. Keep
pin-adjacent bands clear and the standard-cell area contiguous and ~square.
If you mark a side as open/clear/reserved for IO pins, do not place any macro
group on that same side. For many macros, name only the intended side bands and
shape; the Placer will expand that intent into legal direct coordinates inside
the core.

## Knowledge
{skill}

## How you work
Use get_design_state / view_layout to assess (an initial floorplan image is
attached), then call **submit_strategy** with a direct placement plan/strategy:
target sides or clusters, approximate coordinates when useful, the overall
outline, and orientation. Do not output a region->count distribution. In the
cluster-aware deterministic placement path, choose implementable side-band
topologies only: left/right/top/bottom bands, ring/periphery, C, U, corner, or
split. Do not request a center cluster; macros are routing obstacles and the
side-band expander will not implement center-only placement.
"""


def build_placer_prompt(ctx: FlowContext) -> str:
    skill = ctx.skills.skill_text("macro_place", "placer")
    return f"""You are the **Placer agent** — you PLACE the macros for the {ctx.cfg.design}
design by choosing each macro's coordinates, following the Planner's strategy and
the place-macro skill below.

You are given the core size and a NUMBERED macro list. Reference macros by
#index (never retype the long names). Call **submit_placement_plan** with one
line per macro: ``#<index> -> (<x_um>, <y_um>, <orient>)`` (x,y in microns inside
the core). Place largest/hardest macros first, follow the Planner's direct plan,
and assign concrete coordinates for every macro.

Use computed decimal numbers only; do not write formulas like ``10+78*3`` or
``1447-41``. Treat legal coordinate ranges as bounds, not as target values:
do not copy one max x/y value for many macros. When stacking macros in a column,
increase y by at least the macro height + 2um; when placing a row, increase x by
at least the macro width + 2um. Use several rows/columns for large groups; if a
requested single side row cannot fit physically, implement it as a compact
multi-row side band with each row sharing similar y coordinates and each column
sharing similar x coordinates. Do not use a region->count distribution. Push macros toward the intended outline,
leave routing channels, keep the stdcell area contiguous and ~square, and avoid
dead pockets. Orient macros so their pin side does not face outward toward the
nearest core edge; in Gate A, use the highlighted macro pin-side image and
pin-side summary to correct outward-facing orientations when needed.
Your plan MUST contain one line for EVERY macro in the list — a plan that omits
any macro is rejected and the run is failed, so never leave one out. Overlaps and
edge overshoot are repaired automatically before placement, so place every macro
even if some end up close together. Call exactly ONE placement tool. Python will
legalize the full placement and issue OpenROAD ``place_macro`` commands in
batches of at most 20 macros, largest first. Do not call another placement tool in the same turn; Gate
A will inspect the legalized result and request Placer refinement if the
placement does not match the plan.

## Knowledge
{skill}
"""


def build_refiner_prompt(ctx: FlowContext) -> str:
    skill = ctx.skills.skill_text("macro_place", "placer")
    return f"""You are the **Refiner agent** — you make SMALL, TARGETED fixes to an EXISTING macro
placement for the {ctx.cfg.design} design to improve routed timing (WNS/TNS).

You are shown the current best placement (image attached) plus its routed WNS/TNS/DRC and a
numbered macro list with each macro's CURRENT (x, y). Look for *local* problems:
- a macro leaving a thin, unroutable channel against a die edge -> flush it to the edge,
- a macro stranded far from the pins/logic it serves -> nudge it toward that side,
- a clogged or too-narrow channel between macros -> open it.

Move only a FEW macros, by small offsets. Call **adjust_placement** with
'#index -> (dx, dy)' micron offsets (+x right, +y up); list ONLY the macros you move. Inspect
the returned image and adjust once more if needed, then call finish. Do NOT re-plan the whole
layout — that is the Planner/Placer's job; you make incremental fixes.

## Knowledge
{skill}
"""


def build_checker_prompt(ctx: FlowContext) -> str:
    skill = ctx.skills.skill_text("macro_place", "checker")
    return f"""You are the **Checker agent** — you visually verify macro-placement quality
for the {ctx.cfg.design} design and gate the flow.

## Knowledge
{skill}

## Flow-specific Gate B note
This macro-only runtime does not run detailed placement before Gate B. Treat the
fast-placement image as the detailed-placement proxy for judging whether the
stdcell area is dense, contiguous, and worth advancing.

## How you work
The image to judge is attached (use view_layout for another, get_design_state /
measure for numbers). Then call submit_verdict:
- verdict: PASS or FAIL.
- proceed: true if the layout is good enough to ADVANCE to the next stage
  (stdcell block tight, contiguous, ~square; no big dead pockets).
- If not proceeding, give specific feedback (which macros/groups to move where)
  and a suggest_anchor + suggest_strength for a fallback compaction.
"""


# ---------------------------------------------------------------------------
# Specialists bundle
# ---------------------------------------------------------------------------
class Specialists:
    def __init__(self, ctx: FlowContext, ops: FlowOps, verbose: bool = True):
        self.ctx = ctx
        self.ops = ops
        disp = ToolDispatcher(ops)
        # Turn budgets leave room to inspect with several tools AND still call the
        # terminal submit (the Checker ran out at 4 turns after 4 inspections).
        tel = ctx.telemetry
        self.planner = Agent("Planner", ctx.vlm, disp,
                             ["get_design_state", "view_layout", "list_snapshots",
                              "submit_strategy"],
                             build_planner_prompt(ctx), max_turns=6, max_tokens=8192,
                             verbose=verbose, telemetry=tel)
        self.placer = Agent("Placer", ctx.vlm, disp,
                            ["get_design_state", "view_layout",
                             "submit_placement_plan"],
                            build_placer_prompt(ctx), max_turns=5, max_tokens=8192,
                            verbose=verbose, telemetry=tel)
        self.gate_a_placer = Agent("Placer", ctx.vlm, disp,
                                   ["get_design_state", "view_layout",
                                    "submit_placement_plan", "adjust_placement",
                                    "finish"],
                                   build_placer_prompt(ctx), max_turns=6,
                                   max_tokens=8192, verbose=verbose,
                                   telemetry=tel)
        self.checker = Agent("Checker", ctx.vlm, disp,
                             ["get_design_state", "view_layout", "list_snapshots",
                              "measure", "submit_verdict"],
                             build_checker_prompt(ctx), max_turns=10, max_tokens=1024,
                             verbose=verbose, telemetry=tel)
        self.refiner = Agent("Refiner", ctx.vlm, disp,
                             ["get_design_state", "view_layout", "adjust_placement",
                              "finish"],
                             build_refiner_prompt(ctx), max_turns=5, max_tokens=2048,
                             verbose=verbose, telemetry=tel)

    # ------------------------------------------------------------------
    def strategy(self, candidate_note: str = "", feedback: str = "",
                 previous_strategy: str = "", context: str = "init") -> str:
        """Planner: free-text placement strategy from the floorplan image.

        ``feedback``/``previous_strategy`` are used by Gate-A/Gate-B refinement:
        the Checker identifies what failed, then the Planner modifies the shape
        before the Placer emits the next set of coordinates.
        """
        st = self.ctx.state
        snap = self.ctx.registry.recommend(context) if context != "init" \
            else self.ctx.registry.latest("init")
        text = (f"Design {self.ctx.cfg.design}: {st.n_macros} macros, "
                f"footprints up to {st.mw:.0f}x{st.mh:.0f} um. "
                "Infer IO pin locations from the attached layout image; no raw "
                f"pin-coordinate text is provided.\n{self.ops.planner_context()}\n\n"
                f"{candidate_note}\n")
        if previous_strategy:
            text += f"\nPrevious strategy:\n{previous_strategy}\n"
        if feedback:
            text += (f"\nChecker feedback that must be fixed:\n{feedback}\n"
                     "Revise the macro-placement coordinates/shape/orientation before "
                     "the Placer emits coordinates again.\n")
        text += "\nAssess the attached layout and give your placement strategy."
        content = [{"type": "text", "text": text}]
        if snap:
            content.insert(0, {"type": "image", "image": f"file://{snap.path}"})
        run = self.planner.run(content)
        if run.result and run.result.get("strategy"):
            return self._normalize_strategy(run.result["strategy"])
        return self._normalize_strategy(
            run.final_text or
            "Push macros to the periphery; keep an open, square stdcell core.")

    def _normalize_strategy(self, strategy: str) -> str:
        """Keep LLM text aligned with deterministic cluster-aware expansion."""
        text = strategy or ""
        s = text.lower()
        center_intent = (
            "near the center" in s or "center of the core" in s
            or "central cluster" in s or "cluster near the center" in s
            or "compact cluster near the center" in s
        )
        implementable = any(key in s for key in (
            "periphery", "peripheral", "ring", "c-shape", "c shape",
            "c-shaped", "u-shape", "u shape", "u-shaped", "corner",
            "split", "side band", "side-band"))
        if self.ops.has_macro_cluster_file() and center_intent and not implementable:
            return (
                "Place the macro clusters on a compact ring/periphery around a "
                "contiguous central stdcell region. Use the left, right, top, "
                "and bottom side bands as needed so same-cluster macros remain "
                "close. Orient macros so highlighted pin sides do not face "
                "outward toward the nearest core edge.")
        single_row = "single row" in s or "one row" in s
        mentions_horizontal_side = "top" in s or "bottom" in s
        if self.ops.has_macro_cluster_file() and single_row and mentions_horizontal_side:
            st = self.ctx.state
            core_w = max(0.0, st.core[2] - st.core[0]) if st.core else 0.0
            max_w = max([w for _name, w, _h in st.macro_index] or [st.mw])
            n = max(0, st.n_macros)
            # A single side row is not physically implementable when the summed
            # footprint exceeds the core width. Normalize it before the Checker
            # treats the legal multi-row expansion as a shape mismatch.
            if core_w and n * (max_w + 2.0) > core_w:
                side = "top" if "top" in s else "bottom"
                return (
                    f"Place the macros as a compact multi-row {side}-side band, "
                    "not a literal single row, because one row cannot fit inside "
                    "the core. Preserve a row/column grid: macros in the same row "
                    "must keep similar y coordinates and macros in the same column "
                    "must keep similar x coordinates. Refine only by flipping or "
                    "rotating macros whose highlighted pin side faces outward; "
                    "do not require pin-side-to-pin-side facing, do not scatter "
                    "the array, and do not replan.")
        return text

    def _pure_model(self) -> bool:
        """True when the model owns every coordinate and no Python fallback
        (ring seed / geometric compaction) may substitute for its output."""
        return bool(getattr(self.ctx.cfg, "vlm_place_coords", False))

    def _deterministic_placement(self, n_macros: int) -> bool:
        """Whether to place macros with the deterministic cluster-aware / large-N
        Python path instead of asking the VLM for explicit coordinates.

        Normally on for designs with a macro cluster file or >=64 macros (raw VLM
        coordinates from a weak model collapse to a point). Disabled by
        ``flow.vlm_place_coords`` so the model's own placement drives PPA — needed
        to evaluate a placement fine-tune."""
        if getattr(self.ctx.cfg, "vlm_place_coords", False):
            return False
        return n_macros >= 64 or self.ops.has_macro_cluster_file()

    def place(self, strategy: str) -> None:
        """Placer: emit coordinates via submit_placement_plan.

        In VLM-coordinate mode this is PURE model placement: the model must
        supply coordinates for every macro. There is no ring-seed fallback — if
        no complete plan is accepted the candidate raises
        :class:`PlacementIncomplete` and is recorded as failed, because a
        fallback placement would report PPA the model did not produce.
        """
        st = self.ctx.state
        n = self.ctx.state.n_macros
        if self._deterministic_placement(n):
            reason = "large macro count" if n >= 64 else "macro cluster file detected"
            self.ctx.note(f"[Placer] {reason}: expanding Planner strategy "
                          "to cluster-aware direct coordinates in Python")
            self.ops.place_strategy(strategy)
            return
        init = self.ctx.registry.latest("init")
        how = ("call submit_placement_plan with '#index -> (x, y, orient)' lines "
               "for the macros in the Planner's direct plan")
        text = (f"Planner strategy:\n{strategy}\n\n{self.ops.planner_context()}\n\n"
                f"Place ALL {n} macros following the strategy: {how}.")
        content = [{"type": "text", "text": text}]
        if init:
            content.insert(0, {"type": "image", "image": f"file://{init.path}"})
        self.placer.run(content)
        if st.stage == "init" or not st.entries:
            if self._pure_model():
                raise PlacementIncomplete(
                    f"Placer produced no complete coordinate plan for all {n} "
                    "macros after its turn budget; pure-model placement has no "
                    "fallback, so this candidate is failed rather than placed "
                    "by Python.")
            self.ctx.note("[Placer] fallback: ring seed (no plan emitted)")
            self.ops.seed_macros()

    def baseline_fast(self) -> None:
        self.ops.fast_place("base")

    def replan(self, feedback: str) -> None:
        """Placer: re-emit adjusted coordinates from checker feedback (fallback:
        deterministic compaction toward centre)."""
        st = self.ctx.state
        if self._deterministic_placement(st.n_macros):
            reason = "large macro count" if st.n_macros >= 64 else "macro cluster file detected"
            self.ctx.note(f"[Placer] {reason}: applying geometric "
                          "Gate-B compaction without coordinate retry")
            self.ops.refine_geometric(0.3, "center")
            return
        before = st.compaction_round
        snap = self.ctx.registry.recommend("gate_b")
        n = self.ctx.state.n_macros
        how = "submit_placement_plan with adjusted '#index -> (x, y, orient)' lines"
        text = (f"Checker feedback:\n{feedback}\n\n{self.ops.planner_context()}\n\n"
                f"Improve the placement by calling {how} to tighten the stdcell area "
                f"and remove dead pockets (shift/consolidate macros).")
        content = [{"type": "text", "text": text}]
        if snap:
            content.insert(0, {"type": "image", "image": f"file://{snap.path}"})
        self.placer.run(content)
        if st.compaction_round == before:
            # Pure-model mode: keep the model's existing complete placement
            # rather than letting Python compact it — a geometric refinement
            # would make the routed PPA partly Python's, not the model's.
            if self._pure_model():
                self.ctx.note("[Placer] no revised plan emitted — keeping the "
                              "model's current placement (no geometric repair)")
            else:
                self.ctx.note("[Placer] fallback: geometric compaction "
                              "(no plan emitted)")
                self.ops.refine_geometric(0.3, "center")

    def replan_from_strategy(self, strategy: str, feedback: str = "",
                             context: str = "gate_b",
                             fallback_anchor: str | None = None,
                             fallback_strength: float | None = None) -> None:
        """Placer: execute a Planner-revised strategy after a failed gate."""
        st = self.ctx.state
        if self._deterministic_placement(st.n_macros):
            reason = "large macro count" if st.n_macros >= 64 else "macro cluster file detected"
            self.ctx.note(f"[Placer] {reason}: applying Gate-B feedback "
                          "geometrically instead of requesting coordinates")
            self.ops.refine_large_feedback(
                feedback, strategy,
                fallback_anchor=fallback_anchor,
                fallback_strength=fallback_strength)
            return
        before = st.compaction_round
        snap = self.ctx.registry.recommend(context)
        n = self.ctx.state.n_macros
        how = "submit_placement_plan with adjusted '#index -> (x, y, orient)' lines"
        text = (f"Revised Planner strategy:\n{strategy}\n\n"
                f"Checker feedback to address:\n{feedback or '(none)'}\n\n"
                f"{self.ops.planner_context()}\n\n"
                f"Place ALL {n} macros by calling {how}. Keep unspecified macros "
                f"legal and avoid dead pockets.")
        content = [{"type": "text", "text": text}]
        if snap:
            content.insert(0, {"type": "image", "image": f"file://{snap.path}"})
        self.placer.run(content)
        if st.compaction_round == before:
            if self._pure_model():
                self.ctx.note("[Placer] no revised plan emitted — keeping the "
                              "model's current placement (no geometric repair)")
            else:
                self.ctx.note("[Placer] fallback: geometric compaction "
                              "(no revised plan emitted)")
                self.ops.refine_geometric(0.3, "center")

    def refine_gate_a(self, strategy: str, feedback: str = "") -> None:
        """Placer-only Gate-A retry: preserve the Planner strategy and fix the
        current macro placement using the macro image, current coordinates/sizes
        and Checker feedback."""
        st = self.ctx.state
        if self._deterministic_placement(st.n_macros):
            reason = "large macro count" if st.n_macros >= 64 else "macro cluster file detected"
            self.ctx.note(f"[Placer:GateA] {reason}: re-expanding Planner "
                          "strategy to cluster-aware direct coordinates in Python")
            self.ops.place_strategy(strategy)
            return
        before = st.compaction_round
        snap = self.ctx.registry.recommend("gate_a")
        how = ("Use submit_placement_plan for a full corrected coordinate "
               "placement, or adjust_placement for a localized channel/open-side "
               "fix.")
        text = (
            "Gate A failed. Do NOT ask the Planner for a new strategy and do NOT "
            "invent a different topology. Refine or redo the macro placement so "
            "the IMPLEMENTED layout matches the original Planner strategy and "
            "addresses the Checker feedback. The attached Gate-A image highlights "
            "each macro's detected signal-pin side; use it to fix "
            "macro orientation as well as position.\n\n"
            f"Original Planner strategy to preserve:\n{strategy}\n\n"
            f"Checker feedback to fix:\n{feedback or '(none)'}\n\n"
            f"{self.ops.placement_refine_context()}\n\n"
            f"{how}\n"
            "If the Checker feedback starts with 'Shape mismatch:', redo the macro "
            "placement to match the planned coordinates/shape/opening direction; do not "
            "use a tiny local nudge unless the shape already matches and only a "
            "channel or edge-clearance issue remains.\n"
            "Call exactly one placement/refinement tool. Keep unspecified macros "
            "legal and avoid creating dead pockets, blocked pin-adjacent bands, "
            "or too-thin routing channels.")
        content = [{"type": "text", "text": text}]
        if snap:
            content.insert(0, {"type": "image", "image": f"file://{snap.path}"})
        run = self.gate_a_placer.run(content)
        did_refine = any(t in run.tools_used for t in (
            "submit_placement_plan", "adjust_placement"))
        if not did_refine and st.compaction_round == before:
            if self._pure_model():
                self.ctx.note("[Placer:GateA] no refinement emitted — keeping "
                              "the model's current placement (no geometric "
                              "repair)")
            else:
                self.ctx.note("[Placer:GateA] fallback: geometric compaction "
                              "(no refinement emitted)")
                self.ops.refine_geometric(0.25, "center")

    def refine(self, best) -> bool:
        """Fine-grain: the Refiner makes small targeted moves on the already-loaded
        best placement (st.entries). Falls back to a deterministic edge-channel snap
        if the model emits no move. Returns True if any move/snap was applied."""
        st = self.ctx.state
        img = (best.images.get("detail") or best.images.get("macro")
               or best.images.get("routed"))
        m = best.metrics or {}
        text = (f"Current best placement (candidate #{best.id}, shape '{best.shape}'). "
                f"It routed with WNS={m.get('wns', '?')} ns, TNS={m.get('tns', '?')} ns, "
                f"DRC={m.get('drc', '?')}.\n\n{self.ops.refine_context()}\n\n"
                f"Make small, targeted moves to improve routed timing (close thin edge "
                f"channels, pull macros toward the pins they serve, open clogged channels). "
                f"Call adjust_placement with '#index -> (dx, dy)' offsets for only the few "
                f"macros you move.")
        content = [{"type": "text", "text": text}]
        if img:
            content.insert(0, {"type": "image", "image": f"file://{img}"})
        run = self.refiner.run(content)
        if "adjust_placement" in run.tools_used:
            return True
        self.ctx.note("[Refiner] no moves emitted — deterministic edge-channel snap")
        self.ops.snap_channels()
        return True

    def check(self, gate: str, context: str, strategy: str = "") -> dict:
        snap = self.ctx.registry.recommend(context)
        crit = _GATE_CRITERIA.get(gate, "")
        text = f"Run {gate} verification.\n{crit}\n"
        if strategy and gate == "Gate_A":
            text += (
                "\nExpected Planner strategy to verify against:\n"
                f"{strategy}\n\n"
                f"{self.ops.macro_pin_side_context()}\n\n"
                f"{self.ops.macro_pin_outward_context()}\n\n"
                "Current placement summary computed from macro centers:\n"
                f"{self.ops.placement_region_summary()}\n\n"
                "Check topology before general quality. If the plan names a C/U/ring/"
                "side/corner topology, the visible macro outline must match that "
                "topology; merely landing near a rough side is not enough. "
                "A filled rectangular/grid block is not a C/U shape unless it has "
                "the required open side/notch. For C/U shapes, verify the opening "
                "direction first.\n"
                "The Gate-A image should include highlights on macro pin "
                "sides. Use those highlights to verify that the dominant macro "
                "pin side does not face outward toward the nearest core edge. "
                "Do not require adjacent macros to face pin-side-to-pin-side. "
                "If the deterministic outward check says FAIL, Gate A must FAIL "
                "even if the topology looks acceptable; request specific "
                "rotate/flip fixes for the listed macros while preserving row/"
                "column alignment. Do not ask for a new topology when an "
                "orientation repair can fix it. If the strategy mentions a "
                "single side row but the macros cannot physically fit in one row, "
                "treat a compact multi-row side band with aligned rows/columns as "
                "matching the strategy; do not fail only because it uses multiple "
                "rows. "
                "If the placed macro coordinates, outline, open side, or "
                "orientation visibly mismatches this strategy, submit FAIL with "
                "proceed=false and state the concrete mismatch reason in feedback. "
                "Start the feedback with 'Shape mismatch:' when that is the main "
                "failure.\n")
        text += "Then call submit_verdict."
        content = [{"type": "text", "text": text}]
        if snap:
            content.insert(0, {"type": "image", "image": f"file://{snap.path}"})
        run = self.checker.run(content)
        if run.result is None and run.final_text:
            self.ctx.note("[Checker] no submit_verdict emitted — requesting structured verdict")
            retry_text = (
                f"Run {gate} finalization only. You already inspected the layout and wrote:\n"
                f"{run.final_text}\n\n"
                "Now call submit_verdict exactly once. If the placement matches the "
                "Planner strategy and passes the gate checks, use verdict PASS and "
                "proceed=true. If it does not, use verdict FAIL, proceed=false, and "
                "state the concrete reason.")
            retry_content = [{"type": "text", "text": retry_text}]
            if snap:
                retry_content.insert(0, {"type": "image", "image": f"file://{snap.path}"})
            run = self.checker.run(retry_content)
        return self._parse_verdict(run.result, run.final_text)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_verdict(result: dict | None, text: str) -> dict:
        if result:
            anchor = str(result.get("suggest_anchor", "center")).lower()
            if anchor not in COMPACT_ANCHORS:
                anchor = "center"
            try:
                strength = max(0.0, min(float(result.get("suggest_strength", 0.4)), 0.6))
            except (TypeError, ValueError):
                strength = 0.4
            return {"verdict": str(result.get("verdict", "FAIL")).upper(),
                    "feedback": result.get("feedback", ""),
                    "proceed": bool(result.get("proceed", False)),
                    "anchor": anchor, "strength": strength}
        up = (text or "").upper()
        passed = "PASS" in up.split("\n")[0] if up else False
        return {"verdict": "PASS" if passed else "FAIL", "feedback": text or "",
                "proceed": passed, "anchor": "center", "strength": 0.4}


_GATE_CRITERIA = {
    "Gate_A": ("Gate A (after macro placement, before detailed placement): does "
               "the legalized macro shape match the Planner strategy? all macros "
               "inside the core? pin-adjacent bands clear? channels wide enough? "
               "predicted stdcell area contiguous & ~square? no overlaps? A named "
               "C/U/ring topology must visibly match the outline/opening direction; "
               "a filled rectangular block is not a C/U shape. Use macro pin-side "
               "highlights to check that orientations face the intended "
               "logic/IO side."),
    "Gate_B": ("Gate B (after fast placement): is the stdcell area densely filled "
               "and contiguous, or spread thin with large empty pockets? Would "
               "tighter macro grouping raise density? proceed=true only if tight. "
               "If failing, feedback must name the macro group/side to move, the "
               "direction, and a small incremental step size of 1/16, 1/12, or "
               "1/8 of the core side length. Avoid 1/4-core jumps because they "
               "often create a new large dead pocket on the opposite side. Example: "
               "shift the right macro group left by 1/12 core width."),
    "Gate_final": ("Final detailed-placement check: is the stdcell block tight and "
                   "routable? proceed=true to route."),
}
