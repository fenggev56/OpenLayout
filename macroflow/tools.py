"""Tools — the LLM-facing bridge to the system operations.

Each tool is a thin, *safe* wrapper around :class:`macroflow.operations.FlowOps`:
the model chooses *what* to do and with structured placement parameters; Python
does validation, legalization, OpenROAD execution, image capture and file
bookkeeping. The active Placer path provides explicit
``#index -> (x, y, orient)`` coordinates. Python clamps/legalizes before issuing
OpenROAD commands.

All layout images are captured from OpenROAD via ``save_image``. This module does
not call external image-generation APIs.

A :class:`ToolResult` carries text back to the model and, optionally, an image
to *show* it on the next turn (tool-role messages are text-only in Qwen, so the
agent loop injects the image as a follow-up user message).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass

from .geometry import COMPACT_ANCHORS
from .operations import FlowOps

_ANCHORS = sorted(COMPACT_ANCHORS.keys())
_SCRIPT_STAGES = ["init", "detail_place", "post_route"]


@dataclass
class ToolResult:
    text: str
    image: str | None = None        # path to attach so the model can see it
    image_caption: str = ""
    done: bool = False              # set by finish / submit_* (ends an agent turn)
    data: dict | None = None        # structured payload from a submit_* tool


def _fmt(d: dict) -> str:
    return json.dumps(d, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic format; converted to OpenAI for the Qwen template)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_design_state",
        "description": "Return the current design state: stage, macro count and "
                       "footprints, core bounds, macro-area utilization, compaction "
                       "rounds done, current vs best stdcell density, whether a "
                       "detailed-placement checkpoint / route exists. Call this to "
                       "orient yourself before deciding the next action.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_snapshots",
        "description": "List the saved layout snapshots (grouped by stage, latest "
                       "first, capped per stage). Use this to see WHICH images are "
                       "available, then view_layout the most relevant one. Saves you "
                       "from guessing filenames when many images exist.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "view_layout",
        "description": "Show yourself one layout image so you can visually judge it. "
                       "Pick the most relevant snapshot — pass a stage key "
                       "(macro_pin_sides | macro_placed | fast_placed | "
                       "detail_placed | routed | init), "
                       "a filename fragment, or 'latest'. The image is attached for "
                       "your next turn. Judge: are macros spread with dead pockets, or "
                       "tight? do stdcells crowd macro edges? is the stdcell block "
                       "contiguous and ~square?",
        "input_schema": {
            "type": "object",
            "properties": {
                "which": {"type": "string",
                          "description": "stage key, filename fragment, or 'latest'"},
            },
        },
    },
    {
        "name": "seed_macros",
        "description": "Place all macros into a deterministic, overlap-free packed "
                       "perimeter ring (legalized). This is the standard initial "
                       "placement. Call once after init, before compaction.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fast_place",
        "description": "Run a quick global+detailed placement of the CURRENT macro "
                       "arrangement (no PDN) to get a baseline stdcell-density image. "
                       "Use right after seeding to see the starting point for Gate B.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "compact_macros",
        "description": "Pull the macro cluster tighter and toward an anchor, then "
                       "legalize, place and fast-place. THIS is how you optimize the "
                       "layout: increase stdcell density by consolidating macros so "
                       "the stdcell region becomes one contiguous block. Returns the "
                       "new density and whether it improved. Repeat with different "
                       "strength/anchor until density stops improving.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strength": {"type": "number",
                             "description": "0.0 (no move) to 0.6 (very aggressive). "
                                            "Larger when macros are very spread out."},
                "anchor": {"type": "string", "enum": _ANCHORS,
                           "description": "Where to consolidate the cluster — pick the "
                                          "side OPPOSITE the largest empty void."},
            },
            "required": ["strength", "anchor"],
        },
    },
    {
        "name": "measure",
        "description": "Take a quantitative measurement of the current design: "
                       "'density' (stdcell tightness), 'overlaps' (macro overlap "
                       "count), or 'congestion' (cheap Gate-B proxy from density "
                       "and overlap checks; final route reports real routing).",
        "input_schema": {
            "type": "object",
            "properties": {
                "what": {"type": "string", "enum": ["density", "overlaps", "congestion"]},
            },
            "required": ["what"],
        },
    },
    {
        "name": "run_stage",
        "description": "Generic escape hatch to run any scripted flow stage from the "
                       "skills directory by key (init | fast_place). Prefer the "
                       "dedicated tools above; use this only for advanced/extra "
                       "stage runs.",
        "input_schema": {
            "type": "object",
            "properties": {"stage": {"type": "string", "enum": _SCRIPT_STAGES}},
            "required": ["stage"],
        },
    },
    {
        "name": "execute_tcl",
        "description": "Escape hatch: run raw TCL in the OpenROAD session and get the "
                       "(truncated) output. For inspection/queries only — do NOT use "
                       "it to place macros (use seed_macros/compact_macros).",
        "input_schema": {
            "type": "object",
            "properties": {"commands": {"type": "string"}},
            "required": ["commands"],
        },
    },
    {
        "name": "finish",
        "description": "End the agent session when no further macro-placement "
                       "improvement is possible. Provide a short summary.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
    # ---- structured terminals for the specialist agents -------------------
    {
        "name": "submit_plan",
        "description": "Submit your macro-placement STRATEGY after assessing the "
                       "floorplan image and visible IO-pin locations: the anchor to "
                       "consolidate the macro cluster toward and an initial compaction "
                       "strength, plus brief reasoning. Consolidate toward the side "
                       "OPPOSITE the largest expected stdcell region / void.",
        "input_schema": {
            "type": "object",
            "properties": {
                "anchor": {"type": "string", "enum": _ANCHORS},
                "strength": {"type": "number",
                             "description": "initial compaction aggressiveness 0.0-0.6"},
                "reasoning": {"type": "string"},
            },
            "required": ["anchor", "strength"],
        },
    },
    {
        "name": "submit_verdict",
        "description": "Submit your gate verdict for the layout image you inspected. "
                       "verdict PASS/FAIL; proceed=true means the layout is good enough "
                       "to advance to the next stage (stop refining). If FAIL/not "
                       "proceeding, suggest the next compaction anchor + strength.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                "feedback": {"type": "string"},
                "proceed": {"type": "boolean"},
                "suggest_anchor": {"type": "string", "enum": _ANCHORS},
                "suggest_strength": {"type": "number"},
            },
            "required": ["verdict", "feedback"],
        },
    },
    {
        "name": "submit_strategy",
        "description": "Submit your final macro-placement strategy after assessing the "
                       "floorplan: direct coordinate intent for each macro group, the "
                       "overall outline (ring / C / U), and orientation. The Placer "
                       "turns it into explicit coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {"strategy": {"type": "string"}},
            "required": ["strategy"],
        },
    },
    {
        "name": "submit_placement_plan",
        "description": "Place the macros at coordinates YOU choose (this is how you "
                       "place macros). Provide a plan with one line per macro: "
                       "'#<index> -> (<x_um>, <y_um>, <orient>)' using the #index from "
                       "the numbered macro list and x,y in microns inside the core. "
                       "Use computed decimal numbers, not arithmetic formulas. "
                       "Apply the planning rules: push macros to the periphery near "
                       "their pin side, cluster a group together, leave routing "
                       "channels, keep the stdcell area contiguous and ~square, avoid "
                       "dead pockets. You MUST include EVERY macro — a plan that omits "
                       "even one macro is rejected outright. Overlaps and edge overshoot "
                       "are repaired automatically, so never drop a macro to avoid them. "
                       "The placed image is returned — inspect it and call this again to "
                       "refine.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {"type": "string",
                         "description": "newline-separated '#index -> (x, y, orient)' lines"},
                "notes": {"type": "string"},
            },
            "required": ["plan"],
        },
    },
    {
        "name": "adjust_placement",
        "description": "FINE-GRAIN refinement: move a FEW macros by small offsets to fix a "
                       "specific problem you see in the attached image — e.g. close a thin "
                       "unroutable channel between a macro and the die edge, pull a macro "
                       "toward the pins/logic it serves, or open a clogged channel between "
                       "macros. Provide moves as '#<index> -> (<dx_um>, <dy_um>)' OFFSETS in "
                       "microns (+x right, +y up); optionally append an orientation. Only "
                       "list the macros you move — the rest stay put; overlaps are "
                       "auto-legalized. The re-placed image is returned; inspect it and "
                       "adjust again or finish.",
        "input_schema": {
            "type": "object",
            "properties": {
                "moves": {"type": "string",
                          "description": "newline-separated '#index -> (dx, dy)' micron offsets"},
                "notes": {"type": "string"},
            },
            "required": ["moves"],
        },
    },
    {
        "name": "place_in_regions",
        "description": "Legacy helper: place macros by DISTRIBUTION across the 4x4 "
                       "region grid. The active macro-flow agents do not use this "
                       "tool when direct coordinate placement is enabled. "
                       "Provide a region->count map whose counts sum to the macro "
                       "total, following your strategy; Python packs each region and "
                       "legalizes. Example (macros on the right side, C-shape): "
                       '{"r0c3":17,"r1c3":17,"r2c3":17,"r3c3":16,"r0c2":17,"r3c2":17,'
                       '"r0c1":16,"r3c1":15}. Macros are assigned in #index order to '
                       "the regions you list. The placed image is returned to refine.",
        "input_schema": {
            "type": "object",
            "properties": {
                "assignments": {"type": "object",
                                "description": "region label r<row>c<col> -> macro count",
                                "additionalProperties": {"type": "integer"}},
                "orient": {"type": "string"},
            },
            "required": ["assignments"],
        },
    },
]


def schemas_for(names: list[str] | None = None) -> list[dict]:
    """Return the subset of tool schemas for the given names (all if None)."""
    if names is None:
        return list(TOOL_SCHEMAS)
    want = set(names)
    return [t for t in TOOL_SCHEMAS if t["name"] in want]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class ToolDispatcher:
    def __init__(self, ops: FlowOps):
        self.ops = ops
        self.ctx = ops.ctx

    def dispatch(self, name: str, args: dict) -> ToolResult:
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return ToolResult(text=f"[ERROR] unknown tool '{name}'")
        try:
            return handler(args or {})
        except Exception as e:                       # keep the agent alive on tool errors
            tb = traceback.format_exc(limit=3)
            return ToolResult(text=f"[ERROR] tool '{name}' failed: {e}\n{tb}")

    # --- inspection ----------------------------------------------------
    def _t_get_design_state(self, args) -> ToolResult:
        return ToolResult(text=_fmt(self.ops.design_state_summary()))

    def _t_list_snapshots(self, args) -> ToolResult:
        return ToolResult(text=self.ctx.registry.catalog())

    def _t_view_layout(self, args) -> ToolResult:
        which = str(args.get("which", "latest"))
        snap = self.ctx.registry.resolve(which)
        if snap is None:
            return ToolResult(text=f"No snapshot matches '{which}'.\n"
                                   + self.ctx.registry.catalog())
        meta = f" {snap.meta}" if snap.meta else ""
        cap = f"[{snap.stage}] {snap.name}{meta}"
        return ToolResult(text=f"Showing {cap} — image attached below.",
                          image=snap.path, image_caption=cap)

    def _t_measure(self, args) -> ToolResult:
        what = str(args.get("what", "density")).lower()
        if what == "density":
            return ToolResult(text=f"stdcell density = {self.ops.measure_density()}")
        if what == "overlaps":
            return ToolResult(text=f"macro overlaps = {self.ops.check_overlaps()}")
        if what == "congestion":
            return ToolResult(text=_fmt(self.ops.measure_congestion()))
        return ToolResult(text=f"[ERROR] unknown measurement '{what}'")

    def _t_execute_tcl(self, args) -> ToolResult:
        cmds = str(args.get("commands", "")).strip()
        if not cmds:
            return ToolResult(text="[ERROR] no commands provided")
        out = self.ctx.session.send_command(cmds, timeout=600)
        return ToolResult(text=out[-1500:])

    # --- placement actions (auto-attach their result image) -----------
    def _t_seed_macros(self, args) -> ToolResult:
        r = self.ops.seed_macros()
        return ToolResult(text=_fmt(r), image=r.get("macro_image"),
                          image_caption="macros placed (ring seed)")

    def _t_fast_place(self, args) -> ToolResult:
        r = self.ops.fast_place()
        return ToolResult(text=_fmt(r), image=r.get("fast_image"),
                          image_caption="fast-placed (baseline)")

    def _t_compact_macros(self, args) -> ToolResult:
        try:
            strength = float(args.get("strength", 0.3))
        except (TypeError, ValueError):
            return ToolResult(text="[ERROR] 'strength' must be a number 0.0-0.6")
        anchor = str(args.get("anchor", "center")).lower()
        if anchor not in COMPACT_ANCHORS:
            return ToolResult(text=f"[ERROR] 'anchor' must be one of {_ANCHORS}")
        r = self.ops.compact(strength, anchor)
        return ToolResult(text=_fmt(r), image=r.get("fast_image"),
                          image_caption=f"fast-placed after compact (round {r.get('round')}, "
                                        f"density {r.get('density')})")

    def _t_run_stage(self, args) -> ToolResult:
        r = self.ops.run_stage(str(args.get("stage", "")))
        return ToolResult(text=_fmt(r), image=r.get("image"),
                          image_caption=f"stage {args.get('stage')}")

    # --- terminal ------------------------------------------------------
    def _t_finish(self, args) -> ToolResult:
        return ToolResult(text=f"FINISHED: {args.get('summary', '')}", done=True,
                          data={"summary": args.get("summary", "")})

    def _t_submit_plan(self, args) -> ToolResult:
        return ToolResult(text=f"Plan recorded: {args}", done=True, data=dict(args))

    def _t_submit_verdict(self, args) -> ToolResult:
        return ToolResult(text=f"Verdict recorded: {args}", done=True, data=dict(args))

    def _t_submit_strategy(self, args) -> ToolResult:
        return ToolResult(text="Strategy recorded.", done=True,
                          data={"strategy": args.get("strategy", "")})

    def _t_submit_placement_plan(self, args) -> ToolResult:
        plan = str(args.get("plan", ""))
        if not plan.strip():
            return ToolResult(text="[ERROR] empty plan — provide "
                                   "'#index -> (x, y, orient)' lines")
        # First plan starts from the ring base; later plans refine current coords.
        base = "current" if self.ops.st.entries else "ring"
        q = self.ops.preview_plan_quality(plan, base=base)
        if q.get("reject"):
            reasons = "; ".join(q.get("reasons", []))
            return ToolResult(
                text=("[ERROR] Coordinate plan rejected: "
                      f"{reasons}. You MUST give coordinates for ALL "
                      f"{q.get('n_macros')} macros in ONE plan — one "
                      "'#<index> -> (x, y, orient)' line per macro, #0 through "
                      f"#{max(0, int(q.get('n_macros') or 1) - 1)}, with no macro "
                      "omitted. Overlaps and edge overshoot are repaired "
                      "automatically, so do not drop a macro to avoid them; the "
                      "only fatal error is leaving a macro out."),
                data=q)
        r = self.ops.place_plan(plan, base=base, fast=False)
        return ToolResult(
            text=_fmt(r), image=r.get("macro_image"),
            image_caption=(f"macros placed from your plan "
                           f"({r.get('llm_coords_applied')}/{r.get('n_macros')} coords "
                           f"applied, overlaps after legalize {r.get('overlaps')})"),
            done=True,
            data=r)

    def _t_adjust_placement(self, args) -> ToolResult:
        moves = str(args.get("moves", "")).strip()
        if not moves:
            return ToolResult(text="[ERROR] empty moves — provide '#index -> (dx, dy)' "
                                   "micron offset lines")
        r = self.ops.nudge(moves)
        return ToolResult(
            text=_fmt(r), image=r.get("macro_image"),
            image_caption=(f"re-placed after your moves "
                           f"({r.get('applied')} applied, overlaps {r.get('overlaps')})"))

    def _t_place_in_regions(self, args) -> ToolResult:
        assignments = args.get("assignments") or {}
        if not isinstance(assignments, dict) or not assignments:
            return ToolResult(text="[ERROR] provide 'assignments' as "
                                   "{region_label: count}, e.g. {\"r0c3\": 17}")
        orient = str(args.get("orient", "R0"))
        base = "current" if self.ops.st.entries else "ring"
        r = self.ops.place_in_regions(assignments, orient=orient, base=base, fast=False)
        return ToolResult(
            text=_fmt(r), image=r.get("macro_image"),
            image_caption=(f"macros placed by region distribution "
                           f"({r.get('macros_distributed')}/{r.get('n_macros')}, "
                           f"overlaps {r.get('overlaps')})"),
            done=True,
            data=r)
