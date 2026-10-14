# Macro-Placement Optimization Loop — coding-agent prompt

This file is a **prompt for a coding agent** (Claude Code, Codex, or any
agent that can read files, run shell commands and look at images). Paste it
(or point the agent at it) and the agent itself becomes the Planner /
Checker / Optimizer that drives OpenROAD through the macro-placement →
detail-place → CTS → route → timing loop, using the skills under
`skills/skills_Nangate45/`.

It is the second way to run OpenLayout, next to the API-driven framework
(`run_openlayout.py`). Same skills, same gates, same fairness rules — but
the reasoning is done by the interactive coding agent instead of an LLM
reached through an API.

Typical invocation:

```bash
cd OpenLayout
claude      # or: codex
> Read Instructions/macro_placement_loop.md and run it with DESIGN=bp_fe ASPECT=ar2.
```

---

Run a full, **fair** macro-placement → route → timing flow for one design,
driven by the skills under `skills/skills_Nangate45/`. Follow the loop
exactly; the gates and their loop-backs are the point of the workflow. Use
`1.macro_place/SKILL_planner.md` and `SKILL_checker.md` exactly as written —
do not simplify them. Do not omit the timing repair.

## Inputs (set these first)

```
DESIGN      = bp_fe                                   # any design in Benchmark/PlaceReasoner-Bench/
ASPECT      = ar2                                     # ar1 (1:1 core) or ar2 (2:1 core)
DATA_DIR    = Benchmark/PlaceReasoner-Bench/$DESIGN   # <design>.v, constraint.sdc, lef/, lib/, def/
TECH_DIR    = Digital_lib/Nangate45                   # PDK: std-cell lib, tech LEF, cell LEF
FLOORPLAN   = $DATA_DIR/def/2_1_floorplan.def         # ar2;  ar1 -> $DATA_DIR/def/1_1_u60_floorplan.def
CLUSTER     = $DATA_DIR/${DESIGN}_macro_clusters.txt  # macro clusters (or regenerate with dataflow/rtlmp)
WORK_DIR    = work/$DESIGN/$ASPECT                    # scripts, saves_db/, report/, logs
IMAGE_DIR   = $WORK_DIR/images                        # every stage image goes here
RESULT_DIR  = output/macro_loop/$DESIGN/$ASPECT       # final ODB, reports, summary (update in place)
```

Everything below is design-agnostic: only the variables above change
between runs. Benchmark data is not in the repo — fetch it first
(see `Benchmark/README.md`). Macro clusters can be regenerated with
`dataflow/rtlmp` if `$CLUSTER` is missing.

## Operating rules

- **Read before you act.** Read the relevant skill file(s) and any
  image/report you are about to judge before proposing or scripting a change.
- **State the plan before non-trivial edits**, then make the smallest scoped
  change that satisfies the request. Do not revert unrelated changes.
- **Stay fair.** Optimize **only macro placement + halo**. Keep the
  downstream flow (detailed place, CTS, route, repair) at its skill defaults
  — do **not** tune `REPAIR_SETUP_MARGIN` or similar knobs to force timing to
  pass.
- **Record the iteration number** at every gate, and which lever you pulled
  (fine-grain nudge vs coarse-grain replan), so the loop is auditable.
- Use `rg` for search. Keep generated artifacts under `$WORK_DIR`
  (`saves_db/`, `report/`, `generated_scripts/`).
- Full-flow runs need OpenROAD (`openroad` on `$PATH`) and headless image
  rendering needs `xvfb-run`; flag if either is unavailable instead of
  silently skipping a step.

## The loop

```
Info → Plan → Macro place → [Gate A loop] → Detail place → CTS+Route → [Gate B / PPA loop]
```

1. **Info — clusters & names.** Read `$CLUSTER` to get the macro clusters
   and macro list. After step 4 this file is **augmented in place** with each
   macro's pin-side info, so `$CLUSTER` becomes the **single source of truth**
   the Planner reads — do **not** also feed the raw probe dump to the Planner
   (the per-macro `orient=/pins=` line carries everything it needs).
   NOTE ON NAMES: instance names contain **escaped** slashes, e.g.
   `icache_1\/data_mem_banks_0__data_mem_bank\/macro_mem\/mem` (the RTL `/`
   is stored as literal `\/` because `/` is the DB hierarchy divider).
   `findInst` with a plain `/` returns NULL — resolve each plain name to its
   real stored name before placing.

2. **Info — design files.** Netlist, LEF, LIB, SDC are in `$DATA_DIR`; the
   PDK is in `$TECH_DIR`; the floorplan DEF is `$FLOORPLAN`.

3. **Init.** Initialize the design with skill
   `skills/skills_Nangate45/0.init` (`SKILL_init.md`).

4. **Info — floorplan + per-macro pins.** Query die/core geometry per
   `skills/skills_Nangate45/get_die_core_info/SKILL.md`. The same probe also
   emits one line per macro:

   ```
   MACRO #1 name=<resolved-stored-name> master=<cell> w=.. h=.. x=.. y=.. orient=R0 pins=L25,R25,B5,T4
   ```

   where `orient` is the macro orientation and `pins=L<n>,R<n>,B<n>,T<n>` are
   the signal-pin counts on the Left/Right/Bottom/Top edges (the dominant
   edge = the side that should face stdcells).

   **Merge this into `$CLUSTER` (in place), not into a separate file.** For
   each macro line in `$CLUSTER`, append its probe `orient=...` and
   `pins=...` fields so the line becomes, e.g.:

   ```
     clut_mem.clut_mem.u_spram_256x24_composed.u_memory_block_64x24_1  type=memory_block_64x24 size=73.365x216.09um orient=R0 pins=L25,R25,B5,T4
   ```

   Match probe rows to cluster rows by the resolved stored macro name. After
   this step `$CLUSTER` alone tells the Planner: cluster grouping + size +
   orientation + dominant pin side per macro — so step 5 feeds **only**
   `$CLUSTER` (plus the floorplan image), not the full probe log. Keep
   die/core dims (DIE/CORE lines) for your own use in planning the canvas.

5. **Plan + place macros — N candidates in parallel (default N = 5).** Take
   the floorplan image of `$FLOORPLAN` with
   `skills/skills_Nangate45/image_snapshot/SKILL.md`. Then use the planner
   skill `skills/skills_Nangate45/1.macro_place/SKILL_planner.md`.
   **Planner inputs = the augmented `$CLUSTER` (cluster + size + orient +
   pins per macro) plus the floorplan image — nothing else.** Honor its
   planning rules (periphery, pin-side faces stdcell, keep clusters together,
   stdcell region ~AR 1 and contiguous). Generate N candidate placements and
   evaluate them in parallel (one `$WORK_DIR/cand<k>/` each).

6. **Gate A — macro-placement check.** First save the Gate-A image **with
   macro pin-side highlights** using `skills/skills_Nangate45/macro_pin_side`
   (it follows the whole-die-with-pins convention documented in
   `skills/skills_Nangate45/image_snapshot/SKILL.md`); write it to
   `$IMAGE_DIR`. Then run
   `skills/skills_Nangate45/1.macro_place/SKILL_checker.md` → **Gate A**. The
   checker input must include the pin-highlighted layout image.
   - **PASS** → go to step 7.
   - **FAIL** → state the concrete reason for the Planner (use the checker's
     executable feedback format) and go back to **step 5** to refine. Use
     `skills/skills_Nangate45/get_macro_current_location/SKILL.md` to give
     the Planner each macro's current location, and give it the
     `pin_side.png` image for reference. **Budget: 2 refinement attempts.**
     If still failing after 2, record the failure and proceed with the best
     placement so far.

7. **Detailed placement.** Follow `skills/skills_Nangate45/2.detail_place`
   (`skill.md`). Fix issues you hit; do not change parameters unless required.

8. **CTS + route.** Follow `skills/skills_Nangate45/3.post_detail_placement`
   (`skill.md`). Fix issues you hit; do not change parameters unless required.

9. **Gate B — PPA / timing check.** Read PPA (focus on **WNS** and **TNS**)
   with `skills/skills_Nangate45/4.PPA_checker` (`SKILL.md`).
   - **Timing met** (`WNS ≥ 0` and `TNS ≥ 0`, DRC clean) → **DONE**.
   - **Timing missed** → go to step 10.

10. **Optimizer.** Follow `skills/skills_Nangate45/5.optimizer/SKILL.md`:
    refine the **winning** macro placement among the candidates to improve
    PPA, then go back to step 5 with the refined plan. **Budget: 4
    refinement attempts.** After 4, stop and report the best result.

## Outputs / deliverable

- Stage images (Gate-A pin-side image + routed snapshots) in `$IMAGE_DIR`.
- The routed ODB and PPA/DRC reports under `$WORK_DIR/saves_db/` and
  `$WORK_DIR/report/`; copy the winner to `$RESULT_DIR`.
- A final summary (`$RESULT_DIR/summary.md`): per-iteration record (gate
  results, lever pulled, TNS/WNS/power/area/DRC/WL), the winning placement,
  and whether timing was met or the budget was exhausted.
- Runtime breakdown: total wall time, agent time (candidate generation,
  checking, optimization) and OpenROAD runtime.
