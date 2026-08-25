# macroflow - structured multi-agent macro-placement flow

`macroflow` is the structured macro-placement system behind `run_flow.py`.
LLM agents make placement decisions; Python handles OpenROAD execution,
legalization, metrics, snapshots, candidate storage and telemetry.

The current runtime is macro-only. It stops after macro placement plus
fast-placement feedback and saves candidate results. It does not run the old
detail-place + route optimization loop.

## Agents

| Agent | Responsibility |
|---|---|
| Planner | Reads the floorplan image, macro list and region grid, then infers visible IO pin locations from the image and proposes a macro-placement strategy. |
| Placer | Converts the strategy into either `#index -> (x, y, orient)` coordinates or a 4x4 region-count distribution. |
| Checker | Inspects OpenROAD-saved layout images at Gate A and Gate B, returns PASS/FAIL, `proceed`, and actionable feedback. |

Every layout image is produced by OpenROAD `save_image` through `ORSession`.
The system does not call generated-image APIs.

## Active Flow

```text
init
  Python renders/sources floorplan.tcl, writes floorplan.odb, saves the initial image,
  and extracts core bounds, macro sizes and pin boxes for image overlay.

macro candidate batch
  for each candidate:
    Planner creates a distinct macro shape/topology strategy.
    Placer places macros by coordinates or region counts.
    Gate A loop:
      Checker judges the macro-placement image.
      If it fails, Placer refines the current macro placement to match the
      original strategy and Checker feedback; Planner is not re-run.
    Gate B loop:
      Python runs fast placement and saves the image as the detailed-placement proxy.
      Checker judges stdcell tightness/density.
      If it fails, Planner revises the strategy and Placer places again.
      Stop on checker proceed or try limit.
    Save candidate coords, gate verdicts, density, images and ODBs.
```

## Parallel Candidates

`flow.macro_candidates` controls how many macro-placement candidates are
generated. `flow.parallel_workers` controls parallelism.

When `parallel_workers > 1`, the parent process first initializes the floorplan
and has the Planner generate all `n` distinct candidate strategies. It then
launches isolated worker processes to place/check/refine those saved strategies.
Each worker has its own work directory, generated scripts, OpenROAD session and
VLM instance. Worker results are merged into the main
`<work-dir>/candidates/` ledger.

This is intentionally process-level parallelism because OpenROAD sessions are
stateful and should not share a live design database.

## Config Knobs

In `config_macro.yaml`:

```yaml
flow:
  macro_candidates: 3
  gate_a_max_refine: 2
  gate_b_max_refine: 4
  parallel_workers: 1
```

The CLI can override these with `--macro-candidates`, `--gate-a-max-refine`,
`--gate-b-max-refine` and `--parallel-workers`.

## Artifacts

By default, outputs are written under `log_debug/`:

```text
log_debug/images/init/                  initial floorplan snapshots
log_debug/images/macro_placed/          macro-placement snapshots
log_debug/images/macro_placed/batches/  per-20-macro placement batch snapshots
log_debug/images/fast_placed/           fast-placement snapshots
log_debug/images/detail_placed/         detailed-placement snapshots
log_debug/images/routed/                routed snapshots
log_debug/saves_db/                     macro and fast-placement ODB checkpoints
log_debug/report/detail_placed/         detail-placement logs
log_debug/report/routed/                route logs
log_debug/candidates/                   merged candidate ledger
log_debug/candidate_workers/            per-worker isolated outputs when parallel
log_debug/model_output/design_data/     extracted die/macro/pin context by refresh stage
log_debug/model_output/planning/        planner strategy batches
log_debug/model_output/candidates/      candidate diagnostics/failures
log_debug/model_output/run/summary.txt
log_debug/model_output/run/telemetry.json
log_debug/_generated_scripts/
log_debug/openroad_session/
```

Each candidate directory contains:

```text
strategy.txt
coords.json
metrics.json
gate_a.json
gate_b.json
macro.png
fast.png
macro.odb
fast.odb
meta.json
```

## Run

```bash
python run_flow.py \
  --data-dir Benchmark/PlaceReasoner-Bench/bp_fe \
  --design bp_fe \
  --top bp_fe_top \
  --work-dir work
```

Self-test:

```bash
python -m macroflow.selftest
```
