---
name: openroad-debugger
description: >
  Generic Debugger checklist for every OpenROAD flow stage: how to read the
  log evidence, which error codes matter, and when a stage should pass or
  fail. Ported from the Innovus OpenLayout debugger skill to OpenROAD.
---

# OpenROAD Stage Debug Checklist

## How to judge a stage
1. **Crash / non-zero exit** → always `fail` / `fatal`.
2. **[ERROR XXX-nnnn] lines** → usually `fail`. Common families:
   - `ODB-`, `ORD-`: database/setup problem (missing lib/lef/def, bad link).
   - `STA-`: constraint problem (unlinked network, missing clock, bad SDC).
   - `GPL-`: global placement diverged (density too high, macro overlap).
   - `DPL-`: detailed placement/legalization failed (no space, overlaps).
   - `PDN-`: power grid generation failed (channels too narrow, ring
     does not fit, unplaced macros).
   - `CTS-`: clock tree problem (no clock root, buffer list, CTS-0114
     overlapping clock nets).
   - `GRT-`: global route congestion (overflow > 0 after all iterations).
   - `DRT-`: detailed route (remaining DRC violations, pin access).
   - `PSM-`: IR-drop analysis (no power terminals, missing vsrc).
3. **Goal-relevant metrics**: setup/hold WNS, TNS, DRC count, utilization.
   Fail only what THIS stage is responsible for:
   - before CTS: ignore hold violations (clocks still ideal);
   - before detailed route: ignore DRC counts;
   - after route: wns_setup < 0, wns_hold < 0 or drc_violations > 0 is fail
     if the goal demands timing_met / drc_clean.
4. **Warnings**: only fail on warnings that predict later failure
   (e.g. repeated `GRT-0008` overflow, `DPL` displacement warnings, many
   unplaced instances).

## What to report
- `issues`: short, concrete, evidence-backed ("GRT total overflow 132 after
  30 iterations", "hold wns -0.12 after CTS").
- `root_cause`: the single most likely cause (e.g. "density too high in
  macro channels").
- `recommendation`: one actionable hint for the Optimizer, tied to a tunable
  parameter of this stage.
