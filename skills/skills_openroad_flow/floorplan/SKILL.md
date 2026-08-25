---
name: openroad-floorplan
description: >
  Init/floorplan stage: loading Nangate45 libs + netlist + predefined
  floorplan DEF, and what to verify before the flow proceeds.
---

# OpenROAD Init / Floorplan (predefined DEF)

The flow starts from a predefined floorplan DEF (die/core, rows, IO pins,
initial macro positions). The init stage reads liberty (std cells first,
then every macro .lib), tech LEF, cell LEF, macro LEFs, the netlist, the
DEF and the SDC, then sets Nangate45 layer RC.

## Debug checklist
- "unresolved reference" / ODB link errors → a macro LEF or the netlist
  module list is incomplete.
- STA "no clocks found" → SDC did not apply (read order: after linking).
- Utilization sanity: core utilization 40–75% is workable; > 85% will not
  place.
- num_macros reported by OL_FACT decides whether the macro-placement
  optimization stage runs (0 macros → skipped).
