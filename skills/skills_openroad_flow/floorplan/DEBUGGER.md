# Debugger checklist — floorplan

- "unresolved reference" / ODB link errors -> a macro LEF or the netlist
  module list is incomplete.
- STA "no clocks found" -> SDC did not apply.
- Utilization sanity: 40-75% is workable; > 85% will not place.
- num_macros (OL_FACT) decides whether the macro-placement stage runs.
