# Debugger checklist — cts

- `CTS-0114` clock net overlap -> a clock-root net was optimized away
  upstream (dont_touch missing).
- "No clock nets have been found" -> SDC not applied after read_db.
- `DPL-0036` after hold repair -> hold buffers could not be legalized
  (no free sites near their spot).
- Clock skew: for a small design expect < ~15% of the clock period.
- Hold WNS after repair should be >= 0. Setup WNS should not degrade much
  vs placement; a big drop means the tree is too deep.
- check_placement must be clean after buffer legalization.
- Useful queries: clock_skew, worst_slacks, setup_paths, check_placement.
