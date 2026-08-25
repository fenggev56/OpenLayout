# Debugger checklist — detail_place

- `check_placement` must be clean (0 violations); overlaps -> fail.
- GPL divergence (`GPL-` errors, "diverged") -> density too high or macro
  halos blocking too much area.
- `PDN-0179` unable to repair channels -> macro channels too narrow;
  `PDN-0351` ring does not fit -> die margin.
- Setup WNS grossly negative here (< -20% of period) predicts CTS/route
  failure — flag it. Hold is IGNORED before CTS (ideal clocks).
- Many `DPL` warnings / big displacement -> density too high.
- Useful queries: check_placement, worst_slacks, cell_usage.
