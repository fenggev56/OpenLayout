# Debugger checklist — route

- GRT: "Total overflow: 0" required; `GRT-0116` = failed with congestion.
- DRT: "number of violations = N" per iteration must CONVERGE toward 0.
  Final N is the DRC count. N small (<20) and shrinking -> more iterations
  will clear it. N large or growing -> congestion problem.
- Timing after route: setup/hold WNS >= 0 for the goal.
- `DRT-0073 no access point` -> pin access issue near macro edges.

## Expected noise (do not flag as failures)
- `EST-0026 missing route to pin` and `RSZ-0104 single-pin nets` during the
  post-GRT repair phase are estimation warnings on not-yet-final nets —
  normal, NOT evidence of incomplete routing. Judge routing completeness by
  the detailed-route violation count and DRT completion messages only.
- Useful queries: drc_summary, worst_slacks, setup_paths, hold_paths.
