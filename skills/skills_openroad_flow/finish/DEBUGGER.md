# Debugger checklist — finish

- setup WNS >= goal max_wns AND hold WNS >= 0 -> timing met.
- drc_violations == 0 -> DRC clean (count from the route stage DRC report).
- IR drop: worst VDD drop = vdd_voltage - min(voltage file). Rule of thumb:
  < 5% of VDD (Nangate45: < 55 mV) is healthy.
- PSM errors about missing power terminals are analysis-setup issues, not
  design issues — try another ir_source_type before blaming the PDN.
- Useful queries: worst_slacks, setup_paths, drc_summary, power.
