---
name: openroad-ir-drop
description: >
  IR-drop analysis with OpenROAD psm (analyze_power_grid): setup, source
  types, reading the voltage files, and typical failure modes. Adapted from
  the PV_agent run_ir_drop_real.tcl template.
---

# IR drop with analyze_power_grid

```tcl
set_pdnsim_net_voltage -net VDD -voltage 1.1   ;# Nangate45 nominal
set_pdnsim_net_voltage -net VSS -voltage 0.0
analyze_power_grid -net VDD -source_type FULL \
    -voltage_file vdd_voltage.rpt -error_file vdd_pg_error.rpt
analyze_power_grid -net VSS -source_type FULL \
    -voltage_file vss_voltage.rpt -error_file vss_pg_error.rpt
```

- IR drop can only be measured AFTER routing (needs the real PDN + switching
  power from the routed design).
- `-source_type FULL` places ideal sources over the whole top grid;
  `STRAPS`/`BUMPS` model fewer injection points (more pessimistic). If the
  DEF has no power terminals, `check_power_grid` fails (PSM-0025) — skip it
  and rely on the source_type model, or provide a `-vsrc` C4-bump file.
- Voltage file: per-node voltages; worst VDD drop = nominal - min(VDD file);
  worst VSS bounce = max(VSS file).
- Healthy: worst drop < 5% of VDD. 5–10%: check pdn_pitch. >10%: PDN too
  sparse or a connectivity gap (look at the pg_error file).
