# Benchmark

PlaceReasoner-Bench: 8 open-source designs on Nangate45, each at 60% core
utilization under two core aspect ratios (1:1 and 2:1) — 16 placement tasks.
See [PlaceReasoner-Bench/README.md](PlaceReasoner-Bench/README.md) for the
design table (including top module names), directory layout, and provenance.

## Data is not tracked in git

The design data (netlists, DEFs, LEF/LIB) is roughly **589 MB**, so it is
git-ignored rather than committed — large binaries in git history are
permanent and would slow every clone. Only the documentation is tracked.

Expected layout once the data is in place:

```
Benchmark/PlaceReasoner-Bench/
  <design>/
    <design>.v
    constraint.sdc
    lef/  lib/
    def/1_1_u60_floorplan.def     # ar1
    def/2_1_floorplan.def         # ar2
    <design>_macro_clusters.txt
```

Designs: `ariane133`, `ariane81`, `bp_be`, `bp_fe`, `ethernet`,
`swerv_wrapper`, `vga_lcd`, `VeriGPU`.

The designs derive from the open-source MacroPlacement testcases; the
technology is the Nangate45 open cell library with FakeRAM 2.0 memory macros.
All components are open source, so the benchmark is reproducible without
commercial tools or proprietary PDKs.
