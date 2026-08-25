# PlaceReasoner-Bench

An end-to-end macro-placement benchmark built entirely from open-source RTL, an
open-source EDA flow, and an open technology node. Eight designs are each
evaluated at 60% core utilization under two core aspect ratios (1:1 and 2:1),
giving 16 placement tasks.

Data download: **https://huggingface.co/datasets/fenggev566/PlaceReasoner-Bench**
(see [`../README.md`](../README.md) — the design files are not tracked in git).

## Designs

| Design | Top module | #Macros | #Clusters | #Types | Clk (ns) |
|---|---|---:|---:|---:|---:|
| ariane133      | `ariane`         | 133 | 14 | 1 | 4.0 |
| ariane81       | `ariane`         |  81 |  8 | 6 | 4.0 |
| bp_be          | `bp_be_top`      |  10 |  3 | 3 | 2.6 |
| bp_fe          | `bp_fe_top`      |  11 |  4 | 3 | 1.8 |
| ethernet       | `eth_top`        |  64 |  2 | 1 | 8.0 |
| swerv_wrapper  | `swerv_wrapper`  |  28 |  3 | 3 | 2.0 |
| vga_lcd        | `vga_enh_top`    |  62 | 12 | 3 | 2.0 |
| VeriGPU        | `gpu_card`       |  12 |  1 | 1 | 1.2 |
| **Total**      |                  | **401** | **47** | -- | -- |

`#Types` counts distinct macro types; `Clk` is the target clock period.

`Top module` is the design top cell name, as carried by both the `DESIGN` statement
in each DEF and the single (flattened) module in `<design>.v` -- use it for
`link_design` / `-top` arguments. Note it differs from the directory name for all
but `swerv_wrapper`, and that `ariane133` and `ariane81` share the top name
`ariane`, so they must be kept in separate run directories.

## Layout

Each design directory is self-contained:

```
<design>/
  <design>.v                        gate-level netlist (flattened; top module per table above)
  constraint.sdc                    timing constraints
  lef/                              tech LEF + macro LEFs (Nangate45, FakeRAM 2.0)
  lib/                              timing libraries
  def/
    1_1_u60_floorplan.def           floorplan, aspect ratio 1:1, 60% utilization
    2_1_floorplan.def               floorplan, aspect ratio 2:1, 60% utilization
  <design>_macro_clusters.txt       macro cluster assignment
```

The two DEFs differ only in core geometry; both fix the die/core outline and IO
pin placement, and leave macro locations to be determined. A placement method
consumes the DEF plus the LEF/LIB/SDC and emits macro coordinates and
orientations; everything downstream (standard-cell placement, CTS, routing,
timing repair) is held identical across methods so that post-route differences
are attributable to macro placement alone.

`<design>_macro_clusters.txt` gives a connectivity-based grouping of macros,
used as a soft placement constraint that encourages dataflow locality.

## Provenance

- **Designs and flow scaffolding** derive from the open-source MacroPlacement testcases.
- **Technology**: Nangate45 open cell library (`NangateOpenCellLibrary.*`).
- **Memory macros**: FakeRAM 2.0 (`fakeram45_*`).
- **Reference flow**: Yosys for synthesis, OpenROAD for place-and-route.

All components are open source and redistributable, so the benchmark is fully
reproducible without commercial tools or proprietary PDKs.
