# rtlmp - RTL-MP / Hier-RTLMP autoclustering and dataflow

Python implementation of the autoclustering engine (Hier-RTLMP Algorithms
2, 3 and 4) and the dataflow virtual connections (Eq. 1) that turn a flat
gate-level netlist into a clustered netlist for macro placement.

```
run.py              CLI driver, writes <design>_clustered_netlist.json / _clusters.txt
                    and (--macro-only) <design>_macro_clusters.{txt,csv,json}
verilog_netlist.py  structural Verilog + LEF (+ DEF pins/die) front end, pickle cache
netlist.py          compact netlist model (+ legacy OpenDB front end)
hierarchy.py        logical hierarchy from instance names; scope inference from
                    net names for flat netlists (VeriGPU-style inst_N names)
autocluster.py      Steps 1-5: bundled pins, hierarchy break/keep/merge,
                    signature merge, partitioning, mixed split, macro grouping
dataflow.py         Step 6: register-hop BFS, scipy.sparse backend (+ pure Python)
tools/verify_clusters.py   independent RTL-side check of macro clusters
tools/run_bench.py         run + verify all PlaceReasoner-Bench designs, diff vs reference
tests/                     brute-force reference for the dataflow engine, invariants
```

## Run

```bash
cd dataflow
python3 rtlmp/run.py --verilog ../Benchmark/PlaceReasoner-Bench/bp_fe/bp_fe.v \
    --def ../Benchmark/PlaceReasoner-Bench/bp_fe/def/1_1_u60_floorplan.def \
    --design bp_fe --out-dir out --directed          # full flow
python3 rtlmp/run.py --verilog ... --macro-only      # macro cluster files only

python3 rtlmp/tools/run_bench.py --compare ../Benchmark/PlaceReasoner-Bench
python3 rtlmp/tools/verify_clusters.py --verilog D.v --lef lef/*.lef --clusters out/D_macro_clusters.json
python3 rtlmp/tests/test_dataflow_reference.py /tmp/rtlmp_bp_fe.pkl
python3 rtlmp/tests/test_autocluster_invariants.py /tmp/rtlmp_*.pkl
```

Options: `--infer-hierarchy auto|on|off` (scope inference from net names,
auto = only when most instance names are flat), `--dataflow-mode min|sum`
(Eq. 1 at the shortest register distance only, or every hop level decayed by
2^hops), `--no-scipy` (pure-Python BFS, identical results).

## Algorithm notes

* Step 2 follows Algorithm 3 literally: a child is dissolved when
  `num_macro > max_num_macro or num_std > max_num_inst`, is a merge
  candidate when `num_macro < min_num_macro and num_std < min_num_inst`,
  and is kept otherwise.  (A module with exactly `min_num_macro` macros,
  e.g. one cache way with 4 SRAMs, is therefore a cluster of its own.)
* Merge (Algorithm 4): signatures are taken w.r.t. the *reference* clusters
  (kept clusters + bundled pins).  A candidate connected to exactly one
  reference is absorbed into it; candidates with the same signature are
  merged; a candidate left alone, or a merged group still below
  `tiny_cluster_inst` std cells, is attached to the reference it talks to
  most, so no sub-100-cell clusters survive.
* Step 3 partitions only standard cells (BFS region growth + FM refinement);
  the macros of a partitioned cluster stay together.
* Step 5 regroups macros by *near-identical* connection signature
  (Jaccard >= 0.5), evaluated per RTL member module so that bit slices of
  one memory bank are never separated, then by footprint.  Pieces split
  from the same parent count as one neighbour.
* Dataflow: edge weight = number of driver nets of a stage node that reach
  the sink (bounded by the bus width; a path count explodes through
  reconvergent logic).  A single net whose cone reaches more than
  `large_net_threshold` stage nodes is a control signal and is skipped;
  other nets of the same driver are still traced.
* Cluster names: `module(+N)` = N small modules absorbed; `a/b_{0,1,...+5}_c`
  = merged siblings; macro clusters are named after the modules holding the
  macros, with `[master]` appended only when needed to disambiguate.
