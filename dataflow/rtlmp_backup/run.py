"""RTL-MP autoclustering driver.

Examples:

    python3 rtlmp/run.py \
      --verilog Benchmark/PlaceReasoner-Bench/bp_fe/bp_fe.v \
      --skip-dataflow

    openroad -python rtlmp/run.py

With no ``--verilog`` argument the legacy OpenDB/ariane81 path is used.  With a
Verilog input, sibling ``lef/*.lef`` and ``def/pre_place.def`` files are
discovered automatically unless explicit ``--lef`` or ``--def`` arguments are
provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtlmp import hierarchy, netlist, verilog_netlist          # noqa: E402
from rtlmp.autocluster import Autocluster, MACRO_C, PIN, STDCELL  # noqa: E402
from rtlmp.dataflow import Dataflow                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "out")


def parse_args():
    ap = argparse.ArgumentParser(description="Run RTL-MP autoclustering.")
    ap.add_argument("--verilog", help="Structural Verilog netlist.")
    ap.add_argument("--lef", action="append", default=[],
                    help="LEF file with cell/macro metadata. Repeatable.")
    ap.add_argument("--def", dest="def_file",
                    help="Optional DEF for die and IO-pin locations.")
    ap.add_argument("--design", help="Design name for reports.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="Report output directory.")
    ap.add_argument("--cache", help="Pickle cache path.")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="Ignore and refresh the pickle cache.")
    ap.add_argument("--directed", action="store_true",
                    help="Emit directed real/dataflow edge views.")
    ap.add_argument("--skip-dataflow", action="store_true",
                    help="Run Algorithm 1 clustering only.")
    ap.add_argument("--large-net-threshold", type=int,
                    help="Override global/broadcast net fanout threshold.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("[1/4] loading netlist ...", file=sys.stderr)
    nl, design = load_input(args)
    print(f"      design={design}, {nl.num_insts} insts "
          f"({nl.num_macros} macros), {len(nl.bterm_name)} IO bits, "
          f"{len(nl.net_name)} nets, {time.time()-t0:.1f}s", file=sys.stderr)

    print("[2/4] reconstructing logical hierarchy ...", file=sys.stderr)
    hier = hierarchy.Hierarchy(nl)
    print(f"      {hier.stats()}", file=sys.stderr)

    print("[3/4] autoclustering (Algorithm 1) ...", file=sys.stderr)
    t = time.time()
    ac = Autocluster(nl, hier)
    if args.large_net_threshold:
        ac.p.large_net_threshold = args.large_net_threshold
    print(f"      {ac.p}", file=sys.stderr)
    clusters = ac.run()
    bundled = ac.bundled_connections()
    bundled_dir = ac.bundled_connections(directed=True) if args.directed else None
    print(f"      {len(clusters)} clusters + {len(ac.pin_clusters)} bundled "
          f"pins, {len(bundled)} bundled connections"
          f"{' (directed mode)' if args.directed else ''}, "
          f"{time.time()-t:.1f}s", file=sys.stderr)

    if args.skip_dataflow:
        print("[4/4] dataflow skipped by request", file=sys.stderr)
        df = empty_dataflow()
    else:
        print("[4/4] dataflow virtual connections (Eq. 1) ...", file=sys.stderr)
        t = time.time()
        df = Dataflow(nl, ac)
        df.run()
        print(f"      {len(df.virtual_edges)} dataflow edges, "
              f"hops={dict(sorted(df.hop_hist.items()))}, "
              f"{time.time()-t:.1f}s", file=sys.stderr)

    write_reports(
        nl, ac, bundled, df, design=design, out_dir=args.out_dir,
        directed=args.directed, bundled_dir=bundled_dir,
    )
    print(f"\nDONE in {time.time()-t0:.1f}s -> {args.out_dir}/", file=sys.stderr)


def load_input(args):
    if args.verilog:
        lefs = args.lef or verilog_netlist.discover_lefs(args.verilog)
        def_file = args.def_file or verilog_netlist.discover_def(args.verilog)
        cache = args.cache
        if cache is None:
            cache = f"/tmp/rtlmp_{Path(args.verilog).stem}.pkl"
        print(f"      Verilog: {args.verilog}", file=sys.stderr)
        print(f"      LEFs: {len(lefs)} file(s)", file=sys.stderr)
        if def_file:
            print(f"      DEF geometry: {def_file}", file=sys.stderr)
        nl = verilog_netlist.load(
            args.verilog,
            lef_paths=lefs,
            def_path=def_file,
            cache_path=cache,
            rebuild=args.rebuild_cache,
        )
        design = args.design or getattr(nl, "design_name", None) or Path(args.verilog).stem
        return nl, design

    cache = args.cache or "/tmp/ariane81.pkl"
    nl = netlist.load(cache_path=cache, rebuild=args.rebuild_cache)
    return nl, args.design or "ariane81"


def empty_dataflow():
    return SimpleNamespace(
        virtual_edges={},
        virtual_edges_directed={},
        hop_hist={},
    )


def write_reports(nl, ac, bundled, df, design, out_dir, directed=False, bundled_dir=None):
    clusters = ac.real_clusters
    pins = ac.pin_clusters
    cname = {c.id: c.name for c in clusters + pins}
    ctype = {c.id: c.type for c in clusters + pins}

    def lbl(cid):
        return f"{ctype[cid]:7s}:{cname[cid][:28]}"

    jclusters = []
    for c in clusters + pins:
        entry = {
            "id": c.id,
            "name": c.name,
            "type": c.type,
            "num_inst": len(c.inst_ids),
            "num_macro": c.GetNumMacro(nl) if c.type != PIN else 0,
            "area_um2": round(c.area(nl), 3) if c.type != PIN else 0.0,
        }
        if c.type == PIN:
            entry["xy"] = c.xy
            entry["io_pins"] = len(c.bterm_ids)
        if c.type == MACRO_C:
            entry["macro_masters"] = dict(Counter(nl.inst_master[i] for i in c.inst_ids))
        jclusters.append(entry)

    real_edges = [{"a": a, "b": b, "weight": round(w, 3)}
                  for (a, b), w in sorted(bundled.items(), key=lambda kv: -kv[1])]
    virt_edges = [{"a": a, "b": b, "weight": round(w, 3), "kind": "dataflow"}
                  for (a, b), w in sorted(df.virtual_edges.items(), key=lambda kv: -kv[1])]

    seen = set()
    for c in clusters:
        for nb, w in c.virtual.items():
            if (c.id, nb) in df.virtual_edges or (nb, c.id) in df.virtual_edges:
                continue
            key = (min(c.id, nb), max(c.id, nb))
            if key in seen:
                continue
            seen.add(key)
            virt_edges.append({
                "a": key[0], "b": key[1],
                "weight": round(w, 3), "kind": "macro_stdcell",
            })

    out = {
        "design": design,
        "directed": directed,
        "params": vars(ac.p),
        "stats": {
            "instances": nl.num_insts,
            "macros": nl.num_macros,
            "io_bits": len(nl.bterm_name),
            "nets": len(nl.net_name),
            "clusters": len(clusters),
            "bundled_pins": len(pins),
        },
        "clusters": jclusters,
        "real_connections": real_edges,
        "virtual_connections": virt_edges,
    }
    if directed:
        out["real_connections_directed"] = [
            {"src": a, "dst": b, "weight": round(w, 3)}
            for (a, b), w in sorted((bundled_dir or {}).items(), key=lambda kv: -kv[1])
        ]
        out["virtual_connections_directed"] = [
            {"src": a, "dst": b, "weight": round(w, 3), "kind": "dataflow"}
            for (a, b), w in sorted(df.virtual_edges_directed.items(), key=lambda kv: -kv[1])
        ]

    stem = safe_stem(design)
    json_path = os.path.join(out_dir, f"{stem}_clustered_netlist.json")
    txt_path = os.path.join(out_dir, f"{stem}_clusters.txt")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    with open(txt_path, "w") as f:
        w = f.write
        w("=" * 78 + "\n")
        w(f"RTL-MP CLUSTERED NETLIST  (design: {design})\n")
        w("=" * 78 + "\n\n")
        w(f"Hyperparameters: {ac.p}\n\n")
        n_std = sum(1 for c in clusters if c.type == STDCELL)
        n_mac = sum(1 for c in clusters if c.type == MACRO_C)
        w(f"{len(clusters)} clusters  ({n_std} std-cell, {n_mac} macro)  "
          f"+ {len(pins)} bundled IO pins\n\n")

        w("-" * 78 + "\n")
        w("MACRO CLUSTERS (grouped by footprint -> regular tilings)\n")
        w("-" * 78 + "\n")
        for c in sorted(clusters, key=lambda c: -c.GetNumMacro(nl)):
            if c.type != MACRO_C:
                continue
            masters = dict(Counter(nl.inst_master[i] for i in c.inst_ids))
            w(f"  [{c.id:3d}] {c.name[:52]:52s} macros={len(c.inst_ids):3d} "
              f"{masters}\n")

        w("\n" + "-" * 78 + "\n")
        w("STD-CELL CLUSTERS (largest 20)\n")
        w("-" * 78 + "\n")
        for c in sorted((c for c in clusters if c.type == STDCELL),
                        key=lambda c: -len(c.inst_ids))[:20]:
            w(f"  [{c.id:3d}] {c.name[:54]:54s} inst={len(c.inst_ids):6d} "
              f"area={c.area(nl):9.1f}um2\n")

        w("\n" + "-" * 78 + "\n")
        w("STRONGEST DATAFLOW VIRTUAL CONNECTIONS  (Info_Flow / 2^hops)\n")
        w("-" * 78 + "\n")
        for e in virt_edges[:25]:
            if e["kind"] != "dataflow":
                continue
            w(f"  {e['weight']:14.2f}  {ctype[e['a']]:7s}:{cname[e['a']][:28]:28s}"
              f" <-> {ctype[e['b']]:7s}:{cname[e['b']][:28]}\n")

        w("\n" + "-" * 78 + "\n")
        w("STRONGEST REAL (BUNDLED) CONNECTIONS\n")
        w("-" * 78 + "\n")
        for e in real_edges[:25]:
            w(f"  {e['weight']:10.1f}  {ctype[e['a']]:7s}:{cname[e['a']][:28]:28s}"
              f" <-> {ctype[e['b']]:7s}:{cname[e['b']][:28]}\n")

        w("\n" + "-" * 78 + "\n")
        w("MACRO <-> STD-CELL VIRTUAL CONNECTIONS (Step 4, weight=virtual_weight)\n")
        w("-" * 78 + "\n")
        for e in virt_edges:
            if e["kind"] != "macro_stdcell":
                continue
            w(f"  {e['weight']:8.1f}  {ctype[e['a']]:7s}:{cname[e['a']][:28]:28s}"
              f" <-> {ctype[e['b']]:7s}:{cname[e['b']][:28]}\n")

        if directed:
            w("\n" + "=" * 78 + "\n")
            w("DIRECTED VIEW  (driver-cluster --> sink-cluster, dataflow direction)\n")
            w("=" * 78 + "\n")

            w("\nSTRONGEST DIRECTED DATAFLOW CONNECTIONS\n")
            w("-" * 78 + "\n")
            for (a, b), wt in sorted(df.virtual_edges_directed.items(),
                                     key=lambda kv: -kv[1])[:25]:
                w(f"  {wt:14.2f}  {lbl(a):36s} --> {lbl(b)}\n")

            w("\nSTRONGEST DIRECTED REAL (BUNDLED) CONNECTIONS\n")
            w("-" * 78 + "\n")
            for (a, b), wt in sorted((bundled_dir or {}).items(),
                                     key=lambda kv: -kv[1])[:25]:
                w(f"  {wt:10.1f}  {lbl(a):36s} --> {lbl(b)}\n")

    print(f"      wrote {json_path}", file=sys.stderr)
    print(f"      wrote {txt_path}", file=sys.stderr)


def safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


if __name__ == "__main__":
    main()
