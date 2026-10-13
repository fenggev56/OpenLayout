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
import csv
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
    ap.add_argument("--macro-only", "--macro_only", dest="macro_only",
                    action="store_true",
                    help="Write only macro-cluster membership reports and skip dataflow.")
    ap.add_argument("--large-net-threshold", type=int,
                    help="Override global/broadcast net fanout threshold.")
    ap.add_argument("--infer-hierarchy", choices=("auto", "on", "off"), default="auto",
                    help="Infer module scopes from net names for instances whose "
                         "names carry no hierarchy (auto: only for flat netlists).")
    ap.add_argument("--dataflow-mode", choices=("min", "sum"), default="min",
                    help="Eq.1 accumulation: flow at the shortest register distance "
                         "only (min) or every hop level decayed by 2^hops (sum).")
    ap.add_argument("--no-scipy", action="store_true",
                    help="Force the pure-Python dataflow BFS.")
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
    hier = hierarchy.Hierarchy(nl, infer=args.infer_hierarchy)
    print(f"      {hier.stats()}", file=sys.stderr)

    print("[3/4] autoclustering (Algorithm 1) ...", file=sys.stderr)
    t = time.time()
    ac = Autocluster(nl, hier)
    if args.large_net_threshold:
        ac.p.large_net_threshold = args.large_net_threshold
    print(f"      {ac.p}", file=sys.stderr)
    clusters = ac.run()
    for line in ac.log:
        print(f"      {line}", file=sys.stderr)
    if args.macro_only:
        write_macro_only_reports(nl, ac, design=design, out_dir=args.out_dir)
        print(f"\nDONE in {time.time()-t0:.1f}s -> {args.out_dir}/", file=sys.stderr)
        return

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
        df = Dataflow(nl, ac, mode=args.dataflow_mode, use_scipy=not args.no_scipy)
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


def write_macro_only_reports(nl, ac, design, out_dir):
    """Write macro cluster membership with real macro names and types."""
    stem = safe_stem(design)
    json_path = os.path.join(out_dir, f"{stem}_macro_clusters.json")
    csv_path = os.path.join(out_dir, f"{stem}_macro_clusters.csv")
    txt_path = os.path.join(out_dir, f"{stem}_macro_clusters.txt")

    macro_clusters = []
    rows = []
    macro_in_netlist = []
    for c in sorted((c for c in ac.real_clusters if c.type == MACRO_C),
                    key=lambda c: (c.name, c.id)):
        macros = []
        for iid in sorted(c.inst_ids, key=lambda i: nl.inst_name[i]):
            macro = {
                "real_name": nl.inst_name[iid],
                "type": nl.inst_master[iid],
                "width_um": round(nl.inst_w[iid], 6),
                "height_um": round(nl.inst_h[iid], 6),
                "area_um2": round(nl.inst_area[iid], 6),
            }
            macros.append(macro)
            rows.append({
                "cluster_id": c.id,
                "cluster_name": c.name,
                "cluster_type": c.type,
                "macro_real_name": macro["real_name"],
                "macro_type": macro["type"],
                "width_um": macro["width_um"],
                "height_um": macro["height_um"],
                "area_um2": macro["area_um2"],
            })
        macro_clusters.append({
            "cluster_id": c.id,
            "cluster_name": c.name,
            "cluster_type": c.type,
            "macro_count": len(macros),
            "macro_types": dict(Counter(m["type"] for m in macros)),
            "area_um2": round(sum(m["area_um2"] for m in macros), 6),
            "macros": macros,
        })

    for iid in sorted((i for i, kind in enumerate(nl.inst_kind) if kind == netlist.MACRO),
                      key=lambda i: nl.inst_name[i]):
        cid = ac.inst2cl[iid]
        cl = ac.clusters.get(cid)
        macro_in_netlist.append({
            "real_name": nl.inst_name[iid],
            "type": nl.inst_master[iid],
            "width_um": round(nl.inst_w[iid], 6),
            "height_um": round(nl.inst_h[iid], 6),
            "area_um2": round(nl.inst_area[iid], 6),
            "cluster_id": cid,
            "cluster_name": cl.name if cl else None,
            "cluster_type": cl.type if cl else None,
        })

    out = {
        "design": design,
        "stats": {
            "instances": nl.num_insts,
            "macros": nl.num_macros,
            "macro_clusters": len(macro_clusters),
        },
        "macro_in_netlist": macro_in_netlist,
        "macro_clusters": macro_clusters,
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "cluster_id", "cluster_name", "cluster_type",
            "macro_real_name", "macro_type", "width_um", "height_um", "area_um2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w") as f:
        w = f.write
        w("=" * 78 + "\n")
        w(f"RTL-MP MACRO CLUSTERS  (design: {design})\n")
        w("=" * 78 + "\n\n")
        w(f"{len(macro_clusters)} macro clusters, {len(rows)} macros\n\n")

        w("-" * 78 + "\n")
        w("MACRO_IN_NETLIST\n")
        w("-" * 78 + "\n")
        for macro in macro_in_netlist:
            w(f"  {macro['real_name']}  type={macro['type']} "
              f"cluster=[{macro['cluster_id']}] {macro['cluster_name']}\n")
        w("\n")

        w("-" * 78 + "\n")
        w("MACRO_CLUSTERS\n")
        for cluster in macro_clusters:
            w("-" * 78 + "\n")
            w(f"[{cluster['cluster_id']:3d}] {cluster['cluster_name']}\n")
            w(f"type={cluster['cluster_type']} macro_count={cluster['macro_count']} "
              f"macro_types={cluster['macro_types']}\n")
            for macro in cluster["macros"]:
                w(f"  {macro['real_name']}  type={macro['type']} "
                  f"size={macro['width_um']}x{macro['height_um']}um\n")

    print(f"      wrote {json_path}", file=sys.stderr)
    print(f"      wrote {csv_path}", file=sys.stderr)
    print(f"      wrote {txt_path}", file=sys.stderr)


def write_reports(nl, ac, bundled, df, design, out_dir, directed=False, bundled_dir=None):
    clusters = ac.real_clusters
    pins = ac.pin_clusters
    cname = {c.id: c.name for c in clusters + pins}
    ctype = {c.id: c.type for c in clusters + pins}
    pin_summaries = {c.id: pin_io_summary(nl, c) for c in pins}

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
            entry["io_summary"] = pin_summaries[c.id]
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
        "params": dict(vars(ac.p), dataflow_mode=getattr(df, "mode", None)),
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
            {
                "src": a,
                "dst": b,
                "weight": round(w, 3),
                "src_type": ctype[a],
                "dst_type": ctype[b],
                "src_name": cname[a],
                "dst_name": cname[b],
            }
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
        w("BUNDLED IO PIN CLUSTERS\n")
        w("-" * 78 + "\n")
        for c in pins:
            s = pin_summaries[c.id]
            w(f"  [{c.id:3d}] {c.name:8s} xy=({c.xy[0]:.1f}, {c.xy[1]:.1f}) "
              f"inputs={s['INPUT']:4d} outputs={s['OUTPUT']:4d} "
              f"inouts={s['INOUT']:3d}\n")

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

            w("\nDIRECTED IO FLOW SUMMARY\n")
            w("-" * 78 + "\n")
            w("  Top-level INPUT ports drive their bundled pin clusters into the core.\n")
            w("  Top-level OUTPUT ports are sinks driven by std-cell or macro clusters.\n")
            w("  Edges below are aggregated directed star-model net connections.\n")

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

            w("\nSTRONGEST INPUT-PIN --> CORE CONNECTIONS\n")
            w("-" * 78 + "\n")
            for a, b, wt in directed_pin_edges(nl, pins, pin_summaries, bundled_dir or {},
                                               source_kind="INPUT")[:25]:
                w(f"  {wt:10.1f}  {lbl(a):36s} --> {lbl(b)}\n")

            w("\nSTRONGEST CORE --> OUTPUT-PIN CONNECTIONS\n")
            w("-" * 78 + "\n")
            for a, b, wt in directed_pin_edges(nl, pins, pin_summaries, bundled_dir or {},
                                               sink_kind="OUTPUT")[:25]:
                w(f"  {wt:10.1f}  {lbl(a):36s} --> {lbl(b)}\n")

    print(f"      wrote {json_path}", file=sys.stderr)
    print(f"      wrote {txt_path}", file=sys.stderr)


def safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def pin_io_summary(nl, cluster):
    counts = {"INPUT": 0, "OUTPUT": 0, "INOUT": 0, "OTHER": 0}
    examples = {"INPUT": [], "OUTPUT": [], "INOUT": [], "OTHER": []}
    for bid in cluster.bterm_ids:
        direction = nl.bterm_io[bid]
        if direction not in counts:
            direction = "OTHER"
        counts[direction] += 1
        if len(examples[direction]) < 8:
            examples[direction].append(nl.bterm_name[bid])
    return {
        "INPUT": counts["INPUT"],
        "OUTPUT": counts["OUTPUT"],
        "INOUT": counts["INOUT"],
        "OTHER": counts["OTHER"],
        "examples": examples,
    }


def directed_pin_edges(nl, pins, pin_summaries, directed_edges, source_kind=None, sink_kind=None):
    pin_ids = {c.id for c in pins}
    rows = []
    for (a, b), wt in directed_edges.items():
        if source_kind:
            if a not in pin_ids or pin_summaries[a].get(source_kind, 0) <= 0:
                continue
        if sink_kind:
            if b not in pin_ids or pin_summaries[b].get(sink_kind, 0) <= 0:
                continue
        rows.append((a, b, wt))
    return sorted(rows, key=lambda x: -x[2])


if __name__ == "__main__":
    main()
