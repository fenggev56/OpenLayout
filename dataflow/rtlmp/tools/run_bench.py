#!/usr/bin/env python3
"""Run RTL-MP autoclustering + dataflow on every PlaceReasoner-Bench design,
verify the macro clusters against the RTL, and summarise.

    python3 tools/run_bench.py [--bench DIR] [--out DIR] [--designs a b ...]
                               [--jobs N] [--compare DIR] [--skip-dataflow]

Per design it writes the usual reports into --out plus ``<design>_verify.json``
and prints one summary table (runtime, cluster counts, verifier findings).
With ``--compare DIR`` the macro-cluster membership is diffed against the
``<design>_macro_clusters.txt`` found in DIR (e.g. the benchmark's own files).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RTLMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(RTLMP))
DESIGNS = ["ariane133", "ariane81", "bp_be", "bp_fe", "ethernet",
           "swerv_wrapper", "VeriGPU", "vga_lcd"]


def parse_cluster_txt(path):
    """{frozenset(macro names): cluster name} from a *_macro_clusters.txt."""
    clusters = {}
    cur = None
    members = []
    for line in open(path):
        hm = re.match(r"^\[\s*\d+\]\s+(.+?)\s*$", line)
        if hm:
            if cur is not None and members:
                clusters[frozenset(members)] = cur
            cur, members = hm.group(1), []
            continue
        mm = re.match(r"^\s+(\S+)\s+type=\S+\s+size=", line)
        if mm and cur is not None:
            members.append(mm.group(1))
    if cur is not None and members:
        clusters[frozenset(members)] = cur
    return clusters


def run_design(d, args):
    bench = os.path.join(args.bench, d)
    v = os.path.join(bench, f"{d}.v")
    deff = os.path.join(bench, "def", "1_1_u60_floorplan.def")
    cache = os.path.join(args.out, f"cache_{d}.pkl")
    log = os.path.join(args.out, f"{d}.log")
    cmd = [sys.executable, os.path.join(RTLMP, "run.py"), "--verilog", v,
           "--def", deff, "--design", d, "--out-dir", args.out, "--cache", cache,
           "--directed"]
    if args.skip_dataflow:
        cmd.append("--skip-dataflow")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    t_full = time.time() - t0
    # macro-only report (same clustering, writes the *_macro_clusters.* files)
    cmd2 = [sys.executable, os.path.join(RTLMP, "run.py"), "--verilog", v,
            "--def", deff, "--design", d, "--out-dir", args.out, "--cache", cache,
            "--macro-only"]
    with open(log, "a") as f:
        rc2 = subprocess.call(cmd2, stdout=f, stderr=subprocess.STDOUT)
    lefs = sorted(os.path.join(bench, "lef", x) for x in os.listdir(os.path.join(bench, "lef"))
                  if x.endswith(".lef"))
    vjson = os.path.join(args.out, f"{d}_verify.json")
    cmd3 = [sys.executable, os.path.join(HERE, "verify_clusters.py"), "--verilog", v,
            "--lef", *lefs, "--clusters", os.path.join(args.out, f"{d}_macro_clusters.json"),
            "--json-out", vjson]
    with open(log, "a") as f:
        subprocess.call(cmd3, stdout=f, stderr=subprocess.STDOUT)
    return d, rc or rc2, t_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(ROOT, "Benchmark", "PlaceReasoner-Bench"))
    ap.add_argument("--out", default=os.path.join(RTLMP, "..", "out", "bench"))
    ap.add_argument("--designs", nargs="+", default=DESIGNS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--compare", help="dir with reference <design>_macro_clusters.txt")
    ap.add_argument("--skip-dataflow", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda d: run_design(d, args), args.designs))

    print(f"{'design':14s} {'rc':>2s} {'time':>6s} {'inst':>7s} {'macros':>6s} "
          f"{'std cl':>6s} {'mac cl':>6s} {'sizes':28s} {'dflow':>5s} {'verify':s}")
    for d, rc, t in results:
        log = open(os.path.join(args.out, f"{d}.log")).read()
        jpath = os.path.join(args.out, f"{d}_clustered_netlist.json")
        vpath = os.path.join(args.out, f"{d}_verify.json")
        if not os.path.exists(jpath):
            print(f"{d:14s} {rc:2d} FAILED (see {log})")
            continue
        j = json.load(open(jpath))
        cl = j["clusters"]
        n_std = sum(1 for c in cl if c["type"] == "StdCell")
        mac = sorted((c["num_macro"] for c in cl if c["type"] == "Macro"), reverse=True)
        nd = sum(1 for e in j["virtual_connections"] if e["kind"] == "dataflow")
        ver = json.load(open(vpath)) if os.path.exists(vpath) else {}
        probs = ver.get("problems", [])
        tags = {}
        for tag, _ in probs:
            tags[tag] = tags.get(tag, 0) + 1
        vs = ",".join(f"{k}x{n}" for k, n in sorted(tags.items())) or "ok"
        if ver:
            vs += f" bank-split={ver.get('bank_groups_split')}"
        sizes = str(mac)
        if len(sizes) > 28:
            sizes = sizes[:25] + "..."
        print(f"{d:14s} {rc:2d} {t:6.1f} {j['stats']['instances']:7d} {j['stats']['macros']:6d} "
              f"{n_std:6d} {len(mac):6d} {sizes:28s} {nd:5d} {vs}")
        if args.compare:
            ref = os.path.join(args.compare, f"{d}_macro_clusters.txt")
            if not os.path.exists(ref):
                ref = os.path.join(args.compare, d, f"{d}_macro_clusters.txt")
            new = os.path.join(args.out, f"{d}_macro_clusters.txt")
            if os.path.exists(ref) and os.path.exists(new):
                a = parse_cluster_txt(ref)
                b = parse_cluster_txt(new)
                same = len(set(a) & set(b))
                print(f"{'':14s}    vs {os.path.relpath(ref, ROOT)}: {same}/{len(a)} reference "
                      f"clusters identical, {len(b) - same} new/changed")
                for key in sorted(set(b) - set(a), key=lambda k: -len(k)):
                    print(f"{'':14s}      + {len(key):3d} macros  {b[key][:70]}")
                for key in sorted(set(a) - set(b), key=lambda k: -len(k)):
                    print(f"{'':14s}      - {len(key):3d} macros  {a[key][:70]}")


if __name__ == "__main__":
    main()
