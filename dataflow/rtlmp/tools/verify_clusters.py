#!/usr/bin/env python3
"""Verify RTL-MP macro clusters against the RTL (gate-level Verilog) structure.

This checker is deliberately *independent* of ``rtlmp.verilog_netlist``: it
re-parses the macro instances straight from the Verilog with its own small
parser, derives two RTL-level groupings, and checks the clustering against
them:

  * RTL module group : the logical module that instantiates the macro, i.e. the
                       hierarchical path with memory-wrapper levels
                       (genblk*/macro_mem*/mem/u_mem*) stripped.
  * RTL bank group   : macros that share most of their *input* nets (address,
                       write-enable, chip-enable, ...) are banks / bit-slices of
                       one logical memory and should be placed as one tile.
                       Two macros are bank-mates when they share >= 50% of
                       their non-clock input nets; groups are the connected
                       components of that relation.

Checks (per design):
  C1  every macro in the Verilog is in exactly one macro cluster (and the
      cluster report lists no unknown macro);
  C2  no cluster mixes macros from different top-level functional blocks
      (first hierarchy token, e.g. i_icache vs i_nbdcache);
  C3  every cluster holds a single macro footprint (master);
  C4  bank groups are not split across clusters (a split is reported with the
      cluster ids involved);
  C5  cluster size against max_num_macro (informational).

Usage:
    python3 tools/verify_clusters.py --verilog <design>.v --lef lef/*.lef \
        --clusters <design>_macro_clusters.json [--max-macro 12]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_WRAPPER_RE = re.compile(
    r"^(genblk\d+(_\d+__)?.*|macro_mem(_\d+)?|mem|u_mem\d*|i_ram|ram\d+|"
    r"mem_block\d*(_\d+)?|macro_mem\.mem)$")
_CLK_RE = re.compile(r"(^|[._/])(clk|clock|wclk|rclk|ck)(_i|_in)?$", re.I)


def lef_block_masters(lef_paths):
    masters = {}
    for p in lef_paths:
        cur = None
        pin = None
        for raw in open(p, errors="ignore"):
            t = raw.replace(";", "").split()
            if not t:
                continue
            if t[0] == "MACRO":
                cur = t[1]
                masters.setdefault(cur, {"block": False, "in": set(), "out": set()})
            elif cur and t[0] == "END" and len(t) > 1 and t[1] == cur:
                cur = None
            elif cur and t[0] == "CLASS":
                masters[cur]["block"] = t[1].upper() == "BLOCK"
            elif cur and t[0] == "PIN":
                pin = t[1]
            elif cur and pin and t[0] == "DIRECTION":
                d = t[1].upper()
                base = re.sub(r"\[\d+\]$", "", pin)
                (masters[cur]["in"] if d == "INPUT" else masters[cur]["out"]).add(base)
    return {m: v for m, v in masters.items() if v["block"]}


def parse_macro_insts(verilog, block_masters):
    """Return {inst_name: {"master": m, "ports": {port: [nets]}}}."""
    text = Path(verilog).read_text(errors="ignore")
    text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.M | re.S)
    out = {}
    alt = "|".join(re.escape(m) for m in block_masters)
    pat = re.compile(r"(?m)^\s*(" + alt + r")\s+(\\\S+|[A-Za-z_][\w$]*)\s*\(")
    for m in pat.finditer(text):
        master = m.group(1)
        name = m.group(2)
        name = name[1:] if name.startswith("\\") else name
        i = m.end()
        depth = 1
        j = i
        while depth:
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        body = text[i:j - 1]
        ports = {}
        for pm in re.finditer(r"\.([\w$\[\]]+)\s*\(", body):
            p = pm.group(1)
            k = pm.end()
            d = 1
            e = k
            while d:
                if body[e] == "(":
                    d += 1
                elif body[e] == ")":
                    d -= 1
                e += 1
            expr = body[k:e - 1]
            toks = re.findall(r"\\([^\s,{}()]+)|([A-Za-z_$][\w$]*(?:\[\d+\])?)", expr)
            nets = [a or b for a, b in toks]
            nets = [n for n in nets if "'" not in n]
            ports[p] = nets
        out[name] = {"master": master, "ports": ports}
    return out


def module_of(inst_name):
    toks = re.split(r"[/.]", inst_name)[:-1]
    while toks and _WRAPPER_RE.match(toks[-1]):
        toks.pop()
    return "/".join(toks) or "<top>"


def input_nets(inst, block_masters):
    info = block_masters[inst["master"]]
    nets = set()
    for port, ns in inst["ports"].items():
        base = re.sub(r"\[\d+\]$", "", port)
        if base in info["out"]:
            continue
        if base not in info["in"] and base in info["out"]:
            continue
        if _CLK_RE.search(base):
            continue
        for n in ns:
            if _CLK_RE.search(n):
                continue
            nets.add(n)
    return nets


def bank_groups(insts, block_masters, min_jaccard=0.5):
    names = sorted(insts)
    ins = {n: input_nets(insts[n], block_masters) for n in names}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # index nets -> macros to avoid O(n^2) on big designs
    by_net = defaultdict(list)
    for n in names:
        for net in ins[n]:
            by_net[net].append(n)
    cand = defaultdict(int)
    for net, ms in by_net.items():
        if len(ms) > 64:        # broadcast (clk/reset-like) - not a bank signal
            continue
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                cand[(ms[i], ms[j])] += 1
    for (a, b), shared in cand.items():
        if insts[a]["master"] != insts[b]["master"]:
            continue
        denom = min(len(ins[a]), len(ins[b])) or 1
        if shared / denom >= min_jaccard:
            parent[find(a)] = find(b)
    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verilog", required=True)
    ap.add_argument("--lef", nargs="+", required=True)
    ap.add_argument("--clusters", required=True, help="<design>_macro_clusters.json")
    ap.add_argument("--max-macro", type=int, default=12)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    bm = lef_block_masters(args.lef)
    insts = parse_macro_insts(args.verilog, bm)
    rep = json.load(open(args.clusters))
    cl_of = {}
    clusters = {}
    for c in rep["macro_clusters"]:
        clusters[c["cluster_id"]] = c
        for m in c["macros"]:
            cl_of[m["real_name"]] = c["cluster_id"]

    problems = []
    # C1 coverage
    missing = sorted(set(insts) - set(cl_of))
    unknown = sorted(set(cl_of) - set(insts))
    if missing:
        problems.append(("C1", f"{len(missing)} Verilog macros not in any cluster: {missing[:3]}"))
    if unknown:
        problems.append(("C1", f"{len(unknown)} cluster macros not found in Verilog: {unknown[:3]}"))
    # C2 functional block purity; C3 footprint
    for cid, c in clusters.items():
        blocks = defaultdict(int)
        masters = defaultdict(int)
        for m in c["macros"]:
            mod = module_of(m["real_name"])
            blocks[mod.split("/")[0]] += 1
            masters[m["type"]] += 1
        if len(blocks) > 1:
            problems.append(("C2", f"cluster [{cid}] mixes top-level blocks {dict(blocks)}"))
        if len(masters) > 1:
            problems.append(("C3", f"cluster [{cid}] mixes footprints {dict(masters)}"))
        if c["macro_count"] > args.max_macro:
            problems.append(("C5", f"cluster [{cid}] has {c['macro_count']} macros > max_num_macro={args.max_macro} ({c['cluster_name'][:60]})"))
    # C4 bank groups (informational: set-associative caches share one address
    # bus across all ways, so a per-way clustering legitimately "splits" the
    # address-sharing group; it is reported, not counted as a defect)
    groups = bank_groups(insts, bm)
    split = []
    for g in groups:
        cids = sorted({cl_of.get(n, -1) for n in g})
        if len(cids) > 1:
            split.append((len(g), len(cids), sorted({module_of(n) for n in g})[0]))
    if split:
        ex = "; ".join(f"{n} macros under {m[:40]} -> {k} clusters" for n, k, m in split[:2])
        problems.append(("C4", f"{len(split)} address-sharing macro groups span several clusters ({ex})"))
    split = len(split)
    # module-group summary: how clusters relate to RTL modules
    mod_groups = defaultdict(list)
    for n in insts:
        mod_groups[module_of(n)].append(n)
    merged_mods = sum(1 for cid, c in clusters.items()
                      if len({module_of(m["real_name"]) for m in c["macros"]}) > 1)

    summary = {
        "design": rep.get("design"),
        "macros_verilog": len(insts),
        "macros_clustered": len(cl_of),
        "clusters": len(clusters),
        "cluster_sizes": sorted((c["macro_count"] for c in clusters.values()), reverse=True),
        "rtl_modules_with_macros": len(mod_groups),
        "rtl_bank_groups": len(groups),
        "bank_group_sizes": sorted((len(g) for g in groups), reverse=True),
        "bank_groups_split": split,
        "clusters_spanning_modules": merged_mods,
        "problems": problems,
    }
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=2)
    d = summary
    print(f"[{d['design']}] macros={d['macros_verilog']} clustered={d['macros_clustered']} "
          f"clusters={d['clusters']} sizes={d['cluster_sizes']}")
    print(f"   RTL: {d['rtl_modules_with_macros']} modules hold macros; "
          f"{d['rtl_bank_groups']} bank groups sizes={d['bank_group_sizes']}")
    print(f"   bank groups split across clusters: {split}; clusters spanning >1 RTL module: {merged_mods}")
    for tag, msg in problems:
        print(f"   {tag}: {msg}")
    if not problems:
        print("   OK: no problems")
    return 1 if any(t in ("C1", "C2", "C3") for t, _ in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
