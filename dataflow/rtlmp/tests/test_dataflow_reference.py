"""Cross-check the optimised dataflow engine against a brute-force reference.

The reference implementation below is deliberately naive (no memoisation, no
pruning, no sparse algebra): for every macro / IO source it walks the netlist
through combinational cells with a plain DFS and BFS-es over register stages
counting hops.  Its only job is to be obviously correct so that the optimised
``rtlmp.dataflow.Dataflow`` (scipy and pure-Python backends) can be compared
against it edge-for-edge.

    python3 -m pytest tests/ -q
    python3 tests/test_dataflow_reference.py <netlist.pkl> [...]
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rtlmp import hierarchy                                   # noqa: E402
from rtlmp.autocluster import Autocluster                     # noqa: E402
from rtlmp.dataflow import Dataflow                           # noqa: E402
from rtlmp.netlist import COMB, INST, MACRO, REGISTER, Netlist  # noqa: E402


def reference_virtual_edges(nl: Netlist, ac, max_hops=4, max_fanout=500,
                            large_net=500, mode="min"):
    """Brute-force Eq. (1) on the same Netlist / clustering."""
    N = nl.num_insts
    NB = len(nl.bterm_name)
    kind = nl.inst_kind + ["BTERM"] * NB

    def is_comb(v):
        return kind[v] == COMB

    def is_reg(v):
        return kind[v] == REGISTER

    def is_dest(v):
        return v >= N or kind[v] == MACRO

    out_nets = defaultdict(list)
    sinks_of = {}
    for nidx, pins in enumerate(nl.net_pins):
        if len(pins) > large_net:
            continue
        drv = None
        sk = []
        for k, eid, isd in pins:
            node = eid if k == INST else N + eid
            if isd:
                drv = node
            else:
                sk.append(node)
        if drv is None or not sk:
            continue
        out_nets[drv].append(nidx)
        sinks_of[nidx] = sk

    def net_reach(nidx):
        """Stage nodes reachable from one net through comb cells (DFS)."""
        reach = set()
        seen = set()
        stack = list(sinks_of[nidx])
        while stack:
            v = stack.pop()
            if is_comb(v):
                if v in seen:
                    continue
                seen.add(v)
                for n2 in out_nets.get(v, ()):
                    stack.extend(sinks_of[n2])
            else:
                reach.add(v)
            if len(reach) > max_fanout:
                return None
        return reach

    # stage graph: weight = number of driver nets of u reaching v
    fwd = {}
    for u in list(out_nets):
        if is_comb(u):
            continue
        w = defaultdict(int)
        for nidx in out_nets[u]:
            r = net_reach(nidx)
            if r is None:
                continue
            for v in r:
                w[v] += 1
        if w:
            fwd[u] = dict(w)

    def cluster_of(v):
        return ac.bterm2cl.get(v - N, -1) if v >= N else ac.inst2cl[v]

    sources = [i for i in range(N) if kind[i] == MACRO] + [N + b for b in range(NB)]
    edges = defaultdict(float)
    for s in sources:
        cs = cluster_of(s)
        if cs < 0:
            continue
        # BFS over registers, unit cost per register relay
        dist = {s: 0}
        q = deque([s])
        found = defaultdict(dict)        # dst -> {hop: flow}
        while q:
            u = q.popleft()
            r = dist[u]
            for v, w in fwd.get(u, {}).items():
                if is_dest(v) and v != s:
                    found[v][r] = found[v].get(r, 0.0) + w
                if is_reg(v) and r + 1 <= max_hops and v not in dist:
                    dist[v] = r + 1
                    q.append(v)
        for v, hm in found.items():
            cv = cluster_of(v)
            if cv < 0 or cv == cs:
                continue
            if mode == "min":
                h = min(hm)
                edges[(cs, cv)] += hm[h] / 2 ** h
            else:
                edges[(cs, cv)] += sum(f / 2 ** h for h, f in hm.items())
    return dict(edges)


def _load(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def compare(pkl, mode="min"):
    nl = _load(pkl)
    hier = hierarchy.Hierarchy(nl)
    ac = Autocluster(nl, hier)
    ac.run()
    ref = reference_virtual_edges(nl, ac, mode=mode)

    results = {}
    for backend in ("scipy", "python"):
        for c in ac.clusters.values():
            c.virtual.clear()
        df = Dataflow(nl, ac, mode=mode, use_scipy=(backend == "scipy"))
        df.run(verbose=False)
        results[df.backend] = df.virtual_edges_directed
    for name, got in results.items():
        assert set(got) == set(ref), (name, len(got), len(ref),
                                      sorted(set(got) ^ set(ref))[:5])
        for k in ref:
            assert abs(got[k] - ref[k]) <= 1e-6 * max(1.0, abs(ref[k])), (name, k, got[k], ref[k])
    return len(ref), sorted(results)


def test_dataflow_matches_reference():
    import glob
    pkls = sorted(glob.glob(os.environ.get("RTLMP_TEST_PKL_GLOB", "/tmp/rtlmp_*.pkl")))
    if not pkls:
        import pytest
        pytest.skip("no cached netlists found (set RTLMP_TEST_PKL_GLOB)")
    for pkl in pkls[:2]:
        for mode in ("min", "sum"):
            n, backends = compare(pkl, mode)
            assert n > 0 and backends


if __name__ == "__main__":
    for pkl in sys.argv[1:]:
        for mode in ("min", "sum"):
            n, backends = compare(pkl, mode)
            print(f"{os.path.basename(pkl)} mode={mode}: {n} directed edges match "
                  f"the brute-force reference for backends {backends}")
