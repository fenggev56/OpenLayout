"""Structural invariants of the autoclustering result (Algorithm 2/3).

    python3 tests/test_autocluster_invariants.py <netlist.pkl> [...]
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rtlmp import hierarchy                                   # noqa: E402
from rtlmp.autocluster import MACRO_C, MIXED, PIN, STDCELL, Autocluster  # noqa: E402
from rtlmp.netlist import MACRO                               # noqa: E402


def check(nl):
    hier = hierarchy.Hierarchy(nl)
    ac = Autocluster(nl, hier)
    clusters = ac.run()
    p = ac.p
    # every instance in exactly one real cluster
    seen = Counter()
    for c in clusters:
        for i in c.inst_ids:
            seen[i] += 1
    assert len(seen) == nl.num_insts, (len(seen), nl.num_insts)
    assert max(seen.values()) == 1
    assert all(ac.inst2cl[i] >= 0 for i in range(nl.num_insts))
    # every IO bit in exactly one bundled pin
    assert sorted(b for c in ac.pin_clusters for b in c.bterm_ids) == list(range(len(nl.bterm_name)))
    for c in clusters:
        assert c.type in (STDCELL, MACRO_C), c            # no mixed leaf clusters
        kinds = {nl.inst_kind[i] == MACRO for i in c.inst_ids}
        assert kinds == ({True} if c.type == MACRO_C else {False}), c
        if c.type == MACRO_C:
            assert len({nl.inst_master[i] for i in c.inst_ids}) == 1, c   # one footprint
            if c.origin is not None:          # split from a mixed parent (Step 4)
                assert c.virtual, f"macro cluster {c} has no virtual link to std cells"
        else:
            assert len(c.inst_ids) <= p.max_num_inst, c
            assert len(c.inst_ids) >= p.tiny_cluster_inst or c.name == "<top>/glue_logic", c
    n_mac = sum(1 for c in clusters if c.type == MACRO_C)
    n_std = sum(1 for c in clusters if c.type == STDCELL)
    return n_std, n_mac


def test_invariants():
    import glob
    pkls = sorted(glob.glob(os.environ.get("RTLMP_TEST_PKL_GLOB", "/tmp/rtlmp_*.pkl")))
    if not pkls:
        import pytest
        pytest.skip("no cached netlists found (set RTLMP_TEST_PKL_GLOB)")
    for pkl in pkls:
        with open(pkl, "rb") as f:
            check(pickle.load(f))


if __name__ == "__main__":
    for pkl in sys.argv[1:]:
        with open(pkl, "rb") as f:
            n_std, n_mac = check(pickle.load(f))
        print(f"{os.path.basename(pkl)}: invariants OK ({n_std} std-cell, {n_mac} macro clusters)")
