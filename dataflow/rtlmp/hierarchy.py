"""Reconstruct the logical hierarchy from flattened instance names.

The synthesized netlists are a single flat module, but the original RTL
hierarchy survives in the instance path names, e.g.

    i_cache_subsystem/i_icache/sram_block_0__data_sram/genblk1_0__i_ram/macro_mem_0

Both '/' (DEF DIVIDERCHAR) and '.' (SystemVerilog struct/interface scopes
that survived flattening) act as hierarchy separators here.  We split on
both to rebuild the module tree the Autoclustering Engine traverses in a
top-down manner (Algorithm 3).

Some flows (e.g. the VeriGPU testcase) rename every instance to ``inst_N``
and keep the hierarchy only in the *net* names
(``gpu_die_/core_array[0].core_inst/_00480_``).  For such netlists the
hierarchy is inferred per instance by a majority vote over the scope of the
nets on its pins (``infer="auto"`` enables this only when most instance
names carry no scope at all).

Each :class:`Module` exposes ``GetNumInst``, ``GetNumMacro`` and
``GetNumStdCell`` matching the notation used in the paper's pseudo-code.
"""

from __future__ import annotations

import re
from collections import Counter

from .netlist import BTERM, INST, MACRO, Netlist

_SPLIT_RE = re.compile(r"[/.]")


def split_path(name: str) -> list[str]:
    """Hierarchy tokens of an instance / module path ('<top>' -> [])."""
    if not name or name == "<top>":
        return []
    return [t for t in _SPLIT_RE.split(name) if t != ""]


class Module:
    """A node in the reconstructed logical hierarchy."""

    __slots__ = ("name", "path", "parent", "children", "leaves",
                 "_num_inst", "_num_macro")

    def __init__(self, name: str, path: str, parent: "Module | None"):
        self.name = name                  # local name within parent
        self.path = path                  # full hierarchical path
        self.parent = parent
        self.children: dict[str, "Module"] = {}
        self.leaves: list[int] = []       # leaf instance ids directly here
        self._num_inst = -1               # memoized subtree counts
        self._num_macro = -1

    # -- paper-style accessors ---------------------------------------- #
    def GetNumInst(self) -> int:          # noqa: N802 (match paper notation)
        return self._num_inst

    def GetNumMacro(self) -> int:         # noqa: N802
        return self._num_macro

    def GetNumStdCell(self) -> int:       # noqa: N802
        return self._num_inst - self._num_macro

    def has_children(self) -> bool:
        return bool(self.children)

    def subtree_leaves(self):
        """Yield all leaf instance ids in this subtree."""
        stack = [self]
        while stack:
            m = stack.pop()
            yield from m.leaves
            stack.extend(m.children.values())

    def __repr__(self):
        return (f"<Module {self.path or '<top>'} "
                f"inst={self._num_inst} macro={self._num_macro}>")


class Hierarchy:
    def __init__(self, nl: Netlist, infer: str = "auto"):
        """``infer``: 'auto' | 'on' | 'off' - infer scopes from net names for
        instances whose own names carry no hierarchy."""
        self.nl = nl
        self.root = Module("", "", None)
        self.inferred = 0
        paths = self._instance_paths(infer)
        self._build(paths)
        self._annotate(self.root)

    # ------------------------------------------------------------------ #
    def _instance_paths(self, infer: str) -> list[list[str]]:
        nl = self.nl
        paths = [split_path(name)[:-1] for name in nl.inst_name]
        if infer == "off":
            return paths
        flat = [i for i, p in enumerate(paths) if not p]
        if infer == "auto" and len(flat) * 2 < nl.num_insts:
            return paths                   # names already hierarchical
        if not flat:
            return paths
        # majority vote over the scopes of the nets on each flat instance
        votes: list[Counter | None] = [None] * nl.num_insts
        flat_set = set(flat)
        for nidx, pins in enumerate(nl.net_pins):
            scope = split_path(nl.net_name[nidx])[:-1]
            if not scope:
                continue
            key = "/".join(scope)
            for kind, eid, _ in pins:
                if kind == INST and eid in flat_set:
                    c = votes[eid]
                    if c is None:
                        c = votes[eid] = Counter()
                    c[key] += 1
        for i in flat:
            c = votes[i]
            if not c:
                continue
            best, _ = c.most_common(1)[0]
            paths[i] = best.split("/")
            self.inferred += 1
        return paths

    def _build(self, paths):
        root = self.root
        for iid, toks in enumerate(paths):
            node = root
            for depth, tok in enumerate(toks):
                child = node.children.get(tok)
                if child is None:
                    child = Module(tok, "/".join(toks[:depth + 1]), node)
                    node.children[tok] = child
                node = child
            node.leaves.append(iid)

    def _annotate(self, m: Module) -> tuple[int, int]:
        """Post-order fill of subtree instance / macro counts (iterative)."""
        kind = self.nl.inst_kind
        order = []
        stack = [m]
        while stack:
            x = stack.pop()
            order.append(x)
            stack.extend(x.children.values())
        for x in reversed(order):
            ni = len(x.leaves)
            nm = sum(1 for lid in x.leaves if kind[lid] == MACRO)
            for child in x.children.values():
                ni += child._num_inst
                nm += child._num_macro
            x._num_inst = ni
            x._num_macro = nm
        return m._num_inst, m._num_macro

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        depth_hist: dict[int, int] = {}
        n_modules = 0
        stack = [(self.root, 0)]
        while stack:
            m, d = stack.pop()
            n_modules += 1
            depth_hist[d] = depth_hist.get(d, 0) + 1
            stack.extend((c, d + 1) for c in m.children.values())
        return {
            "modules": n_modules,
            "top_children": len(self.root.children),
            "max_depth": max(depth_hist),
            "root_inst": self.root.GetNumInst(),
            "root_macro": self.root.GetNumMacro(),
            "inferred_scopes": self.inferred,
        }
