"""Reconstruct the logical hierarchy from flattened instance names.

The synthesized ariane81 netlist is a single flat module, but the original
RTL hierarchy survives in the instance path names, e.g.

    i_cache_subsystem/i_icache/sram_block_0__data_sram/genblk1_0__i_ram/macro_mem_0

Both '/' (DEF DIVIDERCHAR) and '.' (SystemVerilog struct/interface scopes
that survived flattening) act as hierarchy separators here.  We split on
both to rebuild the module tree the Autoclustering Engine traverses in a
top-down manner (Algorithm 1, lines 5-34).

Each :class:`Module` exposes ``GetNumInst`` and ``GetNumMacro`` matching the
notation used in the paper's pseudo-code.
"""

from __future__ import annotations

import re

from .netlist import MACRO, Netlist

_SPLIT_RE = re.compile(r"[/.]")


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
    def __init__(self, nl: Netlist):
        self.nl = nl
        self.root = Module("", "", None)
        self._build()
        self._annotate(self.root)

    # ------------------------------------------------------------------ #
    def _build(self):
        nl = self.nl
        root = self.root
        for iid, name in enumerate(nl.inst_name):
            tokens = _SPLIT_RE.split(name)
            # tokens[-1] is the leaf instance's local name; the prefix is
            # the module path it lives in.
            node = root
            path_parts = []
            for tok in tokens[:-1]:
                path_parts.append(tok)
                child = node.children.get(tok)
                if child is None:
                    child = Module(tok, "/".join(path_parts), node)
                    node.children[tok] = child
                node = child
            node.leaves.append(iid)

    def _annotate(self, m: Module) -> tuple[int, int]:
        """Post-order fill of subtree instance / macro counts."""
        ni = len(m.leaves)
        nm = sum(1 for lid in m.leaves if self.nl.inst_kind[lid] == MACRO)
        for child in m.children.values():
            ci, cm = self._annotate(child)
            ni += ci
            nm += cm
        m._num_inst = ni
        m._num_macro = nm
        return ni, nm

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        depth_hist: dict[int, int] = {}
        n_modules = 0

        def walk(m: Module, d: int):
            nonlocal n_modules
            n_modules += 1
            depth_hist[d] = depth_hist.get(d, 0) + 1
            for c in m.children.values():
                walk(c, d + 1)

        walk(self.root, 0)
        return {
            "modules": n_modules,
            "top_children": len(self.root.children),
            "max_depth": max(depth_hist),
            "root_inst": self.root.GetNumInst(),
            "root_macro": self.root.GetNumMacro(),
        }
