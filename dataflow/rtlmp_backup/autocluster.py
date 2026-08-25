"""Autoclustering Engine - RTL-MP Algorithm 1.

Converts the flat gate-level netlist + reconstructed logical hierarchy into a
clustered netlist:  nodes are clusters, edges are bundled connections.

The implementation follows the six steps of Section 4 / Algorithm 1:

  Step 1  bundled IO pins (3 per die edge)                       [lines 3-4]
  Step 2  top-down hierarchy decomposition with break / keep /
          merge-by-connection-signature                          [lines 5-34]
  Step 3  break oversized leaf clusters (MLPart substitute)      [line 35]
  Step 4  split mixed clusters into std-cell + macro clusters,
          add virtual std-cell<->macro connections               [lines 36, 38]
  Step 5  group macros by connection signature, then by footprint [line 37]
  (Step 6 - dataflow virtual connections - lives in dataflow.py)  [line 39]

Hyperparameters default to the empirical values from Section 7 (Eq. 9).
"""

from __future__ import annotations

import math
from collections import defaultdict

from .hierarchy import Hierarchy, Module
from .netlist import BTERM, COMB, INST, MACRO, REGISTER, Netlist

# Cluster types
STDCELL = "StdCell"
MACRO_C = "Macro"
MIXED = "Mixed"
PIN = "Pin"


class Cluster:
    _next = 0

    def __init__(self, name, ctype=MIXED, inst_ids=None, xy=None):
        self.id = Cluster._next
        Cluster._next += 1
        self.name = name
        self.type = ctype
        self.inst_ids: list[int] = list(inst_ids) if inst_ids else []
        self.xy = xy                       # (x,y) for bundled-pin clusters
        # name of the mixed parent a std/macro cluster was split from (Step 4),
        # used to re-pair them with a virtual connection after Step 5 splits.
        self.origin = None
        # virtual connections {other_cluster_id: weight} added in steps 4 & 6
        self.virtual: dict[int, float] = defaultdict(float)
        # bitwidth of bundled IO pins (for dataflow); 0 for real clusters
        self.bterm_ids: list[int] = []

    # paper-style accessors -------------------------------------------- #
    def GetNumInst(self, nl):              # noqa: N802
        return len(self.inst_ids)

    def GetNumMacro(self, nl):             # noqa: N802
        return sum(1 for i in self.inst_ids if nl.inst_kind[i] == MACRO)

    def GetNumStdCell(self, nl):           # noqa: N802
        return sum(1 for i in self.inst_ids if nl.inst_kind[i] != MACRO)

    def area(self, nl):
        return sum(nl.inst_area[i] for i in self.inst_ids)

    def __repr__(self):
        return f"<Cluster {self.id} {self.name!r} {self.type} n={len(self.inst_ids)}>"


class Params:
    """RTL-MP hyperparameters (Section 7)."""

    def __init__(self, total_num_inst):
        # Eq. 9 - sizes scale with the design.
        self.min_num_inst = max(1000,
                                math.floor(total_num_inst / (50 * 5000)) * 5000)
        self.max_num_inst = min(50000, 5 * self.min_num_inst)
        self.max_num_macro = 12
        self.min_num_macro = 4
        self.virtual_weight = 500
        self.net_threshold = 5            # min connections => "connected"
        self.large_net_threshold = 500    # ignore clk/reset-like global nets
        self.num_bundled_pins_per_edge = 3

    def __repr__(self):
        return (f"Params(min_inst={self.min_num_inst}, max_inst={self.max_num_inst}, "
                f"min_macro={self.min_num_macro}, max_macro={self.max_num_macro}, "
                f"net_thr={self.net_threshold}, virt_w={self.virtual_weight})")


class Autocluster:
    def __init__(self, nl: Netlist, hier: Hierarchy, params: Params | None = None):
        self.nl = nl
        self.hier = hier
        self.p = params or Params(nl.num_insts)
        Cluster._next = 0
        self.clusters: dict[int, Cluster] = {}
        self.pin_clusters: list[Cluster] = []
        # inst id -> cluster id (filled once the final clustering exists)
        self.inst2cl: list[int] = [-1] * nl.num_insts
        # bterm id -> bundled-pin cluster id
        self.bterm2cl: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    def _add(self, cl: Cluster) -> Cluster:
        self.clusters[cl.id] = cl
        return cl

    # ===== Step 1: bundled IO pins ==================================== #
    def _make_bundled_pins(self):
        nl, p = self.nl, self.p
        lx, ly, ux, uy = nl.die
        n = p.num_bundled_pins_per_edge
        edges = {
            "B": [(lx + (ux - lx) * (k + 0.5) / n, ly) for k in range(n)],
            "T": [(lx + (ux - lx) * (k + 0.5) / n, uy) for k in range(n)],
            "L": [(lx, ly + (uy - ly) * (k + 0.5) / n) for k in range(n)],
            "R": [(ux, ly + (uy - ly) * (k + 0.5) / n) for k in range(n)],
        }
        bundled = []
        for edge, pts in edges.items():
            for k, (x, y) in enumerate(pts):
                c = self._add(Cluster(f"PIN_{edge}{k}", PIN, xy=(x, y)))
                bundled.append((edge, k, c))
        self.pin_clusters = [b[2] for b in bundled]

        # assign each real IO pin to the nearest bundled pin
        def nearest(px, py):
            best, bestd = None, 1e30
            for edge, k, c in bundled:
                cx, cy = c.xy
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < bestd:
                    bestd, best = d, c
            return best

        for bid, (px, py) in enumerate(nl.bterm_xy):
            c = nearest(px, py)
            c.bterm_ids.append(bid)
            self.bterm2cl[bid] = c.id

    # ===== Step 2: top-down hierarchy decomposition =================== #
    def _decompose(self):
        nl, p = self.nl, self.p

        def too_big(m: Module):
            return (m.GetNumInst() > p.max_num_inst or
                    m.GetNumMacro() > p.max_num_macro)

        def big_enough(m: Module):
            return (m.GetNumInst() > p.min_num_inst or
                    m.GetNumMacro() > p.min_num_macro)

        kept: list[Cluster] = []           # cluster_list
        merge_candidates: list[Cluster] = []

        def cluster_for_module(m: Module) -> Cluster:
            return Cluster(m.path or "<top>", MIXED,
                           inst_ids=list(m.subtree_leaves()))

        top = self.hier.root
        break_queue: list[Module] = []
        if too_big(top):
            break_queue.append(top)
        else:
            kept.append(cluster_for_module(top))

        while break_queue:
            c = break_queue.pop(0)
            if c.has_children():
                for child in c.children.values():
                    if too_big(child):
                        break_queue.append(child)   # break further
                    elif big_enough(child):
                        kept.append(cluster_for_module(child))
                    else:
                        merge_candidates.append(cluster_for_module(child))
                # glue logic = leaf instances directly inside c
                if c.leaves:
                    glue = Cluster((c.path or "<top>") + "/glue_logic",
                                   MIXED, inst_ids=list(c.leaves))
                    nmac = glue.GetNumMacro(nl)
                    if len(glue.inst_ids) < p.min_num_inst and nmac < p.min_num_macro:
                        merge_candidates.append(glue)
                    else:
                        kept.append(glue)
            else:
                # oversized leaf module with no sub-hierarchy -> Step 3 (MLPart)
                kept.append(cluster_for_module(c))

        return kept, merge_candidates

    # ----- connection-signature merge -------------------------------- #
    def _assign(self, clusters):
        """Build inst->cluster map for the given clustering."""
        self.inst2cl = [-1] * self.nl.num_insts
        for c in clusters:
            for i in c.inst_ids:
                self.inst2cl[i] = c.id

    def _cluster_adjacency(self, clusters):
        """Directed star-model connection counts between clusters.

        Returns {cluster_id: {neighbor_id: count}} ignoring global nets and
        intra-cluster connections.  Bundled-pin clusters participate too.
        """
        nl, p = self.nl, self.p
        self._assign(clusters)
        adj = defaultdict(lambda: defaultdict(int))

        def cl_of(kind, eid):
            if kind == INST:
                return self.inst2cl[eid]
            return self.bterm2cl.get(eid, -1)

        for pins in nl.net_pins:
            if len(pins) > p.large_net_threshold:
                continue                    # clk / reset etc.
            driver = None
            for kind, eid, is_drv in pins:
                if is_drv:
                    driver = (kind, eid)
                    break
            if driver is None:
                continue
            dc = cl_of(*driver)
            if dc < 0:
                continue
            for kind, eid, is_drv in pins:
                if is_drv:
                    continue
                sc = cl_of(kind, eid)
                if sc < 0 or sc == dc:
                    continue
                adj[dc][sc] += 1            # directed driver -> sink
        return adj

    def _merge_by_signature(self, kept, merge_candidates):
        """Merge small clusters sharing the same connection signature.

        Signature = the *set* of neighbour clusters connected by at least
        ``net_threshold`` connections (topology, not exact counts - Step 2(i)).
        """
        nl, p = self.nl, self.p
        # the full clustering used to evaluate signatures
        universe = kept + merge_candidates
        adj = self._cluster_adjacency(universe)

        def signature(c: Cluster):
            nbrs = set()
            counts = defaultdict(int)
            for nb, w in adj.get(c.id, {}).items():
                counts[nb] += w
            for nb, w in adj.items():        # incoming edges too
                if c.id in w:
                    counts[nb] += w[c.id]
            for nb, w in counts.items():
                if w >= p.net_threshold:
                    nbrs.add(nb)
            return frozenset(nbrs)

        groups: dict[frozenset, list[Cluster]] = defaultdict(list)
        for c in merge_candidates:
            groups[signature(c)].append(c)

        merged: list[Cluster] = []
        for sig, members in groups.items():
            if len(members) == 1:
                merged.append(members[0])
                continue
            base = members[0]
            for other in members[1:]:
                base.inst_ids.extend(other.inst_ids)
            base.name = f"merged[{members[0].name}]+{len(members) - 1}"
            merged.append(base)
        return kept + merged

    # ===== Step 3: MLPart-style break of oversized leaf clusters ====== #
    def _break_oversized(self, clusters):
        p = self.p
        out = []
        for c in clusters:
            if len(c.inst_ids) > p.max_num_inst:
                out.extend(self._bipartition_recursive(c))
            else:
                out.append(c)
        return out

    def _bipartition_recursive(self, c: Cluster):
        """Recursively bisect until every piece <= max_num_inst.

        MLPart is unavailable, so we use a connectivity-aware greedy
        region-growing bisection: grow a connected half from a seed instance,
        keeping densely-connected logic together (min-cut spirit).
        """
        p = self.p
        if len(c.inst_ids) <= p.max_num_inst:
            return [c]
        a, b = self._greedy_bisect(c.inst_ids)
        pieces = []
        for idx, half in enumerate((a, b)):
            sub = Cluster(f"{c.name}.p{idx}", c.type, inst_ids=half)
            pieces.extend(self._bipartition_recursive(sub))
        return pieces

    def _greedy_bisect(self, inst_ids):
        nl, p = self.nl, self.p
        members = set(inst_ids)
        # local adjacency (undirected) restricted to this set, star model,
        # skipping global nets.
        nbr = defaultdict(set)
        for pins in nl.net_pins:
            if len(pins) > p.large_net_threshold:
                continue
            local = [eid for kind, eid, _ in pins
                     if kind == INST and eid in members]
            if len(local) < 2:
                continue
            drv = None
            for kind, eid, is_drv in pins:
                if is_drv and kind == INST and eid in members:
                    drv = eid
                    break
            if drv is None:
                continue
            for eid in local:
                if eid != drv:
                    nbr[drv].add(eid)
                    nbr[eid].add(drv)
        target = len(inst_ids) // 2
        # BFS region growth from the highest-degree seed
        seed = max(inst_ids, key=lambda i: len(nbr.get(i, ())))
        from collections import deque
        visited = {seed}
        frontier = deque([seed])
        half_a = [seed]
        while frontier and len(half_a) < target:
            u = frontier.popleft()
            for v in nbr.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    half_a.append(v)
                    frontier.append(v)
                    if len(half_a) >= target:
                        break
        # any instances not reached (disconnected) go wherever there's room
        half_a_set = set(half_a)
        half_b = [i for i in inst_ids if i not in half_a_set]
        # rebalance if BFS exhausted before reaching target
        if len(half_a) < target:
            need = target - len(half_a)
            half_a.extend(half_b[:need])
            half_b = half_b[need:]
        return half_a, half_b

    # ===== Step 4: split mixed clusters; virtual macro<->stdcell ====== #
    def _split_mixed(self, clusters):
        nl, p = self.nl, self.p
        out = []
        for c in clusters:
            macros = [i for i in c.inst_ids if nl.inst_kind[i] == MACRO]
            stds = [i for i in c.inst_ids if nl.inst_kind[i] != MACRO]
            if macros and stds:
                std_c = Cluster(c.name + "/std", STDCELL, inst_ids=stds)
                mac_c = Cluster(c.name + "/macros", MACRO_C, inst_ids=macros)
                # tag both halves so they can be re-paired with a virtual
                # connection after the Step 5 footprint split (line 38).
                std_c.origin = mac_c.origin = c.name
                out.append(std_c)
                out.append(mac_c)
            elif macros:
                c.type = MACRO_C
                out.append(c)
            else:
                c.type = STDCELL
                out.append(c)
        return out

    # ===== Step 5: group macros by signature, then footprint ========== #
    def _group_macros(self, clusters):
        """Split each macro cluster so that grouped macros share a footprint.

        Grouping identically-sized macros enables regular tilings (Section 5.1).
        We additionally separate by connection signature first.
        """
        nl = self.nl
        out = []
        for c in clusters:
            if c.type != MACRO_C:
                out.append(c)
                continue
            # group by footprint (master => identical size)
            by_size: dict[str, list[int]] = defaultdict(list)
            for i in c.inst_ids:
                by_size[nl.inst_master[i]].append(i)
            if len(by_size) == 1:
                out.append(c)
                continue
            for master, ids in by_size.items():
                sub = Cluster(f"{c.name}[{master}]", MACRO_C, inst_ids=ids)
                sub.origin = c.origin
                out.append(sub)
        return out

    def _wire_macro_stdcell(self, clusters):
        """Step 4 (line 38): connect each macro cluster to its sibling
        std-cell cluster (same mixed parent) with weight ``virtual_weight``.
        Done after Step 5 so the connection points at the final clusters.
        """
        p = self.p
        by_origin = defaultdict(lambda: {"macro": [], "std": []})
        for c in clusters:
            if c.origin is None:
                continue
            if c.type == MACRO_C:
                by_origin[c.origin]["macro"].append(c)
            elif c.type == STDCELL:
                by_origin[c.origin]["std"].append(c)
        for grp in by_origin.values():
            for mc in grp["macro"]:
                for sc in grp["std"]:
                    mc.virtual[sc.id] += p.virtual_weight
                    sc.virtual[mc.id] += p.virtual_weight

    # ===== driver ===================================================== #
    def run(self):
        self._make_bundled_pins()
        kept, merge_candidates = self._decompose()
        clusters = self._merge_by_signature(kept, merge_candidates)
        clusters = self._break_oversized(clusters)
        clusters = self._split_mixed(clusters)
        clusters = self._group_macros(clusters)
        self._wire_macro_stdcell(clusters)

        # finalize registry
        self.clusters = {c.id: c for c in clusters}
        for c in self.pin_clusters:
            self.clusters[c.id] = c
        self._assign(clusters)
        self.real_clusters = clusters
        return clusters

    # ----- bundled connections (clustered netlist edges) -------------- #
    def bundled_connections(self, directed=False):
        """Aggregate real net connectivity to weighted cluster-cluster edges.

        ``directed=False`` (default) folds both directions into one undirected
        edge keyed (min, max) - the form the HPWL-based macro placer consumes.
        ``directed=True`` keeps driver-cluster -> sink-cluster edges (star
        model), preserving dataflow direction.
        """
        adj = self._cluster_adjacency(self.real_clusters + self.pin_clusters)
        edges: dict[tuple[int, int], float] = defaultdict(float)
        for a, nbrs in adj.items():
            for b, w in nbrs.items():
                key = (a, b) if directed else ((a, b) if a < b else (b, a))
                edges[key] += w
        return dict(edges)
