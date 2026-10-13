"""Autoclustering Engine - RTL-MP / Hier-RTLMP Algorithm 2+3+4 (single level).

Converts the flat gate-level netlist + reconstructed logical hierarchy into a
clustered netlist:  nodes are clusters, edges are bundled connections.

  Step 1  bundled IO pins (num_bundled_pins_per_edge per die edge)   [Alg.2 l.3-4]
  Step 2  top-down hierarchy decomposition: break / keep / merge     [Alg.3]
            * break   : num_macro > max_num_macro or num_std > max_num_inst
            * merge   : num_macro < min_num_macro and num_std < min_num_inst
            * keep    : everything else
          merge candidates by connection signature                   [Alg.4]
            * a candidate connected to exactly one reference cluster is
              absorbed into it (Alg.4 lines 14-18)
            * remaining candidates with the same signature are merged
              (Alg.4 line 19); unconnected leftovers go to their strongest
              raw neighbour, else to the closest kept ancestor/glue cluster
  Step 3  break oversized flat clusters (TritonPart substitute)      [Alg.3 l.4-5]
  Step 4  split mixed clusters into std-cell + macro clusters,
          add virtual std-cell<->macro connections                   [Alg.2 l.14,16]
  Step 5  regroup macros by connection signature, then by footprint [Alg.2 l.15]
  (Step 6 - dataflow virtual connections - lives in dataflow.py)     [Alg.2 l.17]

Hyperparameters default to the empirical values from RTL-MP Section 7.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from .hierarchy import Hierarchy, Module, split_path
from .netlist import BTERM, COMB, INST, MACRO, REGISTER, Netlist

# Cluster types
STDCELL = "StdCell"
MACRO_C = "Macro"
MIXED = "Mixed"
PIN = "Pin"


def _common_prefix(strs):
    if not strs:
        return ""
    a, b = min(strs), max(strs)
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def merged_name(paths, max_list=3):
    """Readable name for a cluster built from several module paths.

    ``i_cache_subsystem/i_nbdcache/sram_block_{0,1,2,...+5}__tag_sram`` -
    common hierarchical prefix, then the distinguishing parts in braces.
    """
    paths = [p or "<top>" for p in paths]
    if len(paths) == 1:
        return paths[0]
    toks = [split_path(p) for p in paths]
    common = []
    for col in zip(*toks):
        if all(t == col[0] for t in col):
            common.append(col[0])
        else:
            break
    tails = []
    for t in toks:
        tail = "/".join(t[len(common):]) or "."
        if tail not in tails:
            tails.append(tail)
    tails.sort()
    pre = _common_prefix(tails)
    while pre and pre[-1].isalnum():          # snap to a token boundary
        pre = pre[:-1]
    suf = _common_prefix([t[::-1] for t in tails])[::-1]
    while suf and suf[0].isalnum():
        suf = suf[1:]
    if len(pre) + len(suf) >= min(len(t) for t in tails):
        suf = ""
    mids = [t[len(pre):len(t) - len(suf)] or "." for t in tails]
    if len(mids) > max_list:
        body = ",".join(mids[:max_list]) + f",...+{len(mids) - max_list}"
    else:
        body = ",".join(mids)
    prefix = "/".join(common)
    return (prefix + "/" if prefix else "") + pre + "{" + body + "}" + suf


class Cluster:
    _next = 0

    def __init__(self, name, ctype=MIXED, inst_ids=None, xy=None, members=None):
        self.id = Cluster._next
        Cluster._next += 1
        self.name = name
        self.type = ctype
        self.inst_ids: list[int] = list(inst_ids) if inst_ids else []
        self.xy = xy                       # (x,y) for bundled-pin clusters
        # logical module paths that make up this cluster (for naming / audit)
        self.members: list[str] = list(members) if members else [name]
        # name of the mixed parent a std/macro cluster was split from (Step 4),
        # used to re-pair them with a virtual connection after Step 5 splits.
        self.origin = None
        # virtual connections {other_cluster_id: weight} added in steps 4 & 6
        self.virtual: dict[int, float] = defaultdict(float)
        # IO bits of a bundled-pin cluster; empty for real clusters
        self.bterm_ids: list[int] = []

    # paper-style accessors -------------------------------------------- #
    def GetNumInst(self, nl):              # noqa: N802
        return len(self.inst_ids)

    def GetNumMacro(self, nl):             # noqa: N802
        kind = nl.inst_kind
        return sum(1 for i in self.inst_ids if kind[i] == MACRO)

    def GetNumStdCell(self, nl):           # noqa: N802
        return len(self.inst_ids) - self.GetNumMacro(nl)

    def area(self, nl):
        return sum(nl.inst_area[i] for i in self.inst_ids)

    def absorb(self, other: "Cluster", keep_name: bool = False):
        self.inst_ids.extend(other.inst_ids)
        for m in other.members:
            if m not in self.members:
                self.members.append(m)
        if keep_name:
            self.absorbed = getattr(self, "absorbed", 0) + 1

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
        # Step 3 bisection: FM refinement passes (0 = plain BFS region growth)
        self.bisect_refine_passes = 4
        self.bisect_balance = 0.1         # allowed |a-b| / n imbalance
        # std-cell clusters below this size are noise for a macro placer and
        # are absorbed into their strongest-connected neighbour
        self.tiny_cluster_inst = max(50, self.min_num_inst // 10)

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
        self._inst_adj = None             # lazily built star-model adjacency
        self.log: list[str] = []          # human-readable audit trail

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
                bundled.append(c)
        self.pin_clusters = bundled

        # assign each real IO pin to the nearest bundled pin
        for bid, (px, py) in enumerate(nl.bterm_xy):
            best, bestd = None, 1e30
            for c in bundled:
                cx, cy = c.xy
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < bestd:
                    bestd, best = d, c
            best.bterm_ids.append(bid)
            self.bterm2cl[bid] = best.id

    # ===== Step 2: top-down hierarchy decomposition =================== #
    def _decompose(self):
        nl, p = self.nl, self.p

        def too_big(m: Module):
            return (m.GetNumMacro() > p.max_num_macro or
                    m.GetNumStdCell() > p.max_num_inst)

        def is_candidate(m: Module):
            # Algorithm 3 line 11 (note: strict "<" on both counts)
            return (m.GetNumMacro() < p.min_num_macro and
                    m.GetNumStdCell() < p.min_num_inst)

        def cluster_for_module(m: Module) -> Cluster:
            return Cluster(m.path or "<top>", MIXED,
                           inst_ids=list(m.subtree_leaves()))

        kept: list[Cluster] = []           # new_children
        candidates: list[Cluster] = []     # candidate_clusters

        top = self.hier.root
        if not too_big(top):
            kept.append(cluster_for_module(top))
            return kept, candidates

        queue: deque[Module] = deque([top])
        while queue:
            c = queue.popleft()
            if not c.has_children():
                # oversized flat module -> Step 3 (partitioner)
                kept.append(cluster_for_module(c))
                continue
            for child in c.children.values():
                if too_big(child):
                    queue.append(child)            # dissolve further
                elif is_candidate(child):
                    candidates.append(cluster_for_module(child))
                else:
                    kept.append(cluster_for_module(child))
            # glue logic = leaf instances directly inside c
            if c.leaves:
                glue = Cluster((c.path or "<top>") + "/glue_logic",
                               MIXED, inst_ids=list(c.leaves))
                nmac = glue.GetNumMacro(nl)
                nstd = len(glue.inst_ids) - nmac
                if nmac < p.min_num_macro and nstd < p.min_num_inst:
                    candidates.append(glue)
                else:
                    kept.append(glue)
        return kept, candidates

    # ----- connection-signature merge -------------------------------- #
    def _assign(self, clusters):
        """Build inst->cluster map for the given clustering."""
        self.inst2cl = [-1] * self.nl.num_insts
        for c in clusters:
            cid = c.id
            for i in c.inst_ids:
                self.inst2cl[i] = cid

    def _cluster_adjacency(self, clusters):
        """Directed star-model connection counts between clusters.

        Returns {cluster_id: {neighbor_id: count}} ignoring global nets and
        intra-cluster connections.  Bundled-pin clusters participate too.
        """
        nl, p = self.nl, self.p
        self._assign(clusters)
        inst2cl = self.inst2cl
        bterm2cl = self.bterm2cl
        adj = defaultdict(lambda: defaultdict(int))
        thr = p.large_net_threshold
        for pins in nl.net_pins:
            if len(pins) > thr:
                continue                    # clk / reset etc.
            dc = -1
            for kind, eid, is_drv in pins:
                if is_drv:
                    dc = inst2cl[eid] if kind == INST else bterm2cl.get(eid, -1)
                    break
            if dc < 0:
                continue
            row = adj[dc]
            for kind, eid, is_drv in pins:
                if is_drv:
                    continue
                sc = inst2cl[eid] if kind == INST else bterm2cl.get(eid, -1)
                if sc < 0 or sc == dc:
                    continue
                row[sc] += 1                # directed driver -> sink
        return adj

    @staticmethod
    def _symmetric(adj):
        """Undirected connection counts {a: {b: w}} from a directed adjacency."""
        sym = defaultdict(lambda: defaultdict(int))
        for a, nbrs in adj.items():
            for b, w in nbrs.items():
                sym[a][b] += w
                sym[b][a] += w
        return sym

    def _merge_by_signature(self, kept, candidates):
        """Algorithm 4 (single-level): merge small clusters by signature.

        reference clusters = kept clusters + bundled pins.  The signature of a
        candidate is the set of reference clusters it connects to with at
        least ``net_threshold`` nets.
        """
        p = self.p
        if not candidates:
            return kept
        universe = kept + candidates + self.pin_clusters
        sym = self._symmetric(self._cluster_adjacency(universe))
        ref_ids = {c.id for c in kept} | {c.id for c in self.pin_clusters}
        by_id = {c.id: c for c in universe}

        def signature(c: Cluster):
            return frozenset(nb for nb, w in sym.get(c.id, {}).items()
                             if nb in ref_ids and w >= p.net_threshold)

        groups: dict[frozenset, list[Cluster]] = defaultdict(list)
        absorbed = 0
        for c in candidates:
            sig = signature(c)
            if len(sig) == 1:
                ref = by_id[next(iter(sig))]
                if ref.type != PIN:            # Alg.4 lines 14-18
                    ref.absorb(c, keep_name=True)
                    absorbed += 1
                    continue
            groups[sig].append(c)

        def strongest_ref(c: Cluster):
            nbrs = [(w, nb) for nb, w in sym.get(c.id, {}).items()
                    if nb in ref_ids and by_id[nb].type != PIN]
            return by_id[max(nbrs)[1]] if nbrs else None

        merged: list[Cluster] = []
        leftovers: list[Cluster] = []
        for sig, members in groups.items():
            if not sig:
                leftovers.extend(members)
                continue
            base = members[0]
            for other in members[1:]:
                base.absorb(other)
            # a lone candidate, or a merged group that is still tiny, is not a
            # useful cluster: attach it to the reference it talks to most
            if (len(members) == 1 or base.GetNumStdCell(self.nl) < p.tiny_cluster_inst) \
                    and base.GetNumMacro(self.nl) == 0:
                ref = strongest_ref(base)
                if ref is None:
                    leftovers.append(base)
                    continue
                ref.absorb(base, keep_name=True)
                absorbed += 1
                continue
            merged.append(base)

        # Unconnected leftovers (no neighbour above threshold): attach each to
        # its strongest raw neighbour, else to the closest kept ancestor /
        # glue cluster, else group them together (same empty signature).
        kept_by_path = {}
        for c in kept + merged:
            for m in c.members:
                kept_by_path[m] = c
        orphans = []
        for c in leftovers:
            ref = strongest_ref(c)
            if ref is not None:
                ref.absorb(c, keep_name=True)
                absorbed += 1
                continue
            target = None
            toks = split_path(c.members[0])
            while toks and target is None:
                toks = toks[:-1]
                anc = "/".join(toks) or "<top>"
                target = kept_by_path.get(anc + "/glue_logic") or kept_by_path.get(anc)
            if target is not None:
                target.absorb(c, keep_name=True)
                absorbed += 1
            else:
                orphans.append(c)
        if orphans:
            base = orphans[0]
            for other in orphans[1:]:
                base.absorb(other)
            merged.append(base)

        for c in kept:
            n = getattr(c, "absorbed", 0)
            if n:
                c.name = f"{c.name}(+{n})"
        for c in merged:
            if len(c.members) > 1:
                c.name = merged_name(c.members)
            n = getattr(c, "absorbed", 0)
            if n:
                c.name = f"{c.name}(+{n})"
        self.log.append(f"merge: {len(candidates)} candidates -> {absorbed} absorbed "
                        f"into reference clusters, {len(merged)} merged groups")
        return kept + merged

    # ===== Step 3: partitioner-style break of oversized clusters ====== #
    def _inst_adjacency(self):
        """Star-model instance adjacency (undirected), built once."""
        if self._inst_adj is not None:
            return self._inst_adj
        nl, p = self.nl, self.p
        adj = [[] for _ in range(nl.num_insts)]
        thr = p.large_net_threshold
        for pins in nl.net_pins:
            if len(pins) > thr:
                continue
            drv = -1
            for kind, eid, is_drv in pins:
                if is_drv and kind == INST:
                    drv = eid
                    break
            if drv < 0:
                continue
            for kind, eid, is_drv in pins:
                if kind == INST and eid != drv and not is_drv:
                    adj[drv].append(eid)
                    adj[eid].append(drv)
        self._inst_adj = adj
        return adj

    def _break_oversized(self, clusters):
        """Alg.3 lines 4-5: partition flat clusters with more than
        ``max_num_inst`` standard cells.  Only the standard cells are
        partitioned; the macros of such a cluster stay together as one macro
        cluster (they are regrouped in Step 5) that is tied to every std-cell
        piece of its parent through the Step 4 virtual connection.
        """
        nl, p = self.nl, self.p
        out = []
        for c in clusters:
            if c.GetNumStdCell(nl) <= p.max_num_inst:
                out.append(c)
                continue
            macros = [i for i in c.inst_ids if nl.inst_kind[i] == MACRO]
            stds = [i for i in c.inst_ids if nl.inst_kind[i] != MACRO]
            std_c = Cluster(c.name, STDCELL, inst_ids=stds, members=c.members)
            pieces = self._bipartition_recursive(std_c)
            if macros:
                mac_c = self._macro_cluster(c, macros)
                for piece in pieces:
                    piece.origin = c.name
                out.append(mac_c)
            out.extend(pieces)
            self.log.append(f"step3: {c.name} ({len(stds)} std cells"
                            f"{', ' + str(len(macros)) + ' macros kept whole' if macros else ''})"
                            f" -> {len(pieces)} pieces")
        return out

    def _bipartition_recursive(self, c: Cluster):
        """Recursively bisect until every piece <= max_num_inst std cells."""
        p = self.p
        if len(c.inst_ids) <= p.max_num_inst:
            return [c]
        a, b = self._bisect(c.inst_ids)
        pieces = []
        for idx, half in enumerate((a, b)):
            sub = Cluster(f"{c.name}.p{idx}", c.type, inst_ids=half,
                          members=[f"{m}.p{idx}" for m in c.members])
            pieces.extend(self._bipartition_recursive(sub))
        return pieces

    def _bisect(self, inst_ids):
        """Connectivity-aware bisection (min-cut spirit).

        BFS region growth from the highest-degree seed gives a connected
        half; a few Fiduccia-Mattheyses style refinement passes then move
        boundary instances with positive gain while keeping balance.
        """
        p = self.p
        adj = self._inst_adjacency()
        members = set(inst_ids)
        n = len(inst_ids)
        target = n // 2

        local_deg = {i: sum(1 for v in adj[i] if v in members) for i in inst_ids}
        seed = max(inst_ids, key=local_deg.__getitem__)
        side = {}
        visited = {seed}
        frontier = deque([seed])
        count_a = 0
        order = []
        while frontier and count_a < target:
            u = frontier.popleft()
            order.append(u)
            count_a += 1
            for v in adj[u]:
                if v in members and v not in visited:
                    visited.add(v)
                    frontier.append(v)
        half_a = order
        in_a = set(half_a)
        # unreached (disconnected) instances fill up side A if needed
        rest = [i for i in inst_ids if i not in in_a]
        if len(half_a) < target:
            need = target - len(half_a)
            half_a.extend(rest[:need])
            rest = rest[need:]
        for i in half_a:
            side[i] = 0
        for i in rest:
            side[i] = 1

        if p.bisect_refine_passes > 0:
            self._fm_refine(inst_ids, members, adj, side, p.bisect_refine_passes,
                            p.bisect_balance)
        a = [i for i in inst_ids if side[i] == 0]
        b = [i for i in inst_ids if side[i] == 1]
        return a, b

    @staticmethod
    def _fm_refine(inst_ids, members, adj, side, passes, balance):
        n = len(inst_ids)
        size = [0, 0]
        for i in inst_ids:
            size[side[i]] += 1
        max_imb = max(1, int(balance * n))

        def gain(u):
            s = side[u]
            ext = 0
            inte = 0
            for v in adj[u]:
                if v in members:
                    if side[v] == s:
                        inte += 1
                    else:
                        ext += 1
            return ext - inte

        for _ in range(passes):
            moved = 0
            # boundary vertices only
            cand = [u for u in inst_ids
                    if any(v in members and side[v] != side[u] for v in adj[u])]
            cand.sort(key=gain, reverse=True)
            for u in cand:
                g = gain(u)
                if g <= 0:
                    break
                s = side[u]
                if size[s] - 1 < size[1 - s] + 1 - max_imb:
                    continue                 # would violate balance
                side[u] = 1 - s
                size[s] -= 1
                size[1 - s] += 1
                moved += 1
            if moved == 0:
                break

    def _macro_cluster(self, c: Cluster, macros: list[int]) -> Cluster:
        """Macro half of cluster ``c``, named after the modules that hold the
        macros (not after the std-cell logic they were merged into)."""
        nl = self.nl
        mem = sorted({self._member_of(nl.inst_name[i], c.members) for i in macros})
        mem = [m for m in mem if m] or c.members
        name = merged_name(mem) if len(c.members) > 1 else c.name + "/macros"
        mac_c = Cluster(name, MACRO_C, inst_ids=macros, members=mem)
        mac_c.origin = c.name
        return mac_c

    # ===== Step 4: split mixed clusters; virtual macro<->stdcell ====== #
    def _split_mixed(self, clusters):
        nl = self.nl
        out = []
        for c in clusters:
            macros = [i for i in c.inst_ids if nl.inst_kind[i] == MACRO]
            stds = [i for i in c.inst_ids if nl.inst_kind[i] != MACRO]
            if macros and stds:
                std_c = Cluster(c.name + "/std", STDCELL, inst_ids=stds,
                                members=c.members)
                mac_c = self._macro_cluster(c, macros)
                # tag both halves so they can be re-paired with a virtual
                # connection after the Step 5 split (Alg.2 line 16).
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

    def _absorb_tiny_std(self, clusters):
        """A mixed cluster whose std-cell half is tiny (a few address
        decoders next to a memory bank) leaves a std-cell cluster that is
        noise for the placer: fold it into the std-cell cluster it is most
        connected to and let the Step 4 virtual connection follow it.
        """
        p = self.p
        tiny = [c for c in clusters
                if c.type == STDCELL and c.origin is not None
                and len(c.inst_ids) < p.tiny_cluster_inst]
        if not tiny:
            return clusters
        sym = self._symmetric(self._cluster_adjacency(clusters + self.pin_clusters))
        by_id = {c.id: c for c in clusters}
        self.origin_redirect: dict[str, int] = {}
        gone = set()
        for c in tiny:
            best = None
            for nb, w in sym.get(c.id, {}).items():
                t = by_id.get(nb)
                if t is None or t.type != STDCELL or t.id in gone or t is c:
                    continue
                if best is None or w > best[0]:
                    best = (w, t)
            if best is None:
                continue
            target = best[1]
            target.absorb(c, keep_name=True)
            self.origin_redirect[c.origin] = target.id
            gone.add(c.id)
        out = [c for c in clusters if c.id not in gone]
        self.log.append(f"step4: {len(gone)} tiny std-cell halves "
                        f"(< {p.tiny_cluster_inst} cells) folded into neighbours")
        return out

    # ===== Step 5: group macros by signature, then footprint ========== #
    def _macro_signatures(self, clusters):
        """Per-macro connection signature w.r.t. all other clusters + pins."""
        nl, p = self.nl, self.p
        self._assign(clusters)
        # reference id: pieces split from the same mixed parent (Step 3/4)
        # are one reference, so a partition cut cannot separate a bank array
        ref_of = {}
        origin_ids = {}
        for c in clusters:
            if c.origin is not None:
                ref_of[c.id] = origin_ids.setdefault(c.origin, -2 - len(origin_ids))
            else:
                ref_of[c.id] = c.id
        inst2cl = [ref_of.get(cid, -1) if cid >= 0 else -1 for cid in self.inst2cl]
        bterm2cl = self.bterm2cl
        is_macro = [k == MACRO for k in nl.inst_kind]
        cnt = defaultdict(lambda: defaultdict(int))
        thr = p.large_net_threshold
        for pins in nl.net_pins:
            if len(pins) > thr:
                continue
            drv = None
            for pin in pins:
                if pin[2]:
                    drv = pin
                    break
            if drv is None:
                continue
            dkind, did, _ = drv
            dc = inst2cl[did] if dkind == INST else bterm2cl.get(did, -1)
            dmac = dkind == INST and is_macro[did]
            for kind, eid, is_drv in pins:
                if is_drv:
                    continue
                sc = inst2cl[eid] if kind == INST else bterm2cl.get(eid, -1)
                if dmac and sc != -1 and sc != dc:
                    cnt[did][sc] += 1
                if kind == INST and is_macro[eid] and dc != -1 and dc != sc:
                    cnt[eid][dc] += 1
        sig = {}
        for m, nbrs in cnt.items():
            sig[m] = frozenset(c for c, w in nbrs.items() if w >= p.net_threshold)
        return sig

    @staticmethod
    def _member_of(name: str, members: list[str]) -> str:
        toks = split_path(name)
        best = ""
        for m in members:
            mt = split_path(m)
            if toks[:len(mt)] == mt and len(mt) >= len(split_path(best)):
                best = m
        return best

    @staticmethod
    def _near_identical_groups(member_sig: dict, min_jaccard: float = 0.5) -> dict:
        """Union members whose signature sets overlap (Jaccard >= 0.5): the
        paper asks for *(near-)identical* connectivity, and exact set equality
        is brittle against a single extra neighbour above the net threshold.
        Returns {member: frozenset(signature of its group)}."""
        names = sorted(member_sig)
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a_i in range(len(names)):
            for b_i in range(a_i + 1, len(names)):
                a, b = member_sig[names[a_i]], member_sig[names[b_i]]
                if not a and not b:
                    j = 1.0
                else:
                    j = len(a & b) / len(a | b)
                if j >= min_jaccard:
                    parent[find(names[a_i])] = find(names[b_i])
        groups = defaultdict(set)
        for n in names:
            groups[find(n)] |= member_sig[n]
        return {n: frozenset(groups[find(n)]) for n in names}

    def _group_macros(self, clusters):
        """Alg.2 line 15: regroup the macros of a macro cluster by connection
        signature, then split groups whose macros have different footprints
        so every macro cluster tiles regularly.

        The signature regrouping is what lets macros that were merged from
        *different* logical modules (Step 2) be separated again when they
        talk to different clusters.  It is evaluated per member module, not
        per macro: the macros of one RTL memory bank share address / enable
        nets and must stay together whatever partition cut their bit slices
        happen to land on.
        """
        nl = self.nl
        sig = self._macro_signatures(clusters + self.pin_clusters)
        out = []
        for c in clusters:
            if c.type != MACRO_C:
                out.append(c)
                continue
            multi = len(c.members) > 1
            by_sig: dict[frozenset, list[int]] = defaultdict(list)
            if multi:
                member_sig: dict[str, set] = defaultdict(set)
                member_of = {i: self._member_of(nl.inst_name[i], c.members)
                             for i in c.inst_ids}
                for i in c.inst_ids:
                    member_sig[member_of[i]] |= sig.get(i, frozenset())
                group_of = self._near_identical_groups(member_sig)
                for i in c.inst_ids:
                    by_sig[group_of[member_of[i]]].append(i)
            else:
                by_sig[frozenset()] = list(c.inst_ids)
            groups = []
            for s_, ids in by_sig.items():
                by_size: dict[str, list[int]] = defaultdict(list)
                for i in ids:
                    by_size[nl.inst_master[i]].append(i)
                for master, mids in by_size.items():
                    groups.append((master, sorted(mids, key=lambda i: nl.inst_name[i])))
            if len(groups) == 1:
                out.append(c)
                continue
            groups.sort(key=lambda g: (-len(g[1]), g[0]))
            multi_sig = len(by_sig) > 1
            # name each group after the modules holding its macros; add the
            # footprint (and a counter) only when that alone is ambiguous
            named = []
            for master, ids in groups:
                mem = sorted({self._member_of(nl.inst_name[i], c.members) for i in ids})
                mem = [m for m in mem if m]
                base = merged_name(mem) if (mem and multi) else c.name
                named.append([base, master, ids, mem])
            bases = defaultdict(int)
            for base, master, ids, mem in named:
                bases[base] += 1
            seen = defaultdict(int)
            for base, master, ids, mem in named:
                name = base
                if bases[base] > 1:
                    name = f"{base}[{master}]"
                    seen[name] += 1
                    if seen[name] > 1:
                        name = f"{name}#{seen[name] - 1}"
                sub = Cluster(name, MACRO_C, inst_ids=ids, members=mem or c.members)
                sub.origin = c.origin
                out.append(sub)
            self.log.append(f"step5: {c.name} -> {len(groups)} macro groups "
                            f"({'signature+' if multi_sig else ''}footprint)")
        return out

    def _wire_macro_stdcell(self, clusters):
        """Alg.2 line 16: connect each macro cluster to its sibling std-cell
        cluster (same mixed parent) with weight ``virtual_weight``.
        """
        p = self.p
        by_origin = defaultdict(lambda: {"macro": [], "std": []})
        by_id = {c.id: c for c in clusters}
        redirect = getattr(self, "origin_redirect", {})
        for c in clusters:
            if c.origin is None:
                continue
            if c.type == MACRO_C:
                by_origin[c.origin]["macro"].append(c)
            elif c.type == STDCELL:
                by_origin[c.origin]["std"].append(c)
        for origin, cid in redirect.items():
            if cid in by_id and by_id[cid] not in by_origin[origin]["std"]:
                by_origin[origin]["std"].append(by_id[cid])
        for grp in by_origin.values():
            for mc in grp["macro"]:
                for sc in grp["std"]:
                    mc.virtual[sc.id] += p.virtual_weight
                    sc.virtual[mc.id] += p.virtual_weight

    # ===== driver ===================================================== #
    def run(self):
        self._make_bundled_pins()
        kept, candidates = self._decompose()
        self.log.append(f"decompose: {len(kept)} kept, {len(candidates)} merge candidates")
        clusters = self._merge_by_signature(kept, candidates)
        clusters = self._break_oversized(clusters)
        clusters = self._split_mixed(clusters)
        clusters = self._absorb_tiny_std(clusters)
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
