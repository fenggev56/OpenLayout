"""Dataflow / virtual connections - Hier-RTLMP Algorithm 2 line 17, Eq. (1).

    Virtual_Connections(A, B) = Information_Flow(A, B) / 2 ** Num_Hops

  * Information_Flow  ~  connection bitwidth: the number of *output bits*
                         (driver nets) of the stage nodes in A from which the
                         stage nodes in B are reachable.  A path count would
                         explode through reconvergent logic (multipliers,
                         muxes); counting driver bits bounds every edge by
                         the real bus width.
  * Num_Hops          =  number of register (flip-flop) stages on the
                         shortest register path between A and B.
  * Buffer / comb-cell "transparency": only registers count as hops; all
    combinational cells (including buffers/inverters) are traced through.
  * If Num_Hops > max_hops (default 4) no virtual connection is added.
  * Only macro clusters and bundled IO pins are sources / destinations.

The register-stage graph is built once:

    stage node  := a MACRO instance, a REGISTER instance, or an IO BTerm.
    edge u -> v := v is reachable from one of u's output nets through a cone
                   of purely combinational cells.  Edge weight = number of
                   u's output nets that reach v (bitwidth proxy).
    A single net whose combinational cone reaches more than ``max_fanout``
    stage nodes is a control / global signal (reset, enable, select) and is
    not traced - it carries no dataflow bitwidth.

From every macro / IO source we then BFS over this stage graph, counting a
hop each time we relay through a register, accumulating weighted flow into
the (macro/pin) cluster-pair virtual connections.  The BFS is vectorised
with scipy.sparse when available (all sources at once, one sparse product
per hop); a pure-Python BFS is used otherwise and gives identical results.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict, deque

from .netlist import BTERM, COMB, INST, MACRO, REGISTER, Netlist


class Dataflow:
    # marker for a "broadcast" cone (control / global signal) that fans out
    # to more than ``max_fanout`` stage nodes - never propagated.
    BROADCAST = object()

    def __init__(self, nl: Netlist, ac, max_hops: int = 4, max_fanout=None,
                 mode: str = "min", use_scipy: bool = True):
        """``mode``: 'min' - a (src, dst) pair contributes the flow found at
        its shortest register distance only (aggressive decay, Eq. 1);
        'sum' - every hop level contributes flow / 2**hops (OpenROAD style).
        """
        if mode not in ("min", "sum"):
            raise ValueError("mode must be 'min' or 'sum'")
        self.nl = nl
        self.ac = ac
        self.p = ac.p
        self.max_hops = max_hops
        self.mode = mode
        self.use_scipy = use_scipy
        # a net whose cone reaches more sinks than this is a control signal
        self.max_fanout = max_fanout or ac.p.large_net_threshold
        # the vectorised BFS keeps one dense (sources x registers) visited
        # array; above this budget the pure-Python BFS is used instead
        self.max_dense_bytes = 2_000_000_000
        self.N = nl.num_insts
        self.NB = len(nl.bterm_name)
        kinds = nl.inst_kind
        self.is_reg = [k == REGISTER for k in kinds] + [False] * self.NB
        self.is_dest = [k == MACRO for k in kinds] + [True] * self.NB
        self.is_comb = [k == COMB for k in kinds] + [False] * self.NB
        self.backend = None

    # -- stage-node helpers -------------------------------------------- #
    def _bnode(self, bid):
        return self.N + bid

    def _cluster_of(self, node):
        if node >= self.N:
            return self.ac.bterm2cl.get(node - self.N, -1)
        return self.ac.inst2cl[node]

    # -- driver / sink index ------------------------------------------- #
    def _build_index(self):
        nl, p = self.nl, self.p
        N = self.N
        self.out_nets = defaultdict(list)     # node -> [net_idx]
        self.net_sinks = [None] * len(nl.net_pins)
        thr = p.large_net_threshold
        for nidx, pins in enumerate(nl.net_pins):
            if len(pins) > thr:
                continue                      # clk / reset etc.
            driver = None
            sinks = []
            for kind, eid, is_drv in pins:
                node = eid if kind == INST else N + eid
                if is_drv:
                    driver = node
                else:
                    sinks.append(node)
            if driver is None or not sinks:
                continue
            self.out_nets[driver].append(nidx)
            self.net_sinks[nidx] = sinks

    # -- combinational cone reach ------------------------------------- #
    def _comb_reach_sets(self):
        """reach[c] = frozenset of stage nodes reachable from comb cell c's
        outputs (or BROADCAST).  Computed sinks-first along a topological
        order of the combinational DAG; cells on combinational loops (rare)
        fall back to a guarded DFS."""
        is_comb = self.is_comb
        out_nets = self.out_nets
        net_sinks = self.net_sinks
        max_fanout = self.max_fanout
        BROADCAST = self.BROADCAST

        comb_nodes = [u for u in out_nets if is_comb[u]]
        indeg = defaultdict(int)
        for u in comb_nodes:
            for nidx in out_nets[u]:
                for v in net_sinks[nidx]:
                    if is_comb[v]:
                        indeg[v] += 1
        q = deque(u for u in comb_nodes if indeg[u] == 0)
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for nidx in out_nets.get(u, ()):
                for v in net_sinks[nidx]:
                    if is_comb[v]:
                        indeg[v] -= 1
                        if indeg[v] == 0:
                            q.append(v)
        reach: dict[int, object] = {}
        empty = frozenset()

        def compute(u):
            acc = set()
            for nidx in out_nets.get(u, ()):
                for v in net_sinks[nidx]:
                    if is_comb[v]:
                        sub = reach.get(v, empty)
                        if sub is BROADCAST:
                            return BROADCAST
                        acc |= sub
                    else:
                        acc.add(v)
                if len(acc) > max_fanout:
                    return BROADCAST
            return frozenset(acc)

        for u in reversed(order):
            reach[u] = compute(u)
        self.n_comb_loops = 0
        if len(order) < len(comb_nodes):
            # combinational loops: process remaining cells in any order with
            # the partial results (break the loop conservatively)
            for u in comb_nodes:
                if u not in reach:
                    reach[u] = compute(u)
                    self.n_comb_loops += 1
        return reach

    def _build_stage_graph(self):
        """Forward stage-graph adjacency for every stage node."""
        reach = self._comb_reach_sets()
        is_comb = self.is_comb
        out_nets = self.out_nets
        net_sinks = self.net_sinks
        max_fanout = self.max_fanout
        BROADCAST = self.BROADCAST
        empty = frozenset()
        self.fwd: dict[int, dict[int, int]] = {}
        self.n_control_nets = 0
        stage_nodes = [u for u in out_nets if not is_comb[u]]
        for u in stage_nodes:
            out: dict[int, int] = defaultdict(int)
            for nidx in out_nets[u]:
                acc = set()
                broadcast = False
                for v in net_sinks[nidx]:
                    if is_comb[v]:
                        sub = reach.get(v, empty)
                        if sub is BROADCAST:
                            broadcast = True
                            break
                        acc |= sub
                    else:
                        acc.add(v)
                if broadcast or len(acc) > max_fanout:
                    self.n_control_nets += 1
                    continue
                for v in acc:
                    out[v] += 1
            if out:
                self.fwd[u] = dict(out)
        del reach

    def _build_prune(self):
        """g[node] = min extra register stages from node forward to reach any
        macro/IO endpoint (0-1 BFS on the reversed stage graph).  Used to
        prune dead-end register exploration (optimisation only)."""
        INF = self.max_hops + 1
        is_dest = self.is_dest
        is_reg = self.is_reg
        g = {}
        dq = deque()
        rev = defaultdict(list)
        for u, succ in self.fwd.items():
            hit = False
            for v in succ:
                rev[v].append(u)
                if is_dest[v]:
                    hit = True
            if hit:
                g[u] = 0
                dq.append(u)
        while dq:
            v = dq.popleft()
            nd = g[v] + (1 if is_reg[v] else 0)
            if nd > self.max_hops:
                continue
            for u in rev.get(v, ()):
                if nd < g.get(u, INF):
                    g[u] = nd
                    if is_reg[v]:
                        dq.append(u)
                    else:
                        dq.appendleft(u)
        self.g = g

    # -- BFS backends -------------------------------------------------- #
    def _sources(self):
        nl = self.nl
        srcs = [iid for iid in range(self.N) if nl.inst_kind[iid] == MACRO]
        srcs += [self.N + b for b in range(self.NB)]
        return srcs

    def _bfs_python(self, sources):
        """Per-source BFS.  Returns {(src, dst): {hop: flow}}."""
        max_h = self.max_hops
        g = self.g
        fwd = self.fwd
        is_dest = self.is_dest
        is_reg = self.is_reg
        INF = max_h + 1
        result = {}
        for src in sources:
            dests: dict[int, dict[int, float]] = {}
            best = {src: 0}
            q = deque([(src, 0)])
            while q:
                u, r = q.popleft()
                for v, w in fwd.get(u, {}).items():
                    if is_dest[v] and v != src:
                        dw = dests.get(v)
                        if dw is None:
                            dw = dests[v] = {}
                        dw[r] = dw.get(r, 0.0) + w
                    if is_reg[v]:
                        nr = r + 1
                        if nr > max_h or nr + g.get(v, INF) > max_h:
                            continue
                        if nr < best.get(v, INF):
                            best[v] = nr
                            q.append((v, nr))
            for v, hm in dests.items():
                result[(src, v)] = hm
        return result

    def _bfs_scipy(self, sources):
        """All-sources BFS with sparse matrix products (same semantics)."""
        import numpy as np
        import scipy.sparse as sp

        N = self.N + self.NB
        max_h = self.max_hops
        rows, cols, vals = [], [], []
        for u, succ in self.fwd.items():
            for v, w in succ.items():
                rows.append(u)
                cols.append(v)
                vals.append(w)
        A = sp.csr_matrix((np.asarray(vals, dtype=np.float32),
                           (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        dest_idx = np.asarray([i for i in range(N) if self.is_dest[i]])
        reg_idx = np.asarray([i for i in range(N) if self.is_reg[i]])
        D = A[:, dest_idx].tocsr()                   # N x Nd (weighted)
        R = A[:, reg_idx].tocsr()                    # N x Nr
        R.data[:] = 1.0
        g_reg = np.asarray([self.g.get(int(v), max_h + 1) for v in reg_idx])
        S = len(sources)
        src = np.asarray(sources)
        dpos = {int(v): k for k, v in enumerate(dest_idx)}

        F = sp.csr_matrix((np.ones(S, dtype=np.float32), (np.arange(S), src)),
                          shape=(S, N))
        visited = np.zeros((S, len(reg_idx)), dtype=bool)
        # self-reach (a macro feeding itself) is ignored, as in the BFS
        self_rows = np.asarray([k for k, s in enumerate(sources) if int(s) in dpos], dtype=int)
        self_cols = np.asarray([dpos[int(sources[k])] for k in self_rows], dtype=int)
        result = {}
        for k in range(max_h + 1):
            H = (F @ D).toarray()                    # dense S x Nd for this hop
            H[self_rows, self_cols] = 0.0
            si, di = np.nonzero(H)
            for s_, d_, w in zip(si.tolist(), di.tolist(), H[si, di].tolist()):
                key = (int(sources[s_]), int(dest_idx[d_]))
                hm = result.get(key)
                if hm is None:
                    hm = result[key] = {}
                hm[k] = w
            del H
            if k == max_h:
                break
            Fn = (F @ R).tocoo()
            r, c = Fn.row, Fn.col
            keep = (~visited[r, c]) & (g_reg[c] + (k + 1) <= max_h)
            r, c = r[keep], c[keep]
            if r.size == 0:
                break
            # a (source, register) pair may appear several times in the COO
            # product; de-duplicate before marking visited
            flat = np.unique(r.astype(np.int64) * len(reg_idx) + c)
            r = flat // len(reg_idx)
            c = flat % len(reg_idx)
            visited[r, c] = True
            F = sp.csr_matrix((np.ones(r.size, dtype=np.float32), (r, reg_idx[c])),
                              shape=(S, N))
        return result

    # -- main ---------------------------------------------------------- #
    def run(self, verbose=True):
        t0 = time.time()
        self._build_index()
        self._build_stage_graph()
        if verbose:
            print(f"  [dataflow] stage graph: {len(self.fwd)} nodes, "
                  f"{sum(len(s) for s in self.fwd.values())} edges, "
                  f"{self.n_control_nets} control nets skipped, "
                  f"{time.time()-t0:.1f}s", file=sys.stderr)
        t1 = time.time()
        self._build_prune()
        if verbose:
            reachable = sum(1 for v in self.g.values() if v <= self.max_hops)
            print(f"  [dataflow] prune map: {reachable} nodes within "
                  f"{self.max_hops} hops of an endpoint, {time.time()-t1:.1f}s",
                  file=sys.stderr)

        sources = self._sources()
        t2 = time.time()
        pairs = None
        n_reg = sum(1 for v in self.is_reg if v)
        n_dest = sum(1 for v in self.is_dest if v)
        dense_bytes = len(sources) * (n_reg + 4 * n_dest)   # visited bool + one float32 hop
        if self.use_scipy and dense_bytes <= self.max_dense_bytes:
            try:
                pairs = self._bfs_scipy(sources)
                self.backend = "scipy"
            except ImportError:
                pairs = None
        if pairs is None:
            pairs = self._bfs_python(sources)
            self.backend = "python"
        if verbose:
            print(f"  [dataflow] BFS ({self.backend}): {len(sources)} sources, "
                  f"{len(pairs)} (src,dst) pairs, {time.time()-t2:.1f}s",
                  file=sys.stderr)

        # Directed cluster-pair accumulation.  The BFS source ``csrc`` is the
        # driver side and ``cdst`` the downstream sink, so the key (csrc, cdst)
        # encodes the dataflow direction csrc -> cdst.
        virt_dir: dict[tuple[int, int], float] = defaultdict(float)
        self.hop_hist = defaultdict(int)
        self.pair_flows = pairs          # raw (src, dst) -> {hop: flow}
        for (src, dst), hopmap in pairs.items():
            csrc = self._cluster_of(src)
            cdst = self._cluster_of(dst)
            if csrc < 0 or cdst < 0 or cdst == csrc:
                continue
            if self.mode == "min":
                h = min(hopmap)
                self.hop_hist[h] += 1
                virt_dir[(csrc, cdst)] += hopmap[h] / (2 ** h)
            else:
                self.hop_hist[min(hopmap)] += 1
                virt_dir[(csrc, cdst)] += sum(f / (2 ** h) for h, f in hopmap.items())

        # directed edges: driver-cluster -> sink-cluster
        self.virtual_edges_directed = dict(virt_dir)

        # undirected view: fold both directions into one (min, max) edge.
        virt: dict[tuple[int, int], float] = defaultdict(float)
        for (a, b), w in virt_dir.items():
            key = (a, b) if a < b else (b, a)
            virt[key] += w
        self.virtual_edges = dict(virt)

        # write virtual connections back onto clusters (symmetric affinity)
        for (a, b), w in virt.items():
            self.ac.clusters[a].virtual[b] += w
            self.ac.clusters[b].virtual[a] += w
        return self.virtual_edges
