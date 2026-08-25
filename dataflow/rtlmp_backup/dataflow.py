"""Dataflow / virtual connections - RTL-MP Section 4 Step 6, Eq. (1).

    Virtual_Connections(A, B) = Information_Flow(A, B) / 2 ** Num_Hops

  * Information_Flow  ~  connection bitwidth (number of parallel wires).
  * Num_Hops          =  number of register (flip-flop) stages on the
                         shortest register path between A and B.
  * Buffer / comb-cell "transparency": only registers count as hops; all
    combinational cells (including buffers/inverters) are traced through.
  * If Num_Hops > 4 no virtual connection is added (aggressive decay).
  * Only macro clusters and bundled IO pins are considered (Section 4).

The register-stage graph is built once:

    stage node  := a MACRO instance, a REGISTER instance, or an IO BTerm.
    edge u -> v := v is reachable from u's outputs through a cone of purely
                   combinational cells (memoized ``_comb_reach``).
                   Edge weight = number of wires (bitwidth proxy).

From every macro / IO source we then BFS over this stage graph, counting a
hop each time we relay through a register, accumulating weighted flow into
the (macro/pin) cluster-pair virtual connections.
"""

from __future__ import annotations

from collections import defaultdict, deque

from .netlist import BTERM, COMB, INST, MACRO, REGISTER, Netlist


class Dataflow:
    # marker returned by _comb_reach for a "broadcast" cone (control / global
    # signal) that fans out to more than ``max_fanout`` stage points - such a
    # cone carries no meaningful dataflow bitwidth and is not propagated.
    BROADCAST = object()

    def __init__(self, nl: Netlist, ac, max_hops: int = 4, max_fanout=None):
        self.nl = nl
        self.ac = ac
        self.p = ac.p
        self.max_hops = max_hops
        # a stage node reaching more sinks than this is a broadcast/control
        # signal, not a dataflow bus (default: the global-net threshold).
        self.max_fanout = max_fanout or ac.p.large_net_threshold
        self.N = nl.num_insts
        # unified stage-node id: inst -> iid ; bterm -> N + bid
        self._comb_memo: dict[int, object] = {}
        self._building: set[int] = set()

    # -- stage-node helpers -------------------------------------------- #
    def _bnode(self, bid):
        return self.N + bid

    def _is_stage(self, node):
        if node >= self.N:
            return True                       # IO pin is always a stage point
        return self.nl.inst_kind[node] in (MACRO, REGISTER)

    def _cluster_of(self, node):
        if node >= self.N:
            return self.ac.bterm2cl.get(node - self.N, -1)
        return self.ac.inst2cl[node]

    # -- driver / sink index ------------------------------------------- #
    def _build_index(self):
        nl, p = self.nl, self.p
        # nets a node drives, and each net's sinks (as stage/comb nodes)
        self.out_nets = defaultdict(list)     # node -> [net_idx]
        self.net_sinks = [None] * len(nl.net_pins)
        for nidx, pins in enumerate(nl.net_pins):
            if len(pins) > p.large_net_threshold:
                continue                      # clk / reset etc.
            driver = None
            sinks = []
            for kind, eid, is_drv in pins:
                node = eid if kind == INST else self._bnode(eid)
                if is_drv:
                    driver = node
                else:
                    sinks.append(node)
            if driver is None or not sinks:
                continue
            self.out_nets[driver].append(nidx)
            self.net_sinks[nidx] = sinks

    # -- combinational cone reach (transparent tracing) ---------------- #
    def _comb_reach(self, node):
        """Stage nodes reachable from a *combinational* node's outputs.

        Returns {stage_node: wire_weight}, or ``BROADCAST`` if the cone grows
        beyond ``max_fanout`` (control / global signal).  Memoized; guards
        combinational loops.
        """
        memo = self._comb_memo.get(node)
        if memo is not None:
            return memo
        if node in self._building:            # combinational loop - break it
            return {}
        self._building.add(node)
        out: dict[int, int] = defaultdict(int)
        broadcast = False
        for nidx in self.out_nets.get(node, ()):
            sinks = self.net_sinks[nidx]
            if not sinks:
                continue
            for s in sinks:
                if self._is_stage(s):
                    out[s] += 1
                else:                          # another comb cell -> recurse
                    sub = self._comb_reach(s)
                    if sub is self.BROADCAST:
                        broadcast = True
                        break
                    for st, w in sub.items():
                        out[st] += w
            if broadcast or len(out) > self.max_fanout:
                broadcast = True
                break
        self._building.discard(node)
        result = self.BROADCAST if broadcast else dict(out)
        self._comb_memo[node] = result
        return result

    def _forward(self, stage_node) -> dict[int, int]:
        """Direct stage-graph successors of a stage node, with wire weights.

        Returns {} for broadcast/control drivers (fanout > max_fanout) so they
        do not create dense, meaningless dataflow edges.
        """
        out: dict[int, int] = defaultdict(int)
        for nidx in self.out_nets.get(stage_node, ()):
            sinks = self.net_sinks[nidx]
            if not sinks:
                continue
            for s in sinks:
                if self._is_stage(s):
                    out[s] += 1
                else:
                    sub = self._comb_reach(s)
                    if sub is self.BROADCAST:
                        return {}
                    for st, w in sub.items():
                        out[st] += w
            if len(out) > self.max_fanout:
                return {}
        return dict(out)

    def _build_stage_graph(self):
        """Forward stage-graph adjacency for every stage node."""
        nl = self.nl
        self.fwd: dict[int, dict[int, int]] = {}
        # macros, registers
        for iid in range(self.N):
            if nl.inst_kind[iid] in (MACRO, REGISTER):
                f = self._forward(iid)
                if f:
                    self.fwd[iid] = f
        # IO input ports act as sources/relays
        for bid in range(len(nl.bterm_name)):
            node = self._bnode(bid)
            f = self._forward(node)
            if f:
                self.fwd[node] = f
        # combinational memo no longer needed -> free memory
        self._comb_memo.clear()

    def _build_prune(self):
        """g[node] = min extra register stages from node forward to reach
        any macro/IO endpoint.  Used to prune dead-end register exploration
        during the per-source BFS (0-1 BFS on the reversed stage graph).
        """
        INF = self.max_hops + 1
        g = defaultdict(lambda: INF)
        dq = deque()
        # seed: any node with a direct forward edge to an endpoint can reach
        # it with 0 extra register stages.
        for u, succ in self.fwd.items():
            for v in succ:
                if self._is_dest(v):
                    if g[u] > 0:
                        g[u] = 0
                        dq.appendleft(u)
                    break
        # reversed adjacency
        rev = defaultdict(list)
        for u, succ in self.fwd.items():
            for v in succ:
                rev[v].append(u)
        while dq:
            v = dq.popleft()
            gv = g[v]
            # cost to step onto v from a predecessor: +1 if v is a register
            step = 1 if self.nl_kind_is_register(v) else 0
            nd = gv + step
            if nd > self.max_hops:
                continue
            for u in rev.get(v, ()):
                if nd < g[u]:
                    g[u] = nd
                    if step:
                        dq.append(u)
                    else:
                        dq.appendleft(u)
        self.g = g

    # -- BFS over register stages from one source ---------------------- #
    def _bfs_from(self, src, dests_weight: dict[int, dict[int, float]]):
        """BFS counting register hops; record (dest -> hop -> flow).

        ``dests_weight[dst][hop] += flow`` for every macro/IO dst reached.
        """
        max_h = self.max_hops
        g = self.g
        # best[node] = smallest register-hop at which we expanded it
        best = {src: 0}
        q = deque([(src, 0)])
        while q:
            u, r = q.popleft()
            if r > best.get(u, max_h + 1):
                continue
            for v, w in self.fwd.get(u, {}).items():
                if self._is_dest(v) and v != src:
                    # macro/IO terminal reached after r register stages
                    dw = dests_weight[v]
                    dw[r] = dw.get(r, 0.0) + w
                if self.nl_kind_is_register(v):
                    nr = r + 1
                    # prune: can v still reach an endpoint within budget?
                    if nr > max_h or nr + g.get(v, max_h + 1) > max_h:
                        continue
                    if nr < best.get(v, max_h + 1):
                        best[v] = nr
                        q.append((v, nr))

    # convenience predicates ------------------------------------------- #
    def _is_dest(self, node):
        if node >= self.N:
            return True                       # IO pin
        return self.nl.inst_kind[node] == MACRO

    def nl_kind_is_register(self, node):
        return node < self.N and self.nl.inst_kind[node] == REGISTER

    # -- main ---------------------------------------------------------- #
    def run(self, verbose=True):
        import sys
        import time
        t0 = time.time()
        self._build_index()
        self._build_stage_graph()
        if verbose:
            print(f"  [dataflow] stage graph: {len(self.fwd)} nodes, "
                  f"{time.time()-t0:.1f}s", file=sys.stderr)
        t1 = time.time()
        self._build_prune()
        if verbose:
            reachable = sum(1 for v in self.g.values() if v <= self.max_hops)
            print(f"  [dataflow] prune map: {reachable} nodes within "
                  f"{self.max_hops} hops of an endpoint, {time.time()-t1:.1f}s",
                  file=sys.stderr)

        nl = self.nl
        # sources = macros + IO pins
        sources = [iid for iid in range(self.N) if nl.inst_kind[iid] == MACRO]
        sources += [self._bnode(b) for b in range(len(nl.bterm_name))]

        # Directed cluster-pair accumulation.  The BFS source ``csrc`` is the
        # driver side and ``cdst`` the downstream sink, so the key (csrc, cdst)
        # encodes the dataflow direction csrc -> cdst.
        virt_dir: dict[tuple[int, int], float] = defaultdict(float)
        # also keep raw per-hop breakdown for reporting
        self.hop_hist = defaultdict(int)

        for src in sources:
            csrc = self._cluster_of(src)
            if csrc < 0:
                continue
            dests_weight: dict[int, dict[int, float]] = defaultdict(dict)
            self._bfs_from(src, dests_weight)
            for dst, hopmap in dests_weight.items():
                cdst = self._cluster_of(dst)
                if cdst < 0 or cdst == csrc:
                    continue
                # shortest register-distance dominates (aggressive decay)
                h = min(hopmap)
                flow = hopmap[h]
                contrib = flow / (2 ** h)
                self.hop_hist[h] += 1
                virt_dir[(csrc, cdst)] += contrib

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
