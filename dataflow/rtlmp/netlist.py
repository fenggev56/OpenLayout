"""Netlist extraction from OpenDB into a compact, picklable data model.

This is the substrate that the RTL-MP autoclustering (Algorithm 1) and
dataflow engine (Eq. 1) operate on.  We deliberately flatten OpenDB into
plain Python arrays so that:

  * the heavy graph algorithms run without repeatedly crossing the
    C++/Python boundary, and
  * the extracted netlist can be cached to disk (pickle) for fast
    iteration during development.

Cell classification
-------------------
Every leaf instance is one of:
  * MACRO     - master.isBlock()            (fakeram, regfile, fifo, ...)
  * REGISTER  - sequential std cell         (flip-flop / latch)
  * COMB      - everything else             (combinational std cell, buffers)

Buffers/inverters are a sub-class of COMB; the dataflow engine treats them
as "transparent" (Section 4, Step 2 (iv) and Section 6 of the paper).
"""

from __future__ import annotations

import os
import pickle
import re

# Liberty for the Nangate45 library only contains these sequential families.
# Kept as a regex so other PDKs can be supported by extending the pattern.
_SEQ_RE = re.compile(r"^(DFF|SDFF|DLH|DLL|TLAT|LH|LL|SR)")
# Buffers / inverters - "transparent" elements for dataflow tracing.
_BUF_RE = re.compile(r"^(BUF|INV|CLKBUF|HB|TBUF|TINV)")

# Instance kinds
MACRO = "MACRO"
REGISTER = "REGISTER"
COMB = "COMB"

# Pin endpoint kinds
INST = 0
BTERM = 1


class Netlist:
    """Compact, picklable representation of a flat gate-level netlist."""

    def __init__(self):
        # --- per-instance arrays (index == instance id) -------------------
        self.inst_name: list[str] = []
        self.inst_master: list[str] = []
        self.inst_kind: list[str] = []          # MACRO / REGISTER / COMB
        self.inst_is_buf: list[bool] = []
        self.inst_w: list[float] = []           # microns
        self.inst_h: list[float] = []
        self.inst_area: list[float] = []         # micron^2

        # --- per-IO-pin arrays (index == bterm id) ------------------------
        self.bterm_name: list[str] = []
        self.bterm_io: list[str] = []            # "INPUT" / "OUTPUT" / "INOUT"
        self.bterm_xy: list[tuple[float, float]] = []  # microns

        # --- nets ---------------------------------------------------------
        # For each net: a list of pins, each pin a tuple
        #   (kind, id, is_driver)   kind in {INST, BTERM}
        # A driver sources the net (instance OUTPUT pin, or INPUT IO port).
        self.net_name: list[str] = []
        self.net_pins: list[list[tuple]] = []

        # die area in microns: (lx, ly, ux, uy)
        self.die: tuple[float, float, float, float] = (0, 0, 0, 0)

        # name -> id lookups
        self.inst_id: dict[str, int] = {}
        self.bterm_id: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    @property
    def num_insts(self) -> int:
        return len(self.inst_name)

    @property
    def num_macros(self) -> int:
        return sum(1 for k in self.inst_kind if k == MACRO)

    # ------------------------------------------------------------------ #
    def is_seq_master(self, name: str) -> bool:
        return bool(_SEQ_RE.match(name))

    def is_buf_master(self, name: str) -> bool:
        return bool(_BUF_RE.match(name))


# ---------------------------------------------------------------------- #
def build_from_db(block) -> Netlist:
    """Traverse an OpenDB block and build a :class:`Netlist`."""
    nl = Netlist()
    dbu = block.getDbUnitsPerMicron()

    r = block.getDieArea()
    nl.die = (r.xMin() / dbu, r.yMin() / dbu, r.xMax() / dbu, r.yMax() / dbu)

    # Instances ---------------------------------------------------------
    for inst in block.getInsts():
        m = inst.getMaster()
        mname = m.getName()
        w = m.getWidth() / dbu
        h = m.getHeight() / dbu
        if m.isBlock():
            kind = MACRO
        elif nl.is_seq_master(mname):
            kind = REGISTER
        else:
            kind = COMB
        iid = len(nl.inst_name)
        nl.inst_id[inst.getName()] = iid
        nl.inst_name.append(inst.getName())
        nl.inst_master.append(mname)
        nl.inst_kind.append(kind)
        nl.inst_is_buf.append(nl.is_buf_master(mname))
        nl.inst_w.append(w)
        nl.inst_h.append(h)
        nl.inst_area.append(w * h)

    # IO pins (block terminals) ----------------------------------------
    for bt in block.getBTerms():
        bid = len(nl.bterm_name)
        nl.bterm_id[bt.getName()] = bid
        nl.bterm_name.append(bt.getName())
        nl.bterm_io.append(str(bt.getIoType()))
        loc = bt.getFirstPinLocation()  # [valid, x, y]
        if loc[0]:
            nl.bterm_xy.append((loc[1] / dbu, loc[2] / dbu))
        else:
            nl.bterm_xy.append((0.0, 0.0))

    # Nets --------------------------------------------------------------
    for net in block.getNets():
        if net.isSpecial():
            continue
        pins = []
        for it in net.getITerms():
            io = str(it.getIoType())
            if io == "INOUT":          # power/ground stubs - ignore
                continue
            iid = nl.inst_id.get(it.getInst().getName())
            if iid is None:
                continue
            pins.append((INST, iid, io == "OUTPUT"))
        for bt in net.getBTerms():
            io = str(bt.getIoType())
            bid = nl.bterm_id.get(bt.getName())
            if bid is None:
                continue
            # An INPUT port feeds the core, i.e. it drives the net.
            pins.append((BTERM, bid, io == "INPUT"))
        if len(pins) < 2:
            continue
        nl.net_name.append(net.getName())
        nl.net_pins.append(pins)

    return nl


# ---------------------------------------------------------------------- #
def load(cache_path: str | None = None, rebuild: bool = False) -> Netlist:
    """Load the ariane81 netlist, using a pickle cache when available."""
    if cache_path and os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import odb  # noqa: F401  (ensures odb is importable under openroad -python)
    from openroad import Tech, Design

    data = "Benchmark/PlaceReasoner-Bench"
    a81 = f"{data}/ariane81"
    pdk = f"{data}/ariane133"

    tech = Tech()
    tech.readLef(f"{pdk}/lef/NangateOpenCellLibrary.tech.lef")
    tech.readLef(f"{pdk}/lef/NangateOpenCellLibrary.macro.mod.lef")
    for lef in ("fakeram45_256x16", "fakeram45_256x32", "fakeram45_256x48",
                "ariane_regfile", "fifo_v3_1", "fifo_v3_8"):
        tech.readLef(f"{a81}/lef/{lef}.lef")

    design = Design(tech)
    design.readDef(f"{a81}/def/pre_place.def")
    nl = build_from_db(design.getBlock())

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(nl, f, protocol=pickle.HIGHEST_PROTOCOL)
    return nl
