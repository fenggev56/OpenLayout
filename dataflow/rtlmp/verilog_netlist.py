"""Structural Verilog + LEF frontend for the RTL-MP clustering engine.

The OpenDB frontend in :mod:`rtlmp.netlist` is useful inside OpenROAD, but the
MacroPlacement testcases also ship synthesized structural Verilog.  This module
parses the subset used by those files:

* one flattened top module,
* scalar/vector input/output/inout declarations,
* named-port primitive and macro instances,
* no behavioral statements or assigns.

LEF files provide cell area, macro-vs-stdcell class, and pin directions.  When
that metadata is missing, conservative name-based heuristics keep the graph
usable for clustering.
"""

from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .netlist import BTERM, COMB, INST, MACRO, REGISTER, Netlist


_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.M | re.S)
_DECL_RE = re.compile(
    r"\b(input|output|inout)\b\s+(?:wire\s+|reg\s+)?"
    r"(?:(\[[^\]]+\])\s+)?([^;]+);",
    re.S,
)
_TOKEN_RE = re.compile(
    r"\\[^\s,{}()]+|[A-Za-z_$][A-Za-z0-9_$]*(?:\[[0-9]+\])?"
)
_INSTANCE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+((?:\\\S+)|\S+)\s*\(")
_BUS_RE = re.compile(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]")
_BIT_RE = re.compile(r"^(.*)\[(-?\d+)\]$")
# sized / unsized constant literals (1'b0, 32'h0000_0000, 'd5) - not nets
_LITERAL_RE = re.compile(r"\d*\s*'\s*[sS]?[bBoOdDhH]\s*[0-9a-fA-FxXzZ_?]+")

_SKIP_STMT_PREFIXES = (
    "module ", "endmodule", "input ", "output ", "inout ", "wire ",
    "reg ", "supply0 ", "supply1 ", "tri ", "assign ", "parameter ",
)

_FALLBACK_OUTPUT_PINS = {
    "Q", "QN", "Z", "ZN", "Y", "CO", "S", "SO", "SUM", "COUT", "O",
}


class MasterInfo:
    __slots__ = ("name", "width", "height", "is_block", "pin_dirs")

    def __init__(self, name: str):
        self.name = name
        self.width = 1.0
        self.height = 1.0
        self.is_block = False
        self.pin_dirs: dict[str, str] = {}


def load(
    verilog_path: str,
    lef_paths: Iterable[str] | None = None,
    def_path: str | None = None,
    cache_path: str | None = None,
    rebuild: bool = False,
) -> Netlist:
    """Load a structural Verilog testcase as a :class:`Netlist`."""
    if cache_path and os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    masters = parse_lefs(lef_paths or [])
    nl = build_from_verilog(verilog_path, masters)
    if def_path and os.path.exists(def_path):
        apply_def_geometry(nl, def_path)
    else:
        distribute_bterms(nl)

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(nl, f, protocol=pickle.HIGHEST_PROTOCOL)
    return nl


def discover_lefs(verilog_path: str) -> list[str]:
    """Return sibling ``lef/*.lef`` files for a benchmark design."""
    lef_dir = Path(verilog_path).resolve().parent / "lef"
    if not lef_dir.is_dir():
        return []
    return sorted(str(p) for p in lef_dir.glob("*.lef"))


def discover_def(verilog_path: str) -> str | None:
    """Return sibling ``def/pre_place.def`` when present."""
    p = Path(verilog_path).resolve().parent / "def" / "pre_place.def"
    return str(p) if p.exists() else None


def parse_lefs(paths: Iterable[str]) -> dict[str, MasterInfo]:
    masters: dict[str, MasterInfo] = {}
    for path in paths:
        current: MasterInfo | None = None
        pin: str | None = None
        with open(path, errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                toks = line.replace(";", "").split()
                if len(toks) >= 2 and toks[0] == "MACRO":
                    current = masters.setdefault(toks[1], MasterInfo(toks[1]))
                    pin = None
                    continue
                if current is None:
                    continue
                if len(toks) >= 2 and toks[0] == "END" and toks[1] == current.name:
                    current = None
                    pin = None
                    continue
                if len(toks) >= 2 and toks[0] == "CLASS":
                    current.is_block = toks[1].upper() == "BLOCK"
                elif len(toks) >= 4 and toks[0] == "SIZE":
                    current.width = float(toks[1])
                    current.height = float(toks[3])
                elif len(toks) >= 2 and toks[0] == "PIN":
                    pin = toks[1]
                elif len(toks) >= 2 and toks[0] == "DIRECTION" and pin:
                    current.pin_dirs[pin] = toks[1].upper()
                    base = _base_pin(pin)
                    current.pin_dirs.setdefault(base, toks[1].upper())
                elif len(toks) >= 2 and toks[0] == "END" and pin == toks[1]:
                    pin = None
    return masters


def build_from_verilog(path: str, masters: dict[str, MasterInfo]) -> Netlist:
    nl = Netlist()
    text = Path(path).read_text(errors="ignore")
    text = _COMMENT_RE.sub("", text)

    module_match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text)
    if module_match:
        nl.design_name = module_match.group(1)
    nl.die = (0.0, 0.0, 1000.0, 1000.0)

    port_bits = _add_bterms(nl, text)
    nets: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    for bid, name in enumerate(nl.bterm_name):
        is_driver = nl.bterm_io[bid] == "INPUT"
        nets[name].append((BTERM, bid, is_driver))

    for stmt in text.split(";"):
        s = stmt.strip()
        if not s or s.startswith(_SKIP_STMT_PREFIXES):
            continue
        m = _INSTANCE_RE.match(s)
        if not m:
            continue
        master, inst_name = m.group(1), _clean_name(m.group(2))
        body_start = m.end() - 1
        body = s[body_start + 1:]
        if body.endswith(")"):
            body = body[:-1]
        iid = _add_inst(nl, master, inst_name, masters)
        for port, expr in _parse_named_ports(body):
            direction = _pin_direction(master, port, masters)
            if direction == "INOUT":
                continue
            is_driver = direction == "OUTPUT"
            for net_name in _expr_nets(expr, port_bits):
                nets[net_name].append((INST, iid, is_driver))

    nl.net_name = []
    nl.net_pins = []
    for name, pins in nets.items():
        if len(pins) < 2:
            continue
        if not any(is_drv for _, _, is_drv in pins):
            continue
        nl.net_name.append(name)
        nl.net_pins.append(pins)
    return nl


def _add_bterms(nl: Netlist, text: str) -> dict[str, list[str]]:
    port_bits: dict[str, list[str]] = {}
    for m in _DECL_RE.finditer(text):
        direction = m.group(1).upper()
        bus = m.group(2)
        names = [n.strip() for n in m.group(3).replace("\n", " ").split(",")]
        bits = _bus_bits(bus)
        for raw_name in names:
            name = _clean_name(raw_name)
            if not name:
                continue
            expanded = [f"{name}[{b}]" for b in bits] if bits else [name]
            port_bits[name] = expanded
            for bname in expanded:
                if bname in nl.bterm_id:
                    continue
                bid = len(nl.bterm_name)
                nl.bterm_id[bname] = bid
                nl.bterm_name.append(bname)
                nl.bterm_io.append(direction)
                nl.bterm_xy.append((0.0, 0.0))
    return port_bits


def _add_inst(
    nl: Netlist,
    master: str,
    name: str,
    masters: dict[str, MasterInfo],
) -> int:
    info = masters.get(master)
    if info and info.is_block:
        kind = MACRO
    elif _is_macro_name(master):
        kind = MACRO
    elif nl.is_seq_master(master):
        kind = REGISTER
    else:
        kind = COMB
    w = info.width if info else (50.0 if kind == MACRO else 1.0)
    h = info.height if info else (50.0 if kind == MACRO else 1.0)
    iid = len(nl.inst_name)
    nl.inst_id[name] = iid
    nl.inst_name.append(name)
    nl.inst_master.append(master)
    nl.inst_kind.append(kind)
    nl.inst_is_buf.append(nl.is_buf_master(master))
    nl.inst_w.append(w)
    nl.inst_h.append(h)
    nl.inst_area.append(w * h)
    return iid


def _parse_named_ports(body: str):
    i, n = 0, len(body)
    while i < n:
        dot = body.find(".", i)
        if dot < 0:
            return
        i = dot + 1
        start = i
        while i < n and (body[i].isalnum() or body[i] in "_$[]"):
            i += 1
        port = body[start:i]
        while i < n and body[i].isspace():
            i += 1
        if i >= n or body[i] != "(":
            continue
        i += 1
        depth = 1
        expr_start = i
        while i < n and depth:
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
            i += 1
        yield port, body[expr_start:i - 1]


def _expr_nets(expr: str, port_bits: dict[str, list[str]]) -> list[str]:
    nets: list[str] = []
    if "'" in expr:
        expr = _LITERAL_RE.sub(" ", expr)
    for tok in _TOKEN_RE.findall(expr):
        name = _clean_name(tok)
        if not name:
            continue
        expanded = port_bits.get(name)
        if expanded:
            nets.extend(expanded)
        else:
            nets.append(name)
    return nets


def _pin_direction(
    master: str,
    port: str,
    masters: dict[str, MasterInfo],
) -> str:
    info = masters.get(master)
    if info:
        direction = info.pin_dirs.get(port) or info.pin_dirs.get(_base_pin(port))
        if direction:
            return direction
    base = _base_pin(port)
    lname = base.lower()
    if base in _FALLBACK_OUTPUT_PINS or lname.endswith(("_o", "_out")):
        return "OUTPUT"
    if "rd_out" in lname or lname in {"dout", "data_out"}:
        return "OUTPUT"
    return "INPUT"


def _bus_bits(bus: str | None) -> list[int]:
    if not bus:
        return []
    m = _BUS_RE.search(bus)
    if not m:
        return []
    a, b = int(m.group(1)), int(m.group(2))
    step = 1 if b >= a else -1
    return list(range(a, b + step, step))


def _clean_name(name: str) -> str:
    name = name.strip().rstrip(",")
    if name.startswith("\\"):
        name = name[1:]
    return name.replace("\\[", "[").replace("\\]", "]")


def _base_pin(pin: str) -> str:
    m = _BIT_RE.match(pin)
    return m.group(1) if m else pin


def _is_macro_name(master: str) -> bool:
    return master.startswith(("fakeram", "sram", "SRAM", "mem_", "ariane_regfile"))


def apply_def_geometry(nl: Netlist, def_path: str):
    """Apply die area and top-pin coordinates from a DEF, when available."""
    text = Path(def_path).read_text(errors="ignore")
    units = 1.0
    m = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;", text)
    if m:
        units = float(m.group(1))
    m = re.search(
        r"DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+"
        r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
        text,
    )
    if m:
        nl.die = tuple(float(v) / units for v in m.groups())

    pin_xy = {}
    pins_section = re.search(r"\bPINS\s+\d+\s*;(.*?)\bEND\s+PINS\b", text, re.S)
    pin_text = pins_section.group(1) if pins_section else ""
    for stmt in re.findall(r"\n\s*-\s+(.+?)\s*;", pin_text, re.S):
        name = _clean_def_name(stmt.split()[0])
        loc = re.search(r"\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)", stmt)
        if loc:
            pin_xy[name] = (float(loc.group(1)) / units, float(loc.group(2)) / units)
    hit = 0
    for bid, bname in enumerate(nl.bterm_name):
        xy = pin_xy.get(bname)
        if xy is not None:
            nl.bterm_xy[bid] = xy
            hit += 1
    if hit < len(nl.bterm_name):
        distribute_bterms(nl, keep_existing=True)


def _clean_def_name(name: str) -> str:
    return name.replace("\\[", "[").replace("\\]", "]")


def distribute_bterms(nl: Netlist, keep_existing: bool = False):
    """Spread IO pins around the die perimeter for Verilog-only runs."""
    lx, ly, ux, uy = nl.die
    n = max(1, len(nl.bterm_name))
    per_edge = max(1, (n + 3) // 4)
    for bid in range(len(nl.bterm_name)):
        if keep_existing and nl.bterm_xy[bid] != (0.0, 0.0):
            continue
        side = min(3, bid // per_edge)
        pos = (bid % per_edge + 0.5) / per_edge
        if side == 0:
            xy = (lx + (ux - lx) * pos, ly)
        elif side == 1:
            xy = (ux, ly + (uy - ly) * pos)
        elif side == 2:
            xy = (ux - (ux - lx) * pos, uy)
        else:
            xy = (lx, uy - (uy - ly) * pos)
        nl.bterm_xy[bid] = xy
