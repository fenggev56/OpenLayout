"""Deterministic placement geometry.

The agents decide placement intent and may emit direct macro coordinates; these
pure functions parse, validate, score, legalize, and transform those positions.
Macro positions are read back from OpenROAD or represented as indexed entries so
the code can clamp, reject degenerate model output, and repair overlaps before
the expensive placement stages run.

Parsing helpers (core bounds, macro sizes, current positions) read the output
of the :mod:`macroflow.recipes` TCL queries.
"""

from __future__ import annotations

import re
import sys
import ast
import operator
from pathlib import Path

# legalize.py is a repo-root module (NOT part of openlayout). Make sure the repo
# root is importable whether macroflow is imported as a package or run directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from legalize import legalize_plan_entries  # noqa: E402

DEFAULT_MW = 77.71   # fakeram45_256x16 — the ariane reference footprint
DEFAULT_MH = 40.60
CORE_SNAP_MARGIN = 0.10

VALID_ORIENTS = {"R0", "R90", "R180", "R270", "MX", "MY", "MXR90", "MYR90"}

# Anchor name -> fractional (fx, fy) target inside the core. Compaction pulls
# the macro cluster toward this point so the freed contiguous space ends up
# where the stdcells should go.
COMPACT_ANCHORS = {
    "center":       (0.50, 0.50),
    "bottom-left":  (0.25, 0.25),
    "bottom-right": (0.75, 0.25),
    "top-left":     (0.25, 0.75),
    "top-right":    (0.75, 0.75),
    "bottom":       (0.50, 0.25),
    "top":          (0.50, 0.75),
    "left":         (0.25, 0.50),
    "right":        (0.75, 0.50),
}


# ---------------------------------------------------------------------------
# Parsers (read recipe output)
# ---------------------------------------------------------------------------
def parse_core_bounds(dims_text: str) -> tuple[float, float, float, float]:
    """Extract (cx0, cy0, cx1, cy1) from recipes.DIE_CORE output."""
    cx0_m = re.search(r"Core origin\s*=\s*\(([\d.]+),\s*([\d.]+)\)", dims_text)
    cx1_m = re.search(r"Core x_max\s*=\s*([\d.]+)", dims_text)
    cy1_m = re.search(r"Core y_max\s*=\s*([\d.]+)", dims_text)
    cx0 = float(cx0_m.group(1)) if cx0_m else 10.07
    cy0 = float(cx0_m.group(2)) if cx0_m else 12.6
    cx1 = float(cx1_m.group(1)) if cx1_m else 1447.99
    cy1 = float(cy1_m.group(1)) if cy1_m else 1447.6
    return cx0, cy0, cx1, cy1


def parse_die_bounds(dims_text: str) -> tuple[float, float, float, float]:
    """Extract (dx0, dy0, dx1, dy1) from recipes.DIE_CORE output."""
    dx0_m = re.search(r"Die\s+origin\s*=\s*\(([\d.]+),\s*([\d.]+)\)", dims_text)
    dx1_m = re.search(r"Die\s+x_max\s*=\s*([\d.]+)", dims_text)
    dy1_m = re.search(r"Die\s+y_max\s*=\s*([\d.]+)", dims_text)
    dw_m = re.search(r"Die\s+width\s*=\s*([\d.]+)", dims_text)
    dh_m = re.search(r"Die\s+height\s*=\s*([\d.]+)", dims_text)
    dx0 = float(dx0_m.group(1)) if dx0_m else 0.0
    dy0 = float(dx0_m.group(2)) if dx0_m else 0.0
    dx1 = float(dx1_m.group(1)) if dx1_m else dx0 + (float(dw_m.group(1)) if dw_m else 0.0)
    dy1 = float(dy1_m.group(1)) if dy1_m else dy0 + (float(dh_m.group(1)) if dh_m else 0.0)
    return dx0, dy0, dx1, dy1


def parse_macro_sizes(macros_text: str) -> dict[str, tuple[float, float]]:
    """name -> (w, h). Name is everything before '  master=' so escaped bus
    brackets survive (else size lookups fall back to the fakeram default)."""
    sizes: dict[str, tuple[float, float]] = {}
    for line in macros_text.splitlines():
        if "master=" not in line or "size=" not in line:
            continue
        name = line.split("master=", 1)[0].strip()
        sm = re.search(r"size=([\d.]+)\s*x\s*([\d.]+)", line)
        if name and sm:
            sizes[name] = (float(sm.group(1)), float(sm.group(2)))
    return sizes


def parse_macro_positions(macros_text: str) -> list[dict]:
    """Current macro positions from recipes.MACRO_LIST output (the compaction
    input — read from OpenROAD, never trusted to the VLM)."""
    out: list[dict] = []
    seen: set[str] = set()
    for line in macros_text.splitlines():
        if "master=" not in line or "pos=" not in line:
            continue
        name = line.split("master=", 1)[0].strip()
        pm = re.search(r"pos=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", line)
        if name and pm and name not in seen:
            seen.add(name)
            om = re.search(r"orient=(\S+)", line)
            orient = om.group(1) if om and om.group(1) in VALID_ORIENTS else "R0"
            out.append({"name": name, "x": float(pm.group(1)),
                        "y": float(pm.group(2)), "orient": orient})
    return out


def parse_pin_boxes(pin_box_text: str) -> list[tuple[float, float, float, float]]:
    """Top-level IO pin boxes from recipes.PIN_COORDS output."""
    boxes: list[tuple[float, float, float, float]] = []
    for m in re.finditer(
            r"box=\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)",
            pin_box_text):
        boxes.append(tuple(float(m.group(i)) for i in range(1, 5)))
    return boxes


def parse_macro_index(macros_text: str) -> list[tuple[str, float, float]]:
    """Ordered [(name, w, h)] from recipes.MACRO_LIST output. The list index is
    the stable ``#i`` the Planner uses to place a macro WITHOUT retyping its
    (escaped-bracket) name — Python maps the index back to the exact name."""
    out: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for line in macros_text.splitlines():
        if "master=" not in line or "size=" not in line:
            continue
        name = line.split("master=", 1)[0].strip()
        sm = re.search(r"size=([\d.]+)\s*x\s*([\d.]+)", line)
        if name and sm and name not in seen:
            seen.add(name)
            out.append((name, float(sm.group(1)), float(sm.group(2))))
    return out


def parse_macro_cluster_file(cluster_file) -> list[dict]:
    """Parse an RTL-MP macro-cluster text file.

    Returns a list preserving file order:
    ``[{cluster, macros:[{name,type,width,height}, ...]}, ...]``.
    The placement code uses this as a dataflow proxy: macros in one listed
    cluster are assigned adjacent slots before moving to the next cluster.
    """
    path = Path(cluster_file) if cluster_file else None
    if not path or not path.exists():
        return []
    clusters: list[dict] = []
    current: dict | None = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        hm = re.match(r"^\[\s*\d+\]\s+(.+?)\s*$", line)
        if hm:
            current = {"cluster": hm.group(1), "macros": []}
            clusters.append(current)
            continue
        if current is None or not raw.startswith("  "):
            continue
        mm = re.match(
            r"^\s+(\S+)\s+type=(\S+)\s+size=([\d.]+)x([\d.]+)um\s*$",
            line)
        if mm:
            current["macros"].append({
                "name": mm.group(1),
                "type": mm.group(2),
                "width": float(mm.group(3)),
                "height": float(mm.group(4)),
            })
    return [c for c in clusters if c["macros"]]


def macro_cluster_sizes(cluster_file) -> dict[str, tuple[float, float]]:
    """name -> (w, h) from a macro-cluster file."""
    sizes: dict[str, tuple[float, float]] = {}
    for cluster in parse_macro_cluster_file(cluster_file):
        for m in cluster["macros"]:
            sizes[m["name"]] = (m["width"], m["height"])
    return sizes


def region_grid(core: tuple[float, float, float, float], n: int = 4) -> list[dict]:
    """Partition the core into an n*n grid of regions (the planner skill's
    method). Returns cells labelled ``r<row>c<col>`` with bounds + centre."""
    cx0, cy0, cx1, cy1 = core
    w = (cx1 - cx0) / n
    h = (cy1 - cy0) / n
    cells = []
    for row in range(n):           # row 0 = bottom
        for col in range(n):
            x0 = cx0 + col * w
            y0 = cy0 + row * h
            cells.append({
                "label": f"r{row}c{col}", "x0": round(x0, 1), "y0": round(y0, 1),
                "x1": round(x0 + w, 1), "y1": round(y0 + h, 1),
                "cx": round(x0 + w / 2, 1), "cy": round(y0 + h / 2, 1),
            })
    return cells


def pack_region(region: dict, n: int, mw: float, mh: float,
                gap: float = 2.0) -> list[tuple[float, float]]:
    """Pack ``n`` macros row-by-row inside a region's bounds. Used when the
    Placer assigns a per-region COUNT (the tractable interface for many /
    identical macros — the 8B cannot emit dozens of distinct coordinates, it
    repeats one). Overflow past the region runs upward and is clamped + spread
    by legalization afterwards."""
    x0, y0 = region["x0"], region["y0"]
    sx, sy = mw + gap, mh + gap
    cols = max(1, int((region["x1"] - x0) // sx))
    return [(x0 + (i % cols) * sx, y0 + (i // cols) * sy) for i in range(n)]


_ORIENT_RE = re.compile(r"\b(R0|R90|R180|R270|MXR90|MYR90|MX|MY)\b")
_SAFE_EXPR_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_numeric_expr(expr: str) -> float | None:
    """Evaluate a simple numeric coordinate expression from model output."""
    expr = expr.strip()
    if not expr:
        return None
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_EXPR_OPS:
            return _SAFE_EXPR_OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_EXPR_OPS:
            return _SAFE_EXPR_OPS[type(n.op)](walk(n.operand))
        raise ValueError

    try:
        return float(walk(node))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_xy_from_line(rest: str) -> tuple[float, float] | None:
    """Parse x,y from a plan line, accepting either decimals or simple formulas."""
    rest_no_orient = _ORIENT_RE.sub(" ", rest)
    pm = re.search(r"\(([^)]*)\)", rest_no_orient)
    if pm:
        parts = [p.strip() for p in pm.group(1).split(",")]
        if len(parts) >= 2:
            x = _eval_numeric_expr(parts[0])
            y = _eval_numeric_expr(parts[1])
            if x is not None and y is not None:
                return x, y
    nums = re.findall(r"-?\d+\.?\d*", rest_no_orient)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def parse_indexed_plan(text: str, n_macros: int) -> dict[int, tuple[float, float, str]]:
    """Parse a Planner placement plan into ``{macro_index: (x, y, orient)}``.

    Accepts lines like ``#3 -> (120.5, 200.0, R0)`` / ``3: 120 200 R90`` /
    ``#3 120,200``. The macro index is the first integer on the line; the next
    two numbers (after stripping any orientation token, so the digits in 'R90'
    aren't mistaken for coordinates) are x,y in microns. Tolerant of prose: any
    line without a parseable index+coords pair is skipped, so partial/odd model
    output still yields a usable (legalizable) placement.
    """
    out: dict[int, tuple[float, float, str]] = {}
    for line in text.splitlines():
        im = re.search(r"#?(\d+)", line)
        if not im:
            continue
        idx = int(im.group(1))
        if idx >= n_macros or idx in out:
            continue
        rest = line[im.end():]
        om = _ORIENT_RE.search(rest)
        orient = om.group(1) if om else "R0"
        xy = _parse_xy_from_line(rest)
        if xy is not None:
            out[idx] = (xy[0], xy[1], orient)
    return out


def macro_footprint(macro_sizes: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Largest macro width/height — the ring-slot pitch must fit any macro."""
    if macro_sizes:
        mw = max(w for w, _ in macro_sizes.values())
        mh = max(h for _, h in macro_sizes.values())
        return mw, mh
    return DEFAULT_MW, DEFAULT_MH


def oriented_size(w: float, h: float, orient: str) -> tuple[float, float]:
    """Macro bbox dimensions after OpenROAD orientation is applied."""
    if orient in {"R90", "R270", "MXR90", "MYR90"}:
        return h, w
    return w, h


def placement_stats(entries: list[dict],
                    macro_sizes: dict[str, tuple[float, float]],
                    core: tuple[float, float, float, float],
                    mw: float = DEFAULT_MW,
                    mh: float = DEFAULT_MH,
                    gap: float = 0.0,
                    outside_tol: float = 0.0) -> dict:
    """Cheap geometry score for a raw coordinate plan before legalization.

    This catches degenerate LLM output such as putting every macro at the same
    legal boundary coordinate. It is intentionally independent of OpenROAD so a
    bad tool call can be rejected and retried before macro placement.

    ``outside_tol`` is a slack (um) on the core-boundary test: a macro counts as
    outside only if it overshoots a core edge by MORE than this. Legalization
    clamps any macro into the core (``legalize._clamp_to_core``), so small
    overshoots — e.g. a macro flush to an edge, or a model that rounded the core
    origin 20.14→20.1 — are salvageable and should not reject the whole plan.
    ``0.0`` keeps the original strict behaviour.
    """
    cx0, cy0, cx1, cy1 = core
    boxes = []
    outside = 0
    origins = []
    xs = []
    ys = []
    for e in entries:
        w, h = macro_sizes.get(e["name"], (mw, mh))
        w, h = oriented_size(w, h, e.get("orient", "R0"))
        x = float(e["x"])
        y = float(e["y"])
        boxes.append((x, y, x + w, y + h))
        xs.append(x)
        ys.append(y)
        origins.append((round(x, 1), round(y, 1)))
        overshoot = max(cx0 - x, cy0 - y, (x + w) - cx1, (y + h) - cy1, 0.0)
        if overshoot > outside_tol:
            outside += 1

    overlaps = 0
    n = len(boxes)
    for i in range(n):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = boxes[j]
            if (ax0 < bx1 + gap and ax1 > bx0 - gap and
                    ay0 < by1 + gap and ay1 > by0 - gap):
                overlaps += 1

    return {
        "overlaps": overlaps,
        "outside_core": outside,
        "duplicate_origins": len(origins) - len(set(origins)),
        "unique_x": len({round(x, 1) for x in xs}),
        "unique_y": len({round(y, 1) for y in ys}),
        "span_x": (max(xs) - min(xs)) if xs else 0.0,
        "span_y": (max(ys) - min(ys)) if ys else 0.0,
    }


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def validate_entries(entries: list[dict], cx0: float, cy0: float, cx1: float,
                     cy1: float, mw: float = DEFAULT_MW,
                     mh: float = DEFAULT_MH,
                     macro_sizes: dict[str, tuple[float, float]] | None = None) -> list[dict]:
    """Clamp coordinates inside the core and fix invalid orientations."""
    for e in entries:
        if e.get("orient") not in VALID_ORIENTS:
            e["orient"] = "R0"
        w, h = (macro_sizes or {}).get(e["name"], (mw, mh))
        w, h = oriented_size(w, h, e.get("orient", "R0"))
        lo_x = cx0 + CORE_SNAP_MARGIN
        lo_y = cy0 + CORE_SNAP_MARGIN
        hi_x = cx1 - w - CORE_SNAP_MARGIN
        hi_y = cy1 - h - CORE_SNAP_MARGIN
        if hi_x < lo_x:
            lo_x, hi_x = cx0, cx1 - w
        if hi_y < lo_y:
            lo_y, hi_y = cy0, cy1 - h
        e["x"] = max(lo_x, min(e["x"], hi_x))
        e["y"] = max(lo_y, min(e["y"], hi_y))
    return entries


def perimeter_ring_slots(n_total: int, cx0: float, cy0: float, cx1: float,
                         cy1: float, mw: float, mh: float,
                         gap: float) -> list[tuple[float, float]]:
    """Packed perimeter ring of slot positions (bottom->right->top->left),
    overlap-free by construction so legalization is a near no-op. Adds inner
    rings if there are more macros than perimeter slots."""
    sx = mw + gap
    sy = mh + gap
    bottom = max(1, int((cx1 - cx0 - mw) // sx))
    top = bottom
    right = max(1, int((cy1 - cy0 - mh) // sy))
    left = right

    slots: list[tuple[float, float]] = []
    for i in range(bottom):
        slots.append((cx0 + i * sx, cy0))
    for j in range(right):
        slots.append((cx1 - mw, cy0 + j * sy))
    for i in range(top):
        slots.append((cx1 - mw - i * sx, cy1 - mh))
    for j in range(left):
        slots.append((cx0, cy1 - mh - j * sy))

    if not slots:
        return [(cx0, cy0)]
    ring = 1
    while len(slots) < n_total and ring < 8:
        inset = ring * (mh + gap) * 1.5
        ix0, iy0 = cx0 + inset, cy0 + inset
        ix1, iy1 = cx1 - inset, cy1 - inset
        if ix1 - ix0 < mw or iy1 - iy0 < mh:
            break
        b = max(1, int((ix1 - ix0 - mw) // sx))
        r = max(1, int((iy1 - iy0 - mh) // sy))
        for i in range(b):
            slots.append((ix0 + i * sx, iy0))
        for j in range(r):
            slots.append((ix1 - mw, iy0 + j * sy))
        for i in range(b):
            slots.append((ix1 - mw - i * sx, iy1 - mh))
        for j in range(r):
            slots.append((ix0, iy1 - mh - j * sy))
        ring += 1
    return slots


def _slot_legal(slot: tuple[float, float], placed: list[tuple[float, float]],
                mw: float, mh: float, gap: float) -> bool:
    x, y = slot
    for px, py in placed:
        if (x < px + mw + gap and x + mw > px - gap and
                y < py + mh + gap and y + mh > py - gap):
            return False
    return True


def strategy_sides(strategy: str) -> list[str]:
    """Infer direct side bands from a Planner strategy."""
    s = strategy.lower()
    side_order = ("right", "top", "bottom", "left")

    def add_side(out: list[str], side: str) -> None:
        if side in side_order and side not in out:
            out.append(side)

    def sides_in_phrase(phrase: str) -> list[str]:
        found: list[str] = []
        for side in side_order:
            if re.search(rf"\b{side}(?:[- ](?:edge|side|band|boundary))?\b",
                         phrase):
                add_side(found, side)
        return found

    # Sides to KEEP CLEAR of macros — both "reserve/leave/keep" intent and
    # negated/avoidance phrasing ("avoid the left side", "away from the right",
    # "no macros on top", "C-shape open on the left"). Without the avoidance
    # cases a strategy like "C-shaped periphery, avoiding the left side (IO
    # pins)" was parsed as "place macros ON the left side" — i.e. straight onto
    # the pin side it was told to keep clear.
    reserved: set[str] = set()
    reserve_verbs = (r"leav(?:e|ing)|keep|clear|reserve(?:d)?|avoid(?:ing)?|"
                     r"away\s+from|open(?:ing)?(?:\s+(?:on|to(?:ward)?s?|at))?|"
                     r"no\s+macros?\s+(?:on|along|near|at)|not\s+(?:on|along|near|at)|"
                     r"free\s+of")
    for m in re.finditer(
            rf"\b(?:{reserve_verbs})\b"
            r"[^.:\n;]*\b(right|top|bottom|left)\b",
            s):
        phrase = m.group(0)
        for side in ("right", "top", "bottom", "left"):
            if side in phrase:
                reserved.add(side)
    placement_clause = re.split(
        rf"\b(?:{reserve_verbs})\b",
        s, maxsplit=1)[0]
    sides: list[str] = []
    placement_patterns = [
        r"\b(?:along|on|at|near|against)\s+(?:the\s+)?"
        r"((?:(?:right|top|bottom|left)(?:[- ](?:edge|side|band|boundary))?\b[\s,]*(?:and\s+)?)*)"
        r"\s*(?:edge|edges|side|sides|band|bands|boundary|boundaries|periphery)\b",
        r"\b((?:(?:right|top|bottom|left)(?:[- ](?:edge|side|band|boundary))?\b[\s,]*(?:and\s+)?)*)"
        r"\s*(?:edge|edges|side|sides|band|bands|boundary|boundaries)\b",
    ]
    for pat in placement_patterns:
        for m in re.finditer(pat, placement_clause):
            for side in sides_in_phrase(m.group(1)):
                add_side(sides, side)
    # Never place on a side we were told to keep clear.
    sides = [side for side in sides if side not in reserved]
    # Peripheral / ring / C-shape intent is authoritative: go around the whole
    # die except the reserved (pin / opening) sides.
    peripheral = any(k in s for k in ("periphery", "peripheral", "outline",
                                      "ring", "c-shape", "c shape", "c-shaped"))
    if peripheral or not sides:
        sides = [side for side in side_order if side not in reserved]
    if not sides:
        sides = ["right", "top", "bottom", "left"]
    return sides


def side_band_slots(n_total: int, core: tuple[float, float, float, float],
                    mw: float, mh: float, gap: float,
                    sides: list[str]) -> list[tuple[float, float]]:
    """Generate direct legal coordinates along requested side bands.

    This is not a region grid: it expands a side/top/C-shape strategy into
    concrete lower-left macro origins, adding inner bands only when one band is
    full.
    """
    cx0, cy0, cx1, cy1 = core
    lo_x = cx0 + CORE_SNAP_MARGIN
    lo_y = cy0 + CORE_SNAP_MARGIN
    hi_x = cx1 - mw - CORE_SNAP_MARGIN
    hi_y = cy1 - mh - CORE_SNAP_MARGIN
    sx = mw + gap
    sy = mh + gap
    slots: list[tuple[float, float]] = []
    max_lanes = max(1, int(min(cx1 - cx0, cy1 - cy0) // max(1.0, min(sx, sy))))

    def add(x: float, y: float) -> None:
        if len(slots) >= n_total:
            return
        if x < lo_x - 1e-6 or x > hi_x + 1e-6:
            return
        if y < lo_y - 1e-6 or y > hi_y + 1e-6:
            return
        slot = (round(x, 2), round(y, 2))
        if _slot_legal(slot, slots, mw, mh, gap):
            slots.append(slot)

    for lane in range(max_lanes):
        for side in sides:
            if side == "right":
                x = hi_x - lane * sx
                y = hi_y
                while y >= lo_y - 1e-6:
                    add(x, y)
                    y -= sy
            elif side == "left":
                x = lo_x + lane * sx
                y = hi_y
                while y >= lo_y - 1e-6:
                    add(x, y)
                    y -= sy
            elif side == "top":
                y = hi_y - lane * sy
                x = hi_x
                while x >= lo_x - 1e-6:
                    add(x, y)
                    x -= sx
            elif side == "bottom":
                y = lo_y + lane * sy
                x = hi_x
                while x >= lo_x - 1e-6:
                    add(x, y)
                    x -= sx
        if len(slots) >= n_total:
            break

    if len(slots) < n_total:
        for slot in perimeter_ring_slots(n_total, cx0, cy0, cx1, cy1, mw, mh, gap):
            slot = (round(max(lo_x, min(slot[0], hi_x)), 2),
                    round(max(lo_y, min(slot[1], hi_y)), 2))
            if _slot_legal(slot, slots, mw, mh, gap):
                slots.append(slot)
            if len(slots) >= n_total:
                break
    return slots or [(lo_x, lo_y)]


def compact_entries(entries: list[dict], strength: float, anchor: str,
                    cx0: float, cy0: float, cx1: float, cy1: float) -> list[dict]:
    """Pull macros tighter and toward an anchor.

    1. Contract every macro toward the cluster centroid by ``strength``
       (shrinks the spread while preserving the arrangement).
    2. Translate the whole cluster a fraction toward the anchor point so the
       freed space is contiguous on the opposite side.

    The model only chooses ``strength`` (0..0.6) and ``anchor``; the math here.
    """
    if not entries:
        return entries
    strength = max(0.0, min(strength, 0.6))

    cen_x = sum(e["x"] for e in entries) / len(entries)
    cen_y = sum(e["y"] for e in entries) / len(entries)

    fx, fy = COMPACT_ANCHORS.get(anchor, (0.50, 0.50))
    tgt_x = cx0 + fx * (cx1 - cx0)
    tgt_y = cy0 + fy * (cy1 - cy0)

    shift_x = (tgt_x - cen_x) * strength * 0.5
    shift_y = (tgt_y - cen_y) * strength * 0.5

    for e in entries:
        e["x"] = cen_x + (e["x"] - cen_x) * (1.0 - strength) + shift_x
        e["y"] = cen_y + (e["y"] - cen_y) * (1.0 - strength) + shift_y
    return entries


def legalize(entries: list[dict], macro_sizes: dict[str, tuple[float, float]],
             cx0: float, cy0: float, cx1: float, cy1: float,
             mw: float = DEFAULT_MW, mh: float = DEFAULT_MH,
             gap: float = 2.0) -> list[dict]:
    """Resolve overlaps (force-directed + greedy-snap fallback) in place."""
    return legalize_plan_entries(entries, macro_sizes, cx0, cy0, cx1, cy1,
                                 default_w=mw, default_h=mh, gap=gap)


# ---------------------------------------------------------------------------
# Ring seed (deterministic initial placement)
# ---------------------------------------------------------------------------
def _group_macros(macros_text: str, cluster_file=None) -> dict[str, list[str]]:
    """Group macros by RTL-MP cluster file, falling back to SRAM-block prefix.

    Grouping only affects ring-slot *ordering* (clustering related macros on
    the ring); names are split on '  master=' to keep escaped bus brackets.
    """
    macro_names: list[str] = []
    seen_names: set[str] = set()
    for line in macros_text.splitlines():
        if "master=" not in line:
            continue
        name = line.split("master=", 1)[0].strip()
        if name and name not in seen_names:
            seen_names.add(name)
            macro_names.append(name)

    if cluster_file:
        clusters = parse_macro_cluster_file(cluster_file)
        if clusters:
            remaining = set(macro_names)
            groups: dict[str, list[str]] = {}
            for cluster in clusters:
                names = [m["name"] for m in cluster["macros"]
                         if m["name"] in remaining]
                if names:
                    groups[cluster["cluster"]] = names
                    remaining.difference_update(names)
            if groups:
                for name in macro_names:
                    if name in remaining:
                        groups.setdefault("unclustered", []).append(name)
                return groups

    groups: dict[str, list[str]] = {}
    for name in macro_names:
        gm = re.match(r"(.+?/sram_block_\d+__\w+)", name)
        gkey = gm.group(1) if gm else "other"
        groups.setdefault(gkey, []).append(name)
    return groups


def build_ring_seed(macros_text: str, dims_text: str, gap: float = 5.0,
                    legalize_seed: bool = True, cluster_file=None):
    """Deterministic, overlap-free initial placement: assign every macro to a
    packed perimeter-ring slot, validate to the core, then legalize.

    Returns ``(entries, core, macro_sizes, mw, mh)`` where ``core`` is
    ``(cx0, cy0, cx1, cy1)``. Mirrors the validated classic seed but with the
    advisory VLM call removed — the Planner agent supplies strategy separately.
    """
    core = parse_core_bounds(dims_text)
    cx0, cy0, cx1, cy1 = core
    macro_sizes = parse_macro_sizes(macros_text)
    macro_sizes.update(macro_cluster_sizes(cluster_file))
    mw, mh = macro_footprint(macro_sizes)

    groups = _group_macros(macros_text, cluster_file=cluster_file)
    total = sum(len(v) for v in groups.values())
    slots = perimeter_ring_slots(total, cx0, cy0, cx1, cy1, mw, mh, gap)

    entries: list[dict] = []
    cursor = 0
    for cluster_order, (gkey, names) in enumerate(groups.items()):
        n = len(names)
        seg = slots[cursor:cursor + n]
        while len(seg) < n:
            seg.append(slots[(cursor + len(seg)) % len(slots)])
        for name, (sx, sy) in zip(names, seg):
            entries.append({"name": name, "x": round(sx, 1),
                            "y": round(sy, 1), "orient": "R0",
                            "cluster": gkey, "cluster_order": cluster_order})
        cursor += n

    entries = validate_entries(entries, cx0, cy0, cx1, cy1, mw=mw, mh=mh,
                               macro_sizes=macro_sizes)
    if legalize_seed:
        entries = legalize(entries, macro_sizes, cx0, cy0, cx1, cy1,
                           mw=mw, mh=mh, gap=gap)
    return entries, core, macro_sizes, mw, mh


def build_strategy_seed(macros_text: str, dims_text: str, strategy: str,
                        gap: float = 5.0, legalize_seed: bool = True,
                        cluster_file=None):
    """Direct coordinate seed from a Planner side/C-shape strategy."""
    core = parse_core_bounds(dims_text)
    cx0, cy0, cx1, cy1 = core
    macro_sizes = parse_macro_sizes(macros_text)
    macro_sizes.update(macro_cluster_sizes(cluster_file))
    mw, mh = macro_footprint(macro_sizes)

    groups = _group_macros(macros_text, cluster_file=cluster_file)
    total = sum(len(v) for v in groups.values())
    sides = strategy_sides(strategy)
    slots = side_band_slots(total, core, mw, mh, gap, sides)

    entries: list[dict] = []
    cursor = 0
    for cluster_order, (gkey, names) in enumerate(groups.items()):
        n = len(names)
        seg = slots[cursor:cursor + n]
        while len(seg) < n:
            seg.append(slots[(cursor + len(seg)) % len(slots)])
        for name, (sx, sy) in zip(names, seg):
            entries.append({"name": name, "x": round(sx, 2),
                            "y": round(sy, 2), "orient": "R0",
                            "cluster": gkey, "cluster_order": cluster_order})
        cursor += n

    entries = validate_entries(entries, cx0, cy0, cx1, cy1, mw=mw, mh=mh,
                               macro_sizes=macro_sizes)
    if legalize_seed:
        entries = legalize(entries, macro_sizes, cx0, cy0, cx1, cy1,
                           mw=mw, mh=mh, gap=gap)
    return entries, core, macro_sizes, mw, mh, sides


def assign_areas(entries: list[dict],
                 macro_sizes: dict[str, tuple[float, float]],
                 mw: float = DEFAULT_MW, mh: float = DEFAULT_MH) -> None:
    """Tag each entry with its footprint area so the placer can sort
    largest-first (big macros are hardest to fit — place them first)."""
    for e in entries:
        w, h = macro_sizes.get(e["name"], (mw, mh))
        e["area"] = w * h


# ---------------------------------------------------------------------------
# Exploration helpers — coarse SHAPE presets, fine-grain deltas, channel snap
# ---------------------------------------------------------------------------
def _grid_labels(rows: int = 4, cols: int = 4) -> list[str]:
    return [f"r{r}c{c}" for r in range(rows) for c in range(cols)]


def _dedup(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _shape_labels(name: str, rows: int = 4, cols: int = 4) -> list[str]:
    """Active region labels for a coarse outline on a rows*cols grid (row 0 =
    bottom, col 0 = left). Each preset pushes macros to a distinct band so the
    freed stdcell area has a genuinely different shape."""
    top = rows - 1
    rgt = cols - 1
    col0 = [f"r{r}c0" for r in range(rows)]
    colN = [f"r{r}c{rgt}" for r in range(rows)]
    row0 = [f"r0c{c}" for c in range(cols)]
    rowN = [f"r{top}c{c}" for c in range(cols)]
    perim = [f"r{r}c{c}" for r in range(rows) for c in range(cols)
             if r in (0, top) or c in (0, rgt)]
    presets = {
        "ring":     perim,                          # open square centre
        "c_left":   col0 + row0 + rowN,             # C opening to the right
        "c_right":  colN + row0 + rowN,             # C opening to the left
        "c_bottom": row0 + col0 + colN,             # C opening upward
        "c_top":    rowN + col0 + colN,             # C opening downward
        "corner":   ["r0c0", "r0c1", "r1c0", "r1c1"],  # bottom-left cluster
        "split":    col0 + colN,                    # two columns, centre channel
    }
    return _dedup(presets.get(name, perim))


# Order the coarse loop cycles through (genuinely different outlines first).
COARSE_SHAPE_ORDER = ["ring", "c_left", "c_right", "c_bottom", "c_top",
                      "corner", "split"]


def shape_distribution(name: str, n: int, regions: list[dict]) -> dict[str, int]:
    """Spread ``n`` macros as evenly as possible over a preset shape's active
    regions, returning a ``{region_label: count}`` map summing to ``n`` — the
    input ``place_in_regions`` expects. Regions not present in ``regions`` are
    dropped (small/degenerate grids)."""
    available = {c["label"] for c in regions}
    labels = [l for l in _shape_labels(name) if l in available]
    if not labels or n <= 0:
        return {}
    k = len(labels)
    base, rem = divmod(n, k)
    dist: dict[str, int] = {}
    for i, label in enumerate(labels):
        cnt = base + (1 if i < rem else 0)
        if cnt > 0:
            dist[label] = cnt
    return dist


_DELTA_LINE = re.compile(r"#?(\d+)")


def parse_indexed_deltas(text: str, n_macros: int) -> dict[int, tuple[float, float, str | None]]:
    """Parse fine-grain moves ``#index -> (dx, dy[, orient])`` into
    ``{index: (dx, dy, orient_or_None)}``. dx/dy are signed micron *offsets*
    (+x right, +y up). Tolerant of prose: lines without an index + two numbers
    are skipped, so partial model output still applies."""
    out: dict[int, tuple[float, float, str | None]] = {}
    for line in text.splitlines():
        im = _DELTA_LINE.search(line)
        if not im:
            continue
        idx = int(im.group(1))
        if idx >= n_macros or idx in out:
            continue
        rest = line[im.end():]
        om = _ORIENT_RE.search(rest)
        orient = om.group(1) if om else None
        rest_no_orient = _ORIENT_RE.sub(" ", rest)
        nums = re.findall(r"-?\d+\.?\d*", rest_no_orient)
        if len(nums) >= 2:
            out[idx] = (float(nums[0]), float(nums[1]), orient)
    return out


def snap_to_edges(entries: list[dict], core: tuple[float, float, float, float],
                  macro_sizes: dict[str, tuple[float, float]],
                  min_channel: float = 5.0, mw: float = DEFAULT_MW,
                  mh: float = DEFAULT_MH) -> list[dict]:
    """Deterministic fine fallback: any macro that leaves a *thin* (0 < gap <
    ``min_channel``) sliver against a core edge is snapped flush to that edge,
    eliminating the unroutable channel. Mirrors the user's "move macro to
    eliminate the small channel between macro and side"."""
    cx0, cy0, cx1, cy1 = core
    for e in entries:
        w, h = macro_sizes.get(e["name"], (mw, mh))
        x, y = e["x"], e["y"]
        gl, gr = x - cx0, cx1 - (x + w)
        gb, gt = y - cy0, cy1 - (y + h)
        if 0 < gl < min_channel:
            x = cx0
        elif 0 < gr < min_channel:
            x = cx1 - w
        if 0 < gb < min_channel:
            y = cy0
        elif 0 < gt < min_channel:
            y = cy1 - h
        e["x"], e["y"] = x, y
    return entries
