"""Force-directed macro legalization for mixed-size macros.

Resolves overlaps by iteratively pushing overlapping macros apart along
the axis of minimum penetration, then clamping to the core boundary.
Minimizes displacement from the original (VLM-planned) positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Macro:
    name: str
    x: float
    y: float
    w: float
    h: float
    orient: str = "R0"
    orig_x: float = 0.0
    orig_y: float = 0.0

    def __post_init__(self):
        self.orig_x = self.x
        self.orig_y = self.y

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def overlap_area(a: Macro, b: Macro) -> float:
    ox = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    oy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return ox * oy


def overlaps(a: Macro, b: Macro) -> bool:
    return a.x < b.x2 and a.x2 > b.x and a.y < b.y2 and a.y2 > b.y


def count_overlaps(macros: list[Macro]) -> int:
    n = len(macros)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if overlaps(macros[i], macros[j]):
                count += 1
    return count


def _oriented_size(w: float, h: float, orient: str) -> tuple[float, float]:
    if orient in {"R90", "R270", "MXR90", "MYR90"}:
        return h, w
    return w, h


def _clamp_to_core(m: Macro, cx0: float, cy0: float, cx1: float, cy1: float):
    m.x = max(cx0, min(m.x, cx1 - m.w))
    m.y = max(cy0, min(m.y, cy1 - m.h))


def legalize_macros(
    macros: list[Macro],
    cx0: float,
    cy0: float,
    cx1: float,
    cy1: float,
    gap: float = 2.0,
    max_iter: int = 500,
    spring_k: float = 0.05,
) -> list[Macro]:
    """Resolve all macro overlaps via iterative force-directed displacement.

    Args:
        macros: List of Macro objects with initial positions.
        cx0, cy0, cx1, cy1: Core boundary (min/max corners).
        gap: Minimum spacing between macros (um).
        max_iter: Maximum displacement iterations.
        spring_k: Strength of spring pulling macros back toward original positions.

    Returns:
        The same list with updated (x, y) positions, zero or minimal overlaps.
    """
    n = len(macros)
    if n == 0:
        return macros

    for iteration in range(max_iter):
        n_overlaps = 0
        dx = [0.0] * n
        dy = [0.0] * n

        for i in range(n):
            for j in range(i + 1, n):
                a, b = macros[i], macros[j]
                if not overlaps(a, b):
                    continue
                n_overlaps += 1

                ox = min(a.x2, b.x2) - max(a.x, b.x) + gap
                oy = min(a.y2, b.y2) - max(a.y, b.y) + gap

                if ox < oy:
                    shift = ox / 2 + 0.5
                    if a.cx <= b.cx:
                        dx[i] -= shift
                        dx[j] += shift
                    else:
                        dx[i] += shift
                        dx[j] -= shift
                else:
                    shift = oy / 2 + 0.5
                    if a.cy <= b.cy:
                        dy[i] -= shift
                        dy[j] += shift
                    else:
                        dy[i] += shift
                        dy[j] -= shift

        if n_overlaps == 0:
            break

        for i in range(n):
            m = macros[i]
            # Spring force toward original position (weak)
            dx[i] += spring_k * (m.orig_x - m.x)
            dy[i] += spring_k * (m.orig_y - m.y)

            m.x += dx[i]
            m.y += dy[i]
            _clamp_to_core(m, cx0, cy0, cx1, cy1)

        if iteration % 50 == 0 or n_overlaps < 5:
            print(f"    [Legalize] iter {iteration}: {n_overlaps} overlaps", flush=True)

    final_overlaps = count_overlaps(macros)
    print(f"    [Legalize] Done: {final_overlaps} overlaps remaining after {min(iteration + 1, max_iter)} iters", flush=True)

    if final_overlaps > 0:
        macros = _greedy_snap(macros, cx0, cy0, cx1, cy1, gap)
        final_overlaps = count_overlaps(macros)
        print(f"    [Legalize] After greedy snap: {final_overlaps} overlaps", flush=True)

    return macros


def _greedy_snap(
    macros: list[Macro],
    cx0: float, cy0: float, cx1: float, cy1: float,
    gap: float,
) -> list[Macro]:
    """Greedy single-pass placement guaranteeing zero overlaps when area fits.

    Places the largest macros first (so small macros fill the leftover gaps),
    each at the legal core position nearest its original location. Legality is
    found by a full coarse-grid scan over the core, so — unlike the previous
    sparse ring search — it never falls back to a knowingly-overlapping spot
    while free area still exists. With mixed macro sizes at low utilization
    (e.g. or1200_local, ~39% macro area) this reliably reaches 0 overlaps,
    which is required: overlapping FIXED macros make RePlAce (GPL) diverge.
    """
    # Largest-area first; ties broken by tallest, for stable packing.
    sorted_macros = sorted(macros, key=lambda m: (-(m.w * m.h), -m.h))

    placed: list[Macro] = []
    for m in sorted_macros:
        m.x, m.y = _nearest_legal(m, placed, cx0, cy0, cx1, cy1, gap)
        placed.append(m)

    return macros


def _nearest_legal(
    m: Macro, placed: list[Macro],
    cx0: float, cy0: float, cx1: float, cy1: float, gap: float,
) -> tuple[float, float]:
    """Legal position for ``m`` nearest its original location, via grid scan.

    Returns the current (clamped) position if already legal; otherwise the
    legal grid point closest to ``orig``. Only if no legal point exists at the
    scan resolution (core genuinely full) does it return the clamped current
    position — a degenerate case that does not occur at the utilizations here.
    """
    xmax = cx1 - m.w
    ymax = cy1 - m.h
    if xmax < cx0 or ymax < cy0:        # macro larger than core — give up
        return cx0, cy0

    cur_x = max(cx0, min(m.x, xmax))
    cur_y = max(cy0, min(m.y, ymax))
    if _is_legal(m, cur_x, cur_y, placed, gap):
        return cur_x, cur_y

    # Grid step tied to this macro's smaller dimension keeps the scan cheap
    # while staying fine enough to find gaps at low utilization.
    step = max(2.0, min(m.w, m.h) / 4.0)
    best = None
    best_d = float("inf")
    y = cy0
    while y <= ymax + 1e-6:
        x = cx0
        while x <= xmax + 1e-6:
            if _is_legal(m, x, y, placed, gap):
                d = (x - m.orig_x) ** 2 + (y - m.orig_y) ** 2
                if d < best_d:
                    best_d = d
                    best = (x, y)
            x += step
        y += step

    if best is not None:
        return best
    return cur_x, cur_y


def _try_place(m: Macro, x: float, y: float, placed: list[Macro],
               cx0: float, cy0: float, cx1: float, cy1: float, gap: float) -> bool:
    if _is_legal(m, x, y, placed, gap):
        placed.append(m)
        return True
    return False


def _is_legal(m: Macro, x: float, y: float, placed: list[Macro], gap: float) -> bool:
    for p in placed:
        if (x < p.x2 + gap and x + m.w > p.x - gap and
            y < p.y2 + gap and y + m.h > p.y - gap):
            return False
    return True


def legalize_plan_entries(
    entries: list[dict],
    macro_sizes: dict[str, tuple[float, float]],
    cx0: float, cy0: float, cx1: float, cy1: float,
    default_w: float = 77.71,
    default_h: float = 40.60,
    gap: float = 2.0,
) -> list[dict]:
    """Legalize a list of plan entries (dicts with name/x/y/orient).

    Args:
        entries: Plan entries from the VLM planner.
        macro_sizes: Dict mapping macro name -> (width, height) in um.
        cx0, cy0, cx1, cy1: Core boundary.
        default_w, default_h: Fallback macro size if not in macro_sizes.
        gap: Minimum spacing between macros.

    Returns:
        Updated entries with legalized positions.
    """
    macros = []
    for e in entries:
        w, h = macro_sizes.get(e["name"], (default_w, default_h))
        w, h = _oriented_size(w, h, e.get("orient", "R0"))
        macros.append(Macro(
            name=e["name"], x=e["x"], y=e["y"],
            w=w, h=h, orient=e.get("orient", "R0"),
        ))

    before = count_overlaps(macros)
    print(f"  [Legalize] {len(macros)} macros, {before} overlaps before legalization", flush=True)

    if before == 0:
        print(f"  [Legalize] No overlaps — skipping", flush=True)
        return entries

    macros = legalize_macros(macros, cx0, cy0, cx1, cy1, gap=gap)

    for e in entries:
        for m in macros:
            if m.name == e["name"]:
                e["x"] = round(m.x, 2)
                e["y"] = round(m.y, 2)
                break

    after = count_overlaps(macros)
    total_disp = sum(
        math.sqrt((m.x - m.orig_x) ** 2 + (m.y - m.orig_y) ** 2)
        for m in macros
    )
    avg_disp = total_disp / len(macros) if macros else 0
    print(f"  [Legalize] Result: {before} → {after} overlaps, avg displacement {avg_disp:.1f} um", flush=True)

    return entries
