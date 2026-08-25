"""Metric parsing from OpenROAD console output + placement ranking.

Honest parsing matters: the detailed router prints a per-iteration violation
count, so a naive ``re.search`` grabs an early "0 violations" line — we take the
*last* one (and the orchestrator further overrides DRC from the report file).
"""

from __future__ import annotations

import re


_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_metrics(output: str) -> dict:
    """Extract routed timing, area, wirelength, DRC and power from OpenROAD output."""
    metrics: dict = {}
    # Take the LAST occurrence, not the first. The timing route reports
    # tns/wns/area several times (e.g. a CTS-stage report BEFORE global route +
    # timing repair, then the final post-route report). re.search would grab the
    # early, pre-repair value (e.g. tns -382 instead of the routed -39),
    # corrupting the recorded PPA and the optimizer's candidate ranking.
    for key, pattern in [
        ("wns", rf"wns\s+(?:max\s+)?({_NUM})"),
        ("tns", rf"tns\s+(?:max\s+)?({_NUM})"),
        ("area", rf"Design area\s+({_NUM})"),
        ("wirelength", rf"(?:Total wire length|wire length)\s*=?\s*({_NUM})"),
    ]:
        found = re.findall(pattern, output, re.IGNORECASE)
        if found:
            metrics[key] = float(found[-1])
    util_found = re.findall(
        rf"Design area\s+{_NUM}\s+\S+\^?2?\s+({_NUM})%\s+utilization",
        output, re.IGNORECASE)
    if util_found:
        metrics["utilization"] = float(util_found[-1])
    power_m = re.search(
        rf"^Total\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+100\.0%",
        output, re.IGNORECASE | re.MULTILINE)
    if power_m:
        internal, switching, leakage, total = (float(power_m.group(i))
                                              for i in range(1, 5))
        metrics.update({
            "power": total,
            "power_total": total,
            "power_internal": internal,
            "power_switching": switching,
            "power_leakage": leakage,
        })
    drt_counts = re.findall(r"Number of violations\s*=\s*(\d+)", output)
    if drt_counts:
        metrics["drc"] = int(drt_counts[-1])
    else:
        drc_m = re.search(r"(\d+)\s+violation", output, re.IGNORECASE)
        if drc_m:
            metrics["drc"] = int(drc_m.group(1))
    return metrics


def parse_density(text: str) -> float | None:
    """STDCELL_DENSITY value from recipes.STDCELL_DENSITY output."""
    m = re.search(r"STDCELL_DENSITY:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def parse_overflow(text: str) -> int | None:
    """Global-route total overflow from the congestion proxy output."""
    m = re.search(r"(?:Total\s+)?[Oo]verflow[^\d]*(\d+)", text)
    return int(m.group(1)) if m else None


def parse_overlaps(text: str) -> int | None:
    """Overlapping macro-pair count from recipes.CHECK_OVERLAPS output."""
    lines = [l for l in text.splitlines() if "Total overlaps" in l]
    if not lines:
        return None
    m = re.search(r"Total overlaps:\s*(\d+)", lines[-1])
    return int(m.group(1)) if m else None


def parse_outside_core(text: str) -> int | None:
    m = re.search(r"Total outside core:\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def is_better_proxy(current: dict, best: dict) -> bool:
    """Rank placements by the fast PPA proxy (no full route yet): lower routing
    overflow, then lower estimated wirelength, then higher stdcell density."""
    co = current.get("overflow", float("inf"))
    bo = best.get("overflow", float("inf"))
    if co != bo:
        return co < bo
    cw = current.get("wirelength", float("inf"))
    bw = best.get("wirelength", float("inf"))
    if cw != bw:
        return cw < bw
    return current.get("density", 0.0) > best.get("density", 0.0)
