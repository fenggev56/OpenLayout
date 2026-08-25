"""Metric extraction — parse OpenROAD logs and report files into a flat dict
(the PPA / wirelength / DRC numbers the agents and the final report use).

All parsers are tolerant: they return whatever they can find and never raise
on malformed input (a stage crash still yields a partial metrics dict plus
the error lines for the Debugger agent).
"""

from __future__ import annotations

import re
from pathlib import Path

_NUM = r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"


def parse_log_metrics(text: str) -> dict:
    m: dict = {}

    # timing --------------------------------------------------------------
    # report_worst_slack -max / -min  ->  "worst slack max -0.10"
    for kind, key in (("max", "wns_setup"), ("min", "wns_hold")):
        hits = re.findall(rf"worst slack {kind}\s+{_NUM}", text)
        if hits:
            m[key] = float(hits[-1])
    hits = re.findall(rf"^wns {_NUM}", text, re.MULTILINE)
    if hits and "wns_setup" not in m:
        m["wns_setup"] = float(hits[-1])
    hits = re.findall(rf"^tns {_NUM}", text, re.MULTILINE)
    if hits:
        m["tns"] = float(hits[-1])
    hits = re.findall(rf"tns max\s+{_NUM}", text)
    if hits:
        m["tns"] = float(hits[-1])

    # clock skew ----------------------------------------------------------
    hits = re.findall(rf"[Ss]kew:?\s+{_NUM}", text)
    if hits:
        m["clock_skew"] = float(hits[-1])

    # area / utilization --------------------------------------------------
    hits = re.findall(rf"Design area {_NUM} um?\^2 {_NUM}% utilization", text)
    if hits:
        m["design_area_um2"] = float(hits[-1][0])
        m["utilization_pct"] = float(hits[-1][1])

    # wirelength ----------------------------------------------------------
    for pat in (rf"[Tt]otal wire ?length[:=\s]+{_NUM}\s*um",
                rf"Total wirelength[:=\s]+{_NUM}"):
        hits = re.findall(pat, text)
        if hits:
            m["wirelength_um"] = float(hits[-1])
            break

    # global route congestion --------------------------------------------
    hits = re.findall(rf"[Tt]otal [Oo]verflow\s*:?\s*{_NUM}", text)
    if hits:
        m["grt_overflow"] = float(hits[-1])

    # detailed route DRC convergence -------------------------------------
    hits = re.findall(r"[Nn]umber of violations\s*=\s*(\d+)", text)
    if hits:
        m["drc_violations"] = int(hits[-1])

    # power (report_power total row) -------------------------------------
    hits = re.findall(
        rf"^Total\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}", text, re.MULTILINE)
    if hits:
        i, s, l, t = (float(x) for x in hits[-1])
        m.update({"power_internal_w": i, "power_switching_w": s,
                  "power_leakage_w": l, "power_total_w": t})

    # placement -----------------------------------------------------------
    hits = re.findall(r"HPWL:\s*(\d+)", text)
    if hits:
        m["hpwl"] = int(hits[-1])
    hits = re.findall(r"(\d+) instances .*not placed|unplaced instances?:?\s*(\d+)",
                      text)
    # check_placement violations
    hits = re.findall(r"Placement violations?\s*:?\s*(\d+)", text)
    if hits:
        m["placement_violations"] = int(hits[-1])

    # IR drop -------------------------------------------------------------
    for net in ("VDD", "VSS"):
        hits = re.findall(
            rf"Worstcase voltage.*?{_NUM}\s*V?.*?net {net}|"
            rf"net {net}.*?[Ww]orst.?case voltage\s*:?\s*{_NUM}", text)
    hits = re.findall(rf"[Ww]orst.?case voltage\s*:?\s*{_NUM}", text)
    if hits:
        m["ir_worst_voltage_v"] = float(hits[0])
    hits = re.findall(rf"[Aa]verage IR drop\s*:?\s*{_NUM}", text)
    if hits:
        m["ir_avg_drop_v"] = float(hits[0])
    hits = re.findall(rf"[Ww]orst.?case IR drop\s*:?\s*{_NUM}", text)
    if hits:
        m["ir_worst_drop_v"] = float(hits[0])

    return m


def count_drc_violations(drc_rpt: Path) -> int | None:
    try:
        text = Path(drc_rpt).read_text(errors="replace")
    except OSError:
        return None
    return len(re.findall(r"violation type:", text))


def extract_errors(text: str, limit: int = 40) -> list[str]:
    """Unique [ERROR ...] lines + fatal patterns, most useful for debugging."""
    out: list[str] = []
    seen = set()
    for line in text.splitlines():
        if re.search(r"\[ERROR [A-Z]+-\d+\]|^Error:|FATAL|segmentation fault",
                     line, re.IGNORECASE):
            key = line.strip()[:120]
            if key not in seen:
                seen.add(key)
                out.append(line.strip()[:300])
        if len(out) >= limit:
            break
    return out


def extract_warning_summary(text: str, limit: int = 15) -> list[str]:
    """Count warnings by code, return the most frequent ones."""
    counts: dict[str, int] = {}
    sample: dict[str, str] = {}
    for line in text.splitlines():
        mm = re.search(r"\[WARNING ([A-Z]+-\d+)\]", line)
        if mm:
            code = mm.group(1)
            counts[code] = counts.get(code, 0) + 1
            sample.setdefault(code, line.strip()[:200])
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return [f"{n}x {code}: {sample[code]}" for code, n in ranked]


def tail(text: str, n_lines: int = 60) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])
