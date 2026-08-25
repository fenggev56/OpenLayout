"""Tool queries — a whitelisted set of read-only OpenROAD inspections the
Debugger/Optimizer agents may request when the log evidence is not enough.

The agent puts ``"queries": ["setup_paths", ...]`` in its JSON answer; the
system runs them on the stage's checkpoint in a fresh batch process and asks
the agent once more with the results attached.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import runner

# name -> (description for the prompt, tcl body)
TCL_QUERIES: dict[str, tuple[str, str]] = {
    "worst_slacks": ("setup/hold worst slack + TNS",
                     "report_worst_slack -max\nreport_worst_slack -min\n"
                     "report_tns"),
    "setup_paths": ("5 worst setup paths",
                    "report_checks -path_delay max -group_count 5"),
    "hold_paths": ("5 worst hold paths",
                   "report_checks -path_delay min -group_count 5"),
    "clock_skew": ("clock skew report", "report_clock_skew"),
    "design_area": ("area / utilization", "report_design_area"),
    "cell_usage": ("cell counts by type", "report_cell_usage"),
    "check_placement": ("placement legality check",
                        "catch { check_placement -verbose }"),
    "power": ("power report", "report_power"),
}

# python-side queries (no OpenROAD launch)
FILE_QUERIES = {
    "drc_summary": "DRC violation counts by type (from the stage DRC report)",
}

ALL_QUERY_NAMES = list(TCL_QUERIES) + list(FILE_QUERIES)


def queries_help() -> str:
    lines = [f"- {n}: {d}" for n, (d, _) in TCL_QUERIES.items()]
    lines += [f"- {n}: {d}" for n, d in FILE_QUERIES.items()]
    return "\n".join(lines)


def _drc_summary(report_dir: Path, tag: str) -> str:
    rpt = report_dir / f"{tag}.drc.rpt"
    if not rpt.exists():
        # fall back to the canonical route report
        cands = sorted(report_dir.glob("*.drc.rpt"))
        if not cands:
            return "(no DRC report found)"
        rpt = cands[-1]
    counts: dict[str, int] = {}
    try:
        for m in re.finditer(r"violation type: (.+)", rpt.read_text(errors="replace")):
            t = m.group(1).strip()
            counts[t] = counts.get(t, 0) + 1
    except OSError:
        return "(DRC report unreadable)"
    if not counts:
        return f"{rpt.name}: 0 violations"
    body = "\n".join(f"  {n}x {t}" for t, n in
                     sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"{rpt.name}: {sum(counts.values())} violations\n{body}"


def run_queries(cfg, names: list[str], *, odb: Path | None, tag: str,
                stage: str, telemetry=None, log=print,
                max_chars: int = 8000) -> str:
    """Execute whitelisted queries; return one text block for the prompt."""
    names = [n for n in names if n in ALL_QUERY_NAMES]
    if not names:
        return ""
    parts: list[str] = []
    tcl_names = [n for n in names if n in TCL_QUERIES]
    if tcl_names and odb is not None and Path(odb).exists():
        body = ["define_corners tt",
                f"read_liberty -corner tt {cfg.stdcell_lib}",
                f'foreach _l [glob -nocomplain {cfg.data_dir}/lib/*.lib] {{',
                '    if {[string match "*NangateOpenCellLibrary*" $_l]} continue',
                "    read_liberty -corner tt $_l",
                "}",
                f"read_db {odb}",
                f"catch {{ read_sdc {cfg.sdc_path} }}",
                "catch { set_wire_rc -signal -layer metal3 }",
                "catch { set_propagated_clock [all_clocks] }",
                "catch { estimate_parasitics -placement }"]
        for n in tcl_names:
            body.append(f'puts "==== QUERY {n} ===="')
            body.append(TCL_QUERIES[n][1])
        script = cfg.gen_dir / f"{tag}_query.tcl"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("\n".join(body) + "\n")
        log(f"    [query] running {tcl_names} on {Path(odb).name}")
        res = runner.run_openroad(script, cfg.log_dir / f"{tag}_query.log",
                                  exe=cfg.openroad_exe, timeout_s=900,
                                  telemetry=telemetry, stage=stage)
        text = res.log_text
        # keep only from the first query marker on; drop startup noise
        idx = text.find("==== QUERY")
        parts.append(text[idx:] if idx >= 0 else text[-4000:])
    for n in names:
        if n in FILE_QUERIES:
            parts.append(f"==== QUERY {n} ====\n"
                         + _drc_summary(cfg.report_dir, tag))
    out = "\n".join(parts)
    return out[:max_chars]
