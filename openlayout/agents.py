"""The three flow agents.

Worker    — no LLM. Renders the stage's proven template with the current
            parameters and runs OpenROAD; parses logs/reports into metrics;
            saves a layout snapshot through the image_snapshot skill.
Debugger  — LLM. Inspects metrics + errors against the goal and the stage's
            DEBUGGER.md checklist; may query the tools for extra evidence;
            returns pass/fail + issues. Gets one more budget unit than the
            Optimizer so it can re-check after the last optimizer action.
Optimizer — LLM. Learns from the attempt history and debugger feedback,
            proposes up to K parameter suggestions (values clamped to the
            stage's ranges) which the system runs in parallel; the best one
            (by the config-priority goal score) is promoted.

If budgets run out the stage keeps its best result and the flow moves on —
it never aborts.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import metrics as M
from . import queries as Q
from . import runner
from .skills import load_role_skills, template_path


# ----------------------------------------------------------------------
# goal scoring — priority order comes from the config (goal.priority)
# ----------------------------------------------------------------------

DEFAULT_PRIORITY = ["timing", "drc", "tns", "placement", "wirelength",
                    "power", "area"]


def timing_violation(m: dict) -> float:
    v = 0.0
    if "wns_setup" in m:
        v += max(0.0, -float(m["wns_setup"]))
    if "wns_hold" in m:
        v += max(0.0, -float(m["wns_hold"]))
    return v


_METRIC_FOR = {
    "timing": timing_violation,
    "drc": lambda m: m.get("drc_violations", 0),
    "tns": lambda m: abs(min(0.0, float(m.get("tns", 0.0)))),
    "placement": lambda m: m.get("placement_violations", 0),
    "wirelength": lambda m: m.get("wirelength_um", m.get("hpwl", 0.0)),
    "power": lambda m: m.get("power_total_w", 0.0),
    "area": lambda m: m.get("design_area_um2", 0.0),
    "ir_drop": lambda m: m.get("ir_worst_drop_v", 0.0),
}


def score_metrics(m: dict, *, crashed: bool = False,
                  priority: list[str] | None = None) -> tuple:
    """Lower is better. First element: crashed. Then the goal metrics in the
    configured priority order (config: goal.priority)."""
    pr = [p for p in (priority or DEFAULT_PRIORITY) if p in _METRIC_FOR]
    if not pr:
        pr = DEFAULT_PRIORITY
    return (1 if crashed else 0, *(_METRIC_FOR[p](m) for p in pr))


def goal_check(m: dict, goal: dict) -> tuple[bool, list[str]]:
    """Check final metrics against the configured design goal."""
    fails: list[str] = []
    if goal.get("timing_met", True):
        ws, wh = m.get("wns_setup"), m.get("wns_hold")
        if ws is None:
            fails.append("setup wns unknown")
        elif ws < float(goal.get("max_wns", 0.0)):
            fails.append(f"setup wns {ws} < {goal.get('max_wns', 0.0)}")
        if wh is not None and wh < 0.0:
            fails.append(f"hold wns {wh} < 0")
    if goal.get("drc_clean", True):
        drc = m.get("drc_violations")
        if drc is None:
            fails.append("drc count unknown")
        elif drc > 0:
            fails.append(f"{drc} DRC violations")
    for key, mkey in (("max_area_um2", "design_area_um2"),
                      ("max_power_w", "power_total_w"),
                      ("max_ir_drop_v", "ir_worst_drop_v")):
        lim = goal.get(key)
        if lim is not None and m.get(mkey) is not None \
                and float(m[mkey]) > float(lim):
            fails.append(f"{mkey}={m[mkey]} > {lim}")
    return (not fails), fails


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

class Worker:
    def __init__(self, cfg, telemetry, log=print):
        self.cfg = cfg
        self.telemetry = telemetry
        self.log = log

    def base_params(self, spec, tag: str) -> dict:
        c = self.cfg
        return {
            "data_dir": str(c.data_dir),
            "tech_dir": str(c.tech_dir_path),
            "proj_name": c.design,
            "top_design": c.top,
            "netlist_file": str(c.netlist_path),
            "sdc_file": str(c.sdc_path),
            "fp_def": str(c.floorplan_def_path),
            "save_dir": str(c.save_dir),
            "report_dir": str(c.report_dir),
            "image_dir": str(c.image_dir),
            "tag": tag,
            "num_threads": c.threads,
        }

    def run(self, spec, params: dict, tag: str, *,
            in_odb: Path | None = None, out_odb: Path | None = None,
            gen_dir: Path | None = None, log_dir: Path | None = None,
            image_dir: Path | None = None):
        """Render + run one stage script; snapshot the result; return
        (RunResult, metrics dict)."""
        c = self.cfg
        gen_dir = gen_dir or c.gen_dir
        log_dir = log_dir or c.log_dir
        full = dict(self.base_params(spec, tag))
        full.update(spec.fixed_params)
        full.update(params)
        if in_odb is not None:
            full["in_odb"] = str(in_odb)
        if out_odb is not None:
            full["out_odb"] = str(out_odb)
        script = runner.render(spec.template, full, gen_dir / f"{tag}.tcl")
        self.log(f"    [worker] run {spec.template} tag={tag}")
        res = runner.run_openroad(
            script, log_dir / f"{tag}.log", exe=c.openroad_exe,
            timeout_s=c.tool_timeout_s, telemetry=self.telemetry,
            stage=spec.name)
        m = M.parse_log_metrics(res.log_text)
        report_dir = Path(full["report_dir"])
        drc_rpt = report_dir / f"{tag}.drc.rpt"
        if drc_rpt.exists():
            n = M.count_drc_violations(drc_rpt)
            if n is not None:
                m["drc_violations"] = n
        if not res.ok:
            m["crashed"] = True
        self.log(f"    [worker] done ok={res.ok} t={res.seconds:.0f}s "
                 f"metrics={json.dumps(m)[:220]}")
        # image via the image_snapshot skill (steps that need images)
        if res.ok and out_odb is not None and Path(out_odb).exists():
            self.snapshot(Path(out_odb), tag,
                          image_dir=image_dir or Path(full["image_dir"]),
                          gen_dir=gen_dir, log_dir=log_dir, stage=spec.name)
        return res, m

    def snapshot(self, odb: Path, tag: str, *, image_dir: Path,
                 gen_dir: Path | None = None, log_dir: Path | None = None,
                 stage: str = "") -> Path | None:
        """Save a whole-die PNG of a checkpoint via the image_snapshot skill."""
        c = self.cfg
        tpl = template_path("image_snapshot", "snapshot.tcl",
                            c.skills_flow_dir)
        if tpl is None:
            return None
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        out_png = image_dir / f"{tag}.png"
        gen_dir = gen_dir or c.gen_dir
        script = runner.render(tpl, {
            "in_odb": str(odb), "out_png": str(out_png),
            "img_width": int(c.raw.get("image", {}).get("width", 1024)),
            "img_margin": float(c.raw.get("image", {}).get("margin", 0.03)),
        }, gen_dir / f"{tag}_snapshot.tcl")
        res = runner.run_openroad(
            script, (log_dir or c.log_dir) / f"{tag}_snapshot.log",
            exe=c.openroad_exe, timeout_s=300, telemetry=self.telemetry,
            stage=stage or "snapshot")
        if res.ok and out_png.exists():
            self.log(f"    [snapshot] {out_png.name}")
            return out_png
        self.log(f"    [snapshot] failed for {tag}")
        return None


# ----------------------------------------------------------------------
# LLM agents — shared query round
# ----------------------------------------------------------------------

QUERY_NOTE = """
If the evidence above is not enough, you may request extra tool evidence:
answer ONLY {"queries": ["<name>", ...]} (max %d) using these query names:
%s
The system will run them on the current checkpoint and ask you again. Request
queries only when they would change your answer."""


class _LLMAgent:
    role = "agent"

    def __init__(self, cfg, llm, log=print):
        self.cfg = cfg
        self.llm = llm
        self.log = log

    def _ask_with_queries(self, system: str, user: str, *, spec,
                          odb: Path | None, tag: str) -> dict:
        """One LLM call, plus at most one tool-query round if requested."""
        allow = self.cfg.allow_tool_queries
        if allow:
            user = user + "\n" + QUERY_NOTE % (self.cfg.max_queries,
                                               Q.queries_help())
        out = self.llm.chat_json(system, user, stage=spec.name,
                                 agent=self.role,
                                 dialog_dir=self.cfg.dialog_dir)
        qnames = out.get("queries") if allow else None
        if qnames:
            qnames = [str(n) for n in qnames][:self.cfg.max_queries]
            self.log(f"    [{self.role}] requested queries: {qnames}")
            results = Q.run_queries(self.cfg, qnames, odb=odb, tag=tag,
                                    stage=spec.name, telemetry=self.llm.telemetry,
                                    log=self.log)
            user2 = (user + "\n\n==== TOOL QUERY RESULTS ====\n"
                     + (results or "(queries produced no output)")
                     + "\n\nNow give your FINAL JSON answer "
                       "(no more queries).")
            out = self.llm.chat_json(system, user2, stage=spec.name,
                                     agent=self.role,
                                     dialog_dir=self.cfg.dialog_dir)
            out.pop("queries", None)
        return out


# ----------------------------------------------------------------------
# Debugger
# ----------------------------------------------------------------------

DEBUGGER_SYSTEM = """You are the Debugger agent of an OpenROAD physical
design flow. You check the current stage result against the design goal and
the stage checklist, using the log evidence given. Be precise and grounded:
only report issues supported by the data. Answer STRICT JSON:
{"verdict": "pass" | "fail",
 "severity": "none" | "minor" | "major" | "fatal",
 "issues": ["<short issue>", ...],
 "root_cause": "<one-line most likely root cause or empty>",
 "recommendation": "<one-line advice for the Optimizer or empty>"}
Rules: verdict=pass when the stage completed and no goal-relevant problem
remains at this stage. A crashed run is always fail/fatal. Do not fail a
stage for metrics that later stages will naturally fix (e.g. hold before CTS,
DRC before routing). The "design context" section is BACKGROUND HISTORY from
earlier stages/attempts — judge ONLY the current run's log evidence and
metrics; never report an old failure as a current issue."""


class Debugger(_LLMAgent):
    role = "debugger"

    def check(self, spec, *, params: dict, m: dict, res,
              context_summary: str, attempt_no: int,
              odb: Path | None = None, tag: str = "") -> dict:
        skills = load_role_skills("debugger", spec.skill_topics,
                                  self.cfg.skills_flow_dir)
        errors = M.extract_errors(res.log_text)
        warns = M.extract_warning_summary(res.log_text)
        user = f"""stage: {spec.name} (attempt {attempt_no})
design goal: {json.dumps(self.cfg.goal)}
stage params: {json.dumps(params)}
stage completed without crash: {res.ok} (tool time {res.seconds:.0f}s)
extracted metrics: {json.dumps(m)}

error lines:
{chr(10).join(errors) if errors else '(none)'}

warning summary:
{chr(10).join(warns) if warns else '(none)'}

log tail:
{M.tail(res.log_text, 40)}

design context so far:
{context_summary}

{skills}

Check the design at this stage. Answer the JSON verdict."""
        try:
            out = self._ask_with_queries(DEBUGGER_SYSTEM, user, spec=spec,
                                         odb=odb, tag=tag or spec.name)
        except Exception as exc:  # LLM outage → conservative local fallback
            self.log(f"    [debugger] LLM failed ({exc}); local heuristic")
            bad = (not res.ok) or bool(errors)
            out = {"verdict": "fail" if bad else "pass",
                   "severity": "fatal" if not res.ok else
                   ("major" if errors else "none"),
                   "issues": errors[:5],
                   "root_cause": "", "recommendation": ""}
        out.setdefault("verdict", "fail")
        out.setdefault("issues", [])
        self.log(f"    [debugger] verdict={out['verdict']} "
                 f"issues={out['issues'][:3]}")
        return out


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------

OPTIMIZER_SYSTEM = """You are the Optimizer agent of an OpenROAD physical
design flow. Given the stage's attempt history (parameters -> metrics), the
Debugger's findings and the tunable parameter space, propose parameter
changes for the next runs. The system will run every suggestion in parallel
and keep the best result, so make suggestions DIVERSE (attack the problem
from different angles). Learn from the history: never repeat a parameter set
that already ran; move further in directions that helped, revert directions
that hurt. Answer STRICT JSON:
{"suggestions": [{"params": {<param>: <value>, ...},
                  "rationale": "<why>"}, ...],
 "stop": false,
 "reason": "<if stop=true, why further optimization is pointless>"}
Only use parameter names from the given parameter space; values must respect
the stated ranges. Set stop=true only when the goal is already met or no
parameter change can plausibly help (e.g. failure is caused by the input
design, not this stage). The "design context" section is background history —
base decisions on the attempt history and the current debugger findings."""


class Optimizer(_LLMAgent):
    role = "optimizer"

    def propose(self, spec, *, history: list[dict], debug_report: dict,
                context_summary: str, n_suggestions: int,
                odb: Path | None = None, tag: str = "",
                own_history: str = "") -> dict:
        skills = load_role_skills("optimizer", spec.skill_topics,
                                  self.cfg.skills_flow_dir)
        hist_lines = []
        for h in history:
            hist_lines.append(
                f"- attempt {h['tag']}: params={json.dumps(h['params'])} "
                f"-> ok={h['ok']} metrics={json.dumps(h['metrics'])[:400]}"
                f"{'  << current best' if h.get('best') else ''}")
        user = f"""stage: {spec.name}
design goal: {json.dumps(self.cfg.goal)} (priority order: {self.cfg.goal_priority})
tunable parameter space:
{spec.param_space}

attempt history (this invocation):
{chr(10).join(hist_lines)}

your previous actions on THIS stage (earlier invocations — learn from them,
never repeat a losing direction):
{own_history or '(none)'}

debugger findings: {json.dumps(debug_report)}

design context:
{context_summary}

{skills}

Propose up to {n_suggestions} diverse parameter suggestions (or stop)."""
        try:
            out = self._ask_with_queries(OPTIMIZER_SYSTEM, user, spec=spec,
                                         odb=odb, tag=tag or spec.name)
        except Exception as exc:
            self.log(f"    [optimizer] LLM failed ({exc}); no suggestions")
            return {"suggestions": [], "stop": True, "reason": str(exc)}
        sugg = out.get("suggestions") or []
        # keep only whitelisted params, clamped to their stated ranges
        clean = []
        for s in sugg[:n_suggestions]:
            p = {}
            for k, v in (s.get("params") or {}).items():
                if k not in spec.tunable_params:
                    continue
                rng = spec.param_ranges.get(k)
                if rng is not None and isinstance(v, (int, float)):
                    lo, hi = rng
                    if v < lo or v > hi:
                        self.log(f"    [optimizer] clamp {k}={v} to "
                                 f"[{lo},{hi}]")
                        v = min(max(v, lo), hi)
                        v = type(lo)(v) if isinstance(lo, int) else v
                p[k] = v
            if p:
                clean.append({"params": p,
                              "rationale": str(s.get("rationale", ""))[:400]})
        out["suggestions"] = clean
        self.log(f"    [optimizer] {len(clean)} suggestion(s) "
                 f"stop={out.get('stop', False)}")
        return out

    # ------------------------------------------------------------------
    def run_suggestions(self, worker: Worker, spec, base_params: dict,
                        suggestions: list[dict], round_no: int,
                        in_odb: Path | None, parallel: bool) -> list[dict]:
        """Run each suggestion in parallel; return attempt records."""
        c = self.cfg

        def _one(k_s):
            k, s = k_s
            tag = f"{spec.name}_opt{round_no}_{k}"
            p = dict(base_params)
            p.update(s["params"])
            out_odb = c.save_dir / f"{tag}.odb"
            res, m = worker.run(spec, p, tag, in_odb=in_odb, out_odb=out_odb)
            return {"tag": tag, "params": p, "changed": s["params"],
                    "rationale": s["rationale"], "ok": res.ok, "metrics": m,
                    "odb": str(out_odb), "res": res}

        items = list(enumerate(suggestions))
        if parallel and len(items) > 1:
            with ThreadPoolExecutor(max_workers=len(items)) as ex:
                return list(ex.map(_one, items))
        return [_one(i) for i in items]


def promote(attempt: dict, canonical_odb: Path, log=print) -> None:
    """Copy a winning variant's odb to the stage's canonical output path."""
    src = Path(attempt.get("odb", ""))
    if src.exists() and src != canonical_odb:
        shutil.copyfile(src, canonical_odb)
        log(f"    [select] promoted {src.name} -> {canonical_odb.name}")
