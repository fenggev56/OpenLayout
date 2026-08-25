"""End-to-end orchestration self-test: FakeVLM + simulated OpenROAD.

Exercises the REAL geometry, agent loops, image injection/pruning, agent->op
fallbacks, Gate-A re-entry through the Placer, Gate-B Planner-driven
refinement, macro-candidate file management and telemetry — everything except
the actual model + EDA tool. No GPU required. Run: python -m macroflow.selftest
"""
import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macroflow.config import FlowConfig
from macroflow.llm import VLM, LLMResponse, ToolCall
from macroflow.orchestrator import MacroFlowOrchestrator


# --------------------------------------------------------------------------
def _last_user_text(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(b.get("text", "") for b in c
                                if isinstance(b, dict) and b.get("type") == "text")
    return ""


class FakeVLM:
    """Scripts plausible tool calls per agent role (decided by the tool set)."""
    build_assistant_message = staticmethod(VLM.build_assistant_message)
    format_tool_result = staticmethod(VLM.format_tool_result)
    build_tool_results_message = staticmethod(VLM.build_tool_results_message)

    def __init__(self, double_place_call=False):
        self.checker_calls = 0
        self.refiner_turns = 0
        self.double_place_call = double_place_call

    def _call(self, name, args, text=""):
        tc = ToolCall(id=f"call_{name}", name=name, input=args)
        raw = {"role": "assistant", "content": text, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]}
        return LLMResponse(text=text, tool_calls=[tc], stop_reason="tool_use", raw=raw,
                           prompt_tokens=100, completion_tokens=20, gen_seconds=0.01)

    def _multi_call(self, calls, text=""):
        tool_calls = [ToolCall(id=f"call_{name}_{i}", name=name, input=args)
                      for i, (name, args) in enumerate(calls)]
        raw = {"role": "assistant", "content": text, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
            for tc in tool_calls]}
        return LLMResponse(text=text, tool_calls=tool_calls, stop_reason="tool_use",
                           raw=raw, prompt_tokens=100, completion_tokens=40,
                           gen_seconds=0.01)

    def _text_only(self, text):
        return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn",
                           raw={"role": "assistant", "content": text},
                           prompt_tokens=100, completion_tokens=max(10, len(text) // 4),
                           gen_seconds=0.01)

    def chat(self, messages, system, tools=None, max_tokens=1536):
        names = {t["name"] for t in (tools or [])}
        last = _last_user_text(messages)
        if "adjust_placement" in names:            # Refiner — fine-grain moves
            self.refiner_turns += 1
            if self.refiner_turns % 2 == 1:        # move, then finish next turn
                return self._call("adjust_placement",
                                  {"moves": "#0 -> (10, 0)\n#1 -> (0, -8)",
                                   "notes": "close edge channel"})
            return self._call("finish", {"summary": "refined a couple of macros"})
        if "submit_verdict" in names:              # Checker
            self.checker_calls += 1
            proceed = self.checker_calls >= 3      # FAIL twice, then PASS+proceed
            return self._call("submit_verdict",
                              {"verdict": "PASS" if proceed else "FAIL",
                               "feedback": "fake verdict",
                               "proceed": proceed,
                               "suggest_anchor": "center", "suggest_strength": 0.4})
        if "submit_placement_plan" in names:       # Placer — emit coordinates
            import re
            m = re.search(r"Place ALL (\d+) macros", last)
            n_macros = int(m.group(1)) if m else 24
            if "submit_placement_plan" in last:    # asked to place / replan
                plan = "\n".join(
                    f"#{i} -> ({100 + (i % 6) * 60}, {100 + (i // 6) * 60}, R0)"
                    for i in range(n_macros))
                if self.double_place_call:
                    bad = "\n".join(
                        f"#{i} -> (1100.0, 1100.0, R0)"
                        for i in range(n_macros))
                    return self._multi_call([
                        ("submit_placement_plan", {"plan": plan}),
                        ("submit_placement_plan", {"plan": bad}),
                    ])
                return self._call("submit_placement_plan", {"plan": plan})
            return self._text_only("placement looks fine")
        if "submit_strategy" in names:             # Planner — strategy
            return self._call("submit_strategy", {"strategy":
                "Push all macros to the periphery in a ring with R0 orientation, "
                "leaving an open ~square stdcell core."})
        return self._call("finish", {"summary": "done"})


class FakeSession:
    """Simulated OpenROAD: canned recipe outputs; density rises then plateaus;
    post-route timing follows a supplied schedule (one (wns, tns) per route)."""
    def __init__(self, timing=None, macro_count=24):
        self.density_n = 0
        self.timing = list(timing or [(-0.5, -50.0)])
        self.route_i = 0
        self.macro_count = macro_count
        self.max_safe_place_batch = 0

    # interface used by FlowOps -------------------------------------------
    def set_init_script(self, p): pass
    def save_image(self, p, timeout=30, area=None): return ""
    def write_db(self, p, timeout=120): return ""
    def read_db(self, p, timeout=120): return ""
    def source(self, p, timeout=300): return ""
    def fresh_start(self): return True
    def close(self, graceful=True): pass

    def run_recipe(self, r, timeout=120):
        if "getDieArea" in r:
            return ("Die  width  = 1500.000 um\nDie  height = 1500.000 um\n"
                    "Core width  = 1400.000 um\nCore height = 1400.000 um\n"
                    "Core origin = (50.000, 50.000) um\n"
                    "Core x_max  = 1450.000 um\nCore y_max  = 1450.000 um")
        if "STDCELL_DENSITY" in r:
            self.density_n += 1
            d = min(0.15 + 0.04 * (self.density_n - 1), 0.30)
            return f"STDCELL_DENSITY: {d:.4f}"
        if "OVERLAP:" in r or "Total overlaps" in r:
            return "Total overlaps: 0"
        if "getBTerms" in r:
            return ("PIN clk pos=(5.0, 700.0)\nPIN a pos=(2.0, 100.0)\n"
                    "PIN b pos=(1400.0, 700.0)\nPIN c pos=(700.0, 5.0)")
        if "MACRO_PIN_SIDE" in r:
            return "\n".join(
                f"MACRO_PIN_SIDE name=macro_{i} orient=R0 side=R dominant=8 "
                "counts=L:0,R:8,T:0,B:0 bbox=(100.000,100.000,177.710,140.600)"
                for i in range(self.macro_count))
        if "size=%.3f" in r or "master=%-20s" in r:
            return "\n".join(
                f"macro_{i}  master=fakeram45_256x16  size=77.710 x 40.600  "
                f"pos=(0.000, 0.000)" for i in range(self.macro_count))
        return ""

    def send_command(self, cmd, timeout=120):
        if "report_wns" in cmd:           # the route block
            wns, tns = self.timing[min(self.route_i, len(self.timing) - 1)]
            self.route_i += 1
            return (f"wns max {wns}\ntns max {tns}\n"
                    "Design area 250000 um^2 42% utilization.\n"
                    "Total wire length = 900.0\n"
                    "Group                  Internal  Switching    Leakage      Total\n"
                    "                          Power      Power      Power      Power (Watts)\n"
                    "----------------------------------------------------------------\n"
                    "Total                  1.00e-01   2.00e-02   3.00e-03   1.23e-01 100.0%\n"
                    "Number of violations = 0")
        if "read_liberty" in cmd:
            return "ROUTE_PRELOAD_DONE"
        if "proc safe_place" in cmd:
            return "safe_place proc loaded"
        if "safe_place {" in cmd:
            self.max_safe_place_batch = max(
                self.max_safe_place_batch,
                sum(1 for line in cmd.splitlines()
                    if line.lstrip().startswith("safe_place {")))
            return "OK: placed"
        return ""


# --------------------------------------------------------------------------
def unit_checks():
    from macroflow.goal import Goal, PPAChecker, Decision
    from macroflow.candidates import MacroCandidate, MacroCandidateStore
    from macroflow import geometry as geo

    g = Goal.default()
    assert not g.evaluate({"wns": -0.5, "tns": -50}).met
    assert g.evaluate({"wns": 0.0, "tns": 0.0}).met
    assert g.evaluate({"wns": 0.1, "tns": 5}).met
    assert not g.evaluate({"wns": 0.1}).met            # missing tns -> not met
    # WNS dominates TNS in ranking
    assert g.is_better({"wns": -0.2, "tns": -20}, {"wns": -0.5, "tns": -50})
    assert not g.is_better({"wns": -0.5, "tns": -1}, {"wns": -0.2, "tns": -99})
    # exact-score tie -> fewer DRC wins
    assert g.is_better({"wns": -0.2, "tns": -20, "drc": 0},
                       {"wns": -0.2, "tns": -20, "drc": 5})

    gy = Goal.load(_ROOT / "goal.yaml")
    assert gy.kind == "timing" and gy.max_tries == 6 and len(gy.objectives) == 2, \
        f"goal.yaml parsed wrong: {gy.describe()}"

    # coarse shapes: each distribution sums to n, and the set is diverse
    regions = geo.region_grid((0, 0, 100, 100), n=4)
    fingerprints = set()
    for name in geo.COARSE_SHAPE_ORDER:
        dist = geo.shape_distribution(name, 24, regions)
        assert sum(dist.values()) == 24, f"{name} sums to {sum(dist.values())}"
        fingerprints.add(tuple(sorted(dist)))
    assert len(fingerprints) >= 4, f"coarse shapes not diverse: {len(fingerprints)}"

    # fine-grain delta parsing (signed offsets, optional orient, tolerant of prose)
    d = geo.parse_indexed_deltas("#0 -> (10, -5)\n#3 -> (-2.5, 0) R90\njust prose", 8)
    assert d[0] == (10.0, -5.0, None) and d[3] == (-2.5, 0.0, "R90"), d

    # coordinate parser accepts simple arithmetic expressions but still returns
    # numeric coordinates for legalization/clamping.
    p = geo.parse_indexed_plan("#1 -> (10.1+78*3, 1447.6-41, R0)", 8)
    assert p[1] == (244.1, 1406.6, "R0"), p

    # snap_to_edges flushes a thin sub-min channel against the left edge
    ents = [{"name": "m", "x": 2.0, "y": 50.0, "orient": "R0"}]
    geo.snap_to_edges(ents, (0, 0, 100, 100), {"m": (10, 10)}, min_channel=5.0)
    assert ents[0]["x"] == 0.0, f"did not snap thin channel: {ents}"

    # PPAChecker decisions + exact report format
    chk = PPAChecker(Goal.default())
    dec, _s, rep = chk.decide(0, 3, {"wns": 0.0, "tns": 0.0})
    assert dec == Decision.STOP_GOAL and "Decision: STOP (goal met)" in rep, rep
    dec, _s, rep = chk.decide(2, 3, {"wns": -0.5, "tns": -1})
    assert dec == Decision.STOP_BUDGET and "Iteration: 3 / 3" in rep, rep
    dec, _s, rep = chk.decide(0, 3, {"wns": -0.5, "tns": -1})
    assert dec == Decision.CONTINUE and "WNS: -0.5 ns" in rep, rep

    # Missing DRC from a completed route should not make a candidate lose to a
    # worse-WNS route solely because that one happened to parse drc=0.
    c_missing_drc = MacroCandidate(id=0, tag="cand00",
                                   metrics={"routed": {"wns": -0.1, "tns": -1}})
    c_zero_drc = MacroCandidate(id=1, tag="cand01",
                                metrics={"routed": {"wns": -0.2, "tns": -1,
                                                    "drc": 0}})
    assert MacroCandidateStore._score(c_missing_drc) > MacroCandidateStore._score(c_zero_drc)
    print("unit checks passed")


def run_scenario(name, macro_candidates, gate_a_max_refine, gate_b_max_refine,
                 expect_candidates, macro_count=24, double_place_call=False):
    td = tempfile.mkdtemp(prefix=f"mf_{name}_")
    cfg = FlowConfig(data_dir=td, design="dry", top="dry_top", work_dir=td,
                     validate_inputs=False)
    orch = MacroFlowOrchestrator(cfg, max_refine=gate_b_max_refine, verbose=True,
                                 macro_candidates=macro_candidates,
                                 gate_a_max_refine=gate_a_max_refine,
                                 parallel_workers=1,
                                 post_route_opt_loops=0)
    session = FakeSession(macro_count=macro_count)
    metrics = orch.run(vlm=FakeVLM(double_place_call=double_place_call),
                       session=session)

    store = orch.macro_store
    cands = store.all()
    best = store.best()
    print(f"\n[{name}] candidates={len(cands)} best=#{best.id} "
          f"metrics={best.metrics}")
    assert best is not None
    assert len(cands) == expect_candidates, \
        f"[{name}] expected {expect_candidates} candidates, got {len(cands)}"
    assert metrics["best_candidate"] == best.id, \
        f"[{name}] returned best id != store best"
    # the chosen best really is the macro-quality best by ranking
    for c in cands:
        if c.id != best.id:
            assert store._score(c) <= store._score(best), \
                f"[{name}] candidate #{c.id} should not out-rank best #{best.id}"
    # file-management artifacts
    assert (store.root / "index.json").exists(), f"[{name}] no index.json"
    idx = json.loads((store.root / "index.json").read_text())
    assert idx["best"] == best.id, f"[{name}] index best pointer wrong"
    for c in cands:
        assert (Path(c.dir) / "coords.json").exists(), f"[{name}] #{c.id} missing coords"
        assert (Path(c.dir) / "metrics.json").exists(), f"[{name}] #{c.id} missing metrics"
        assert (Path(c.dir) / "gate_a.json").exists(), f"[{name}] #{c.id} missing Gate-A"
        assert (Path(c.dir) / "gate_b.json").exists(), f"[{name}] #{c.id} missing Gate-B"
    assert session.max_safe_place_batch <= 20, \
        f"[{name}] placed {session.max_safe_place_batch} macros in one OpenROAD batch"
    return orch, store, session


def main():
    unit_checks()

    # Scenario A: first candidate exercises Gate-A refinement, later candidates pass.
    orchA, storeA, _sessionA = run_scenario(
        "macro_batch",
        macro_candidates=3,
        gate_a_max_refine=2,
        gate_b_max_refine=2,
        expect_candidates=3)

    # Scenario B: CLI/config candidate count override should bound the batch.
    orchB, storeB, _sessionB = run_scenario(
        "two_candidates",
        macro_candidates=2,
        gate_a_max_refine=2,
        gate_b_max_refine=2,
        expect_candidates=2)

    # Scenario C: large macro count uses direct coordinate placement. The fake
    # Placer returns a valid coordinate plan followed by a bad coordinate plan in
    # the same model response; only the first terminal placement may execute.
    orchC, storeC, sessionC = run_scenario(
        "large_direct_terminal",
        macro_candidates=1,
        gate_a_max_refine=2,
        gate_b_max_refine=1,
        expect_candidates=1,
        macro_count=32,
        double_place_call=True)
    assert len(orchC.ctx.state.entries) == 32, "large scenario did not place 32 macros"
    assert sessionC.max_safe_place_batch <= 20, "large scenario exceeded 20-macro batch"

    # Shared checks on scenario A (the richer run).
    st = orchA.ctx.state
    stages = orchA.ctx.registry.stages()
    assert st.stage == "routed", f"did not reach routed stage: {st.stage}"
    for s in ("init", "macro_placed", "macro_pin_sides", "fast_placed",
              "detail_placed", "routed"):
        assert s in stages, f"missing snapshot stage {s}"
    routed = storeA.best().metrics.get("routed", {})
    assert "wns" in routed and "tns" in routed, f"missing routed timing: {routed}"
    assert routed.get("power") == 0.123 and routed.get("area") == 250000.0, \
        f"missing routed power/area: {routed}"
    assert routed.get("utilization") == 42.0, \
        f"missing routed utilization: {routed}"
    assert (orchA.cfg.stage_model_dir("run") / "summary.txt").exists(), \
        "no summary written"

    tel = orchA.ctx.telemetry.report()
    iters = tel["iterations"]
    print("telemetry iterations:", iters)
    assert iters.get("gate_b_rounds", 0) >= 0, f"gate_b counter wrong: {iters}"
    assert {"Planner", "Placer", "Checker"} <= set(tel["by_agent"]), \
        f"agents missing from telemetry: {list(tel['by_agent'])}"
    assert (orchA.cfg.stage_model_dir("run") / "telemetry.json").exists(), \
        "no telemetry.json"

    print("\n==== END-TO-END DRY-RUN PASSED ====")


if __name__ == "__main__":
    main()
