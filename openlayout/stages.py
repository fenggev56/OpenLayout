"""Stage specifications — template, canonical odb names, tunable parameter
space (what the Optimizer may touch) and the skills each agent receives.

Templates live in ``openlayout/scripts/`` and use guarded TCL variables so a
rendered header (``set var value``) overrides any subset of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageSpec:
    name: str
    template: str
    output_odb: str                       # file name under save_dir
    input_stage: str | None = None        # stage whose output_odb this reads
    fixed_params: dict = field(default_factory=dict)
    default_params: dict = field(default_factory=dict)
    tunable_params: set = field(default_factory=set)
    param_ranges: dict = field(default_factory=dict)   # name -> (lo, hi)
    param_space: str = ""
    # skill topic dirs under the flow skill library; each may hold SKILL.md
    # (shared), DEBUGGER.md and OPTIMIZER.md (role-specific) — extensible
    skill_topics: list = field(default_factory=list)


FLOORPLAN = StageSpec(
    name="floorplan",
    template="floorplan_init.tcl",
    output_odb="00_floorplan.odb",
    skill_topics=["floorplan"],
)

MACRO_PLACE = StageSpec(          # executed specially (macroflow candidates)
    name="macro_place",
    template="apply_macro_coords.tcl",
    output_odb="01_macro_placed.odb",
    input_stage="floorplan",
    skill_topics=[],
)

DETAIL_PLACE = StageSpec(
    name="detail_place",
    template="detail_place.tcl",
    output_odb="02_detail_place.odb",
    input_stage="macro_place",
    default_params={
        "place_density": 0.60,         # proven bp_fe/ariane setting; 0.0 = auto
        "halo_um": 5.0,
        "pdn_pitch": 56.0,
        "routability_driven": 1,
        "timing_driven": 0,            # proven flow places without -timing_driven
        "run_resizer": 1,
    },
    tunable_params={"place_density", "halo_um", "pdn_pitch",
                    "routability_driven", "timing_driven", "run_resizer"},
    param_ranges={"place_density": (0.0, 0.95), "halo_um": (2.0, 10.0),
                  "pdn_pitch": (40.0, 80.0), "routability_driven": (0, 1),
                  "timing_driven": (0, 1), "run_resizer": (0, 1)},
    param_space="""- place_density: float, 0.0 (auto) or 0.40..0.95. Higher packs
  cells tighter (shorter wires, more congestion); lower spreads them.
- halo_um: float 2.0..10.0. Placement keepout halo around each macro.
- pdn_pitch: float 40.0..80.0. PDN stripe pitch in um (metal8/9 stripes);
  larger pitch frees routing tracks, smaller improves IR drop.
- routability_driven: 0/1. global_placement -routability_driven.
- timing_driven: 0/1. global_placement -timing_driven.
- run_resizer: 0/1. Run repair_design + repair_timing -setup after placement.""",
    skill_topics=["detail_place"],
)

CTS = StageSpec(
    name="cts",
    template="cts.tcl",
    output_odb="03_cts.odb",
    input_stage="detail_place",
    default_params={
        "wire_unit": 20,
        "root_buf": "CLKBUF_X3",
        "buf_list": "CLKBUF_X1 CLKBUF_X2 CLKBUF_X3",
        "cluster_size": 30,
        "cluster_diameter": 100,
        "repair_setup": 1,
        "repair_hold": 1,
        "hold_margin": 0.05,
    },
    tunable_params={"wire_unit", "root_buf", "buf_list", "cluster_size",
                    "cluster_diameter", "repair_setup", "repair_hold",
                    "hold_margin"},
    param_ranges={"wire_unit": (10, 100), "cluster_size": (10, 60),
                  "cluster_diameter": (40, 200), "repair_setup": (0, 1),
                  "repair_hold": (0, 1), "hold_margin": (0.0, 0.15)},
    param_space="""- wire_unit: int 10..100. CTS wire unit (buffer insertion granularity).
- root_buf: one of CLKBUF_X1 CLKBUF_X2 CLKBUF_X3 BUF_X4 BUF_X8.
- buf_list: space-separated subset of CLKBUF_X1 CLKBUF_X2 CLKBUF_X3.
- cluster_size: int 10..60 (sink clustering size).
- cluster_diameter: int 40..200 um (sink clustering max diameter).
- repair_setup: 0/1 run post-CTS repair_timing -setup.
- repair_hold: 0/1 run post-CTS repair_timing -hold.
- hold_margin: float 0.0..0.15 for hold repair.""",
    skill_topics=["cts"],
)

ROUTE = StageSpec(
    name="route",
    template="route.tcl",
    output_odb="04_route.odb",
    input_stage="cts",
    default_params={
        "layer_adjustment": 0.3,
        "sig_min_layer": "metal2",
        "sig_max_layer": "metal9",
        "clk_min_layer": "metal3",
        "clk_max_layer": "metal7",
        "congestion_iterations": 30,
        "droute_end_iter": 8,
        "post_grt_repair_design": 1,
        "post_grt_repair_setup": 1,
        "post_grt_repair_hold": 1,
        "hold_margin": 0.05,
    },
    tunable_params={"layer_adjustment", "sig_min_layer", "sig_max_layer",
                    "clk_min_layer", "clk_max_layer",
                    "congestion_iterations", "droute_end_iter",
                    "post_grt_repair_design", "post_grt_repair_setup",
                    "post_grt_repair_hold", "hold_margin"},
    param_ranges={"layer_adjustment": (0.1, 0.7),
                  "congestion_iterations": (10, 80),
                  "droute_end_iter": (5, 30),
                  "post_grt_repair_design": (0, 1),
                  "post_grt_repair_setup": (0, 1),
                  "post_grt_repair_hold": (0, 1),
                  "hold_margin": (0.0, 0.15)},
    param_space="""- layer_adjustment: float 0.1..0.7. Global-route capacity derate on
  metal2-metal10 (lower = more capacity used, fewer detours; higher = safer).
- sig_min_layer/sig_max_layer: signal routing layer range (metal2..metal10).
- clk_min_layer/clk_max_layer: clock routing layer range.
- congestion_iterations: int 10..80 global route congestion iterations.
- droute_end_iter: int 5..30. More detailed-route iterations remove more
  DRCs at the cost of runtime.
- post_grt_repair_design / post_grt_repair_setup / post_grt_repair_hold: 0/1
  timing repair after global route (before detailed route).
- hold_margin: float 0.0..0.15 for post-GRT hold repair.""",
    skill_topics=["route"],
)

FINISH = StageSpec(
    name="finish",
    template="finish.tcl",
    output_odb="05_finish.odb",
    input_stage="route",
    default_params={
        "ir_source_type": "FULL",
        "vdd_voltage": 1.1,
    },
    tunable_params={"ir_source_type", "vdd_voltage",
                    # upstream reruns (owner mapping in flow.py):
                    "pdn_pitch", "droute_end_iter", "layer_adjustment",
                    "hold_margin"},
    param_ranges={"vdd_voltage": (0.9, 1.32), "pdn_pitch": (40.0, 80.0),
                  "droute_end_iter": (5, 30), "layer_adjustment": (0.1, 0.7),
                  "hold_margin": (0.0, 0.15)},
    param_space="""- ir_source_type: FULL | STRAPS | BUMPS. Voltage source model for
  analyze_power_grid.
- vdd_voltage: float, nominal VDD (Nangate45 = 1.1).
Upstream parameters (choosing one re-runs the flow from the owning stage
with previous successful scripts, then re-checks):
- pdn_pitch (detail_place): denser PDN reduces IR drop.
- layer_adjustment, droute_end_iter, hold_margin (route): fix residual DRC
  or hold violations found at signoff.""",
    skill_topics=["finish", "ir_drop"],
)

ALL_STAGES: dict[str, StageSpec] = {
    s.name: s for s in
    (FLOORPLAN, MACRO_PLACE, DETAIL_PLACE, CTS, ROUTE, FINISH)
}

# which stage owns each finish-tunable upstream parameter
PARAM_OWNER = {
    "pdn_pitch": "detail_place",
    "layer_adjustment": "route",
    "droute_end_iter": "route",
    "hold_margin": "route",
}

STAGE_ORDER = ["floorplan", "macro_place", "detail_place", "cts", "route",
               "finish"]
