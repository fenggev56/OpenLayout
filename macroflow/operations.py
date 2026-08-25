"""EDA operations — the "Python handles system operation" half.

Every operation an agent can trigger is a method here: it drives OpenROAD,
applies the deterministic geometry, manages placement state, saves/registers
images, and returns a small structured summary. The agents never touch OpenROAD
or coordinates directly — they call tools, which call these.

The placement sequence faithfully reproduces the validated classic flow:
reload the pristine ``floorplan.odb`` → place macros in size-sorted batches →
fast-place for feedback → save candidate artifacts. Detail/route helpers remain
available as explicit stage utilities but are not part of the macro-candidate
optimization loop.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from . import geometry as geo
from . import metrics as M
from . import recipes as R
from .context import FlowContext

BATCH = 20
SIDES = ("L", "R", "T", "B")
OPPOSITE_SIDE = {"L": "R", "R": "L", "T": "B", "B": "T"}
ORIENT_SIDE_MAP = {
    "R0": {"L": "L", "R": "R", "T": "T", "B": "B"},
    "R90": {"L": "B", "R": "T", "T": "L", "B": "R"},
    "R180": {"L": "R", "R": "L", "T": "B", "B": "T"},
    "R270": {"L": "T", "R": "B", "T": "R", "B": "L"},
    "MX": {"L": "L", "R": "R", "T": "B", "B": "T"},
    "MY": {"L": "R", "R": "L", "T": "T", "B": "B"},
    "MXR90": {"L": "T", "R": "B", "T": "L", "B": "R"},
    "MYR90": {"L": "B", "R": "T", "T": "R", "B": "L"},
}
PREFERRED_ORIENTS = ("R0", "MX", "MY", "R180", "R90", "R270", "MXR90", "MYR90")
FOOTPRINT_PRESERVING_ORIENTS = ("R0", "MX", "MY", "R180")

# Timeout ceilings (send_command returns as soon as the command finishes; these
# are headroom for large designs, not expected latencies).
T_FAST = 1800
T_DETAIL = 28800   # 8h: detail does TWO global_placement passes; large designs
                   # (ariane81 154k cells) measured ~3.1h. Must exceed the real
                   # runtime or the source() times out mid-run and the follow-up
                   # write_db desyncs (ORD-0007 -> route "no network linked").
T_ROUTE = 28800
T_CONGEST = 5400


class FlowOps:
    def __init__(self, ctx: FlowContext):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.s = ctx.session
        self.scripts: dict[str, str] = {}

    @property
    def st(self):
        return self.ctx.state

    # ==================================================================
    # Init
    # ==================================================================
    def init_design(self) -> dict:
        """Render the stage templates, source init, load the place proc, and
        collect the design data. Leaves the design floorplanned."""
        cfg, sk = self.cfg, self.ctx.skills
        self.scripts["init"] = str(cfg.render(sk.template_path("init")))
        self.scripts["detail_place"] = str(cfg.render(sk.template_path("detail_place")))
        post_route_template = sk.template_path("post_route")
        if cfg.route_mode_effective == "timing" \
                or not post_route_template.exists():
            # timing mode, or the light template is missing — fall back to
            # the full cts_route.tcl instead of failing every candidate
            post_route_template = post_route_template.with_name("cts_route.tcl")
        self.scripts["post_route"] = str(cfg.render(post_route_template))
        # The fast-place Gate-B proxy stage was removed from the workflow
        # (the new flow checks PPA/timing after route, not a pre-route density
        # proxy). No fast_place template is rendered.

        self.ctx.note(f"[init] sourcing {self.scripts['init']}")
        self.s.source(self.scripts["init"], timeout=300)
        if self.cfg.validate_inputs and not self.cfg.floorplan_odb.exists():
            raise RuntimeError(
                "OpenROAD init did not create floorplan.odb. Check the init "
                "log for missing input files or TCL errors: "
                f"{self.s.log_path}"
            )
        self.s.set_init_script(self.scripts["init"])
        self.s.send_command(R.SAFE_PLACE_PROC, timeout=10)
        self.refresh_design_data("init")
        self.ctx.snapshot("init", "00_initial_floorplan.png", "initial floorplan")
        self.st.stage = "init"
        self.ctx.note(f"[init] {self.st.n_macros} macros, core={self._core_str()}")
        return self.design_state_summary()

    def refresh_design_data(self, tag: str) -> int:
        st = self.st
        st.dims_text = self.s.run_recipe(R.DIE_CORE, timeout=30)
        st.macros_text = self.s.run_recipe(R.MACRO_LIST, timeout=60)
        pin_box_text = self.s.run_recipe(R.PIN_COORDS, timeout=60)
        st.die = geo.parse_die_bounds(st.dims_text)
        st.core = geo.parse_core_bounds(st.dims_text)
        st.pin_boxes = geo.parse_pin_boxes(pin_box_text)
        st.macro_sizes = geo.parse_macro_sizes(st.macros_text)
        st.mw, st.mh = geo.macro_footprint(st.macro_sizes)
        st.macro_index = geo.parse_macro_index(st.macros_text)
        st.regions = geo.region_grid(st.core, n=4)
        st.n_macros = st.macros_text.count("master=")
        mo = self.cfg.stage_model_dir("design_data", tag)
        (mo / "dims.txt").write_text(st.dims_text)
        (mo / "macros.txt").write_text(st.macros_text)
        (mo / "pin_boxes_summary.txt").write_text(
            f"pin_box_count: {len(st.pin_boxes)}\n"
            "raw pin-coordinate text omitted; pin locations are provided to the "
            "model visually through layout snapshots.\n")
        return st.n_macros

    def highlight_macro_pin_sides(self, tag: str | None = None) -> dict:
        """Draw ruler highlights on each macro's dominant signal-pin side.

        Gate A uses this annotated image so the Checker and Placer can reason
        about orientation, not just macro outline. The text summary is also kept
        in state because some headless OpenROAD builds may not render rulers.
        """
        st = self.st
        tag = tag or st.try_tag or "gate_a"
        out = self.s.run_recipe(R.MACRO_PIN_SIDES, timeout=120)
        sides = self._parse_macro_pin_sides(out)
        st.macro_pin_sides = sides
        mo = self.cfg.stage_model_dir("design_data", "macro_pin_sides")
        (mo / f"{tag}.txt").write_text(out)
        snap = self.ctx.snapshot(
            "macro_pin_sides",
            f"{tag}_macro_pin_sides.png",
            "macro pin-side rulers for Gate A",
            macro_pin_sides=len(sides),
        )
        self._overlay_macro_pin_sides(snap.path)
        self.ctx.note(f"[pin-sides] highlighted {len(sides)}/{st.n_macros} macros")
        return {
            "macro_pin_sides": len(sides),
            "image": snap.path,
            "summary": self.macro_pin_side_context(max_lines=16),
        }

    @staticmethod
    def _parse_macro_pin_sides(text: str) -> dict:
        sides = {}
        pat = re.compile(
            r"MACRO_PIN_SIDE\s+name=(?P<name>\S+)\s+orient=(?P<orient>\S+)\s+"
            r"side=(?P<side>\S+)\s+dominant=(?P<dominant>\d+)\s+"
            r"counts=L:(?P<L>\d+),R:(?P<R>\d+),T:(?P<T>\d+),B:(?P<B>\d+)\s+"
            r"bbox=\((?P<bbox>[^)]*)\)")
        for line in (text or "").splitlines():
            m = pat.search(line)
            if not m:
                continue
            counts = {k: int(m.group(k)) for k in ("L", "R", "T", "B")}
            bbox = []
            for v in m.group("bbox").split(","):
                try:
                    bbox.append(float(v))
                except ValueError:
                    pass
            sides[m.group("name")] = {
                "orient": m.group("orient"),
                "side": m.group("side"),
                "dominant": int(m.group("dominant")),
                "counts": counts,
                "bbox": bbox,
            }
        return sides

    def _overlay_macro_pin_sides(self, path) -> None:
        """Draw visible pin-side bars on the saved PNG for headless OpenROAD.

        OpenROAD's gui::add_ruler is unavailable in non-GUI mode, but save_image
        still works. This reproduces the ruler intent directly on the raster
        image from the parsed macro bboxes.
        """
        if not self.st.macro_pin_sides:
            return
        try:
            from PIL import Image, ImageDraw
            img = Image.open(path).convert("RGBA")
        except Exception:
            return
        area = self.st.image_area(self.cfg.save_image_margin)
        if not area:
            return
        ax0, ay0, ax1, ay1 = area
        aw = max(1e-9, ax1 - ax0)
        ah = max(1e-9, ay1 - ay0)
        w_px, h_px = img.size

        def px(x: float) -> float:
            return (x - ax0) / aw * (w_px - 1)

        def py(y: float) -> float:
            return (ay1 - y) / ah * (h_px - 1)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for info in self.st.macro_pin_sides.values():
            bbox = info.get("bbox") or []
            side = info.get("side")
            if len(bbox) != 4 or side not in {"L", "R", "T", "B"}:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            left, right = px(x0), px(x1)
            top, bottom = py(y1), py(y0)
            color = (255, 64, 0, 230)
            shadow = (0, 0, 0, 180)
            if side == "L":
                p1, p2 = (left, top), (left, bottom)
            elif side == "R":
                p1, p2 = (right, top), (right, bottom)
            elif side == "T":
                p1, p2 = (left, top), (right, top)
            else:
                p1, p2 = (left, bottom), (right, bottom)
            draw.line([p1, p2], fill=shadow, width=7)
            draw.line([p1, p2], fill=color, width=4)
        try:
            Image.alpha_composite(img, overlay).convert("RGB").save(path)
        except Exception:
            pass

    # ==================================================================
    # Placement primitives
    # ==================================================================
    def reload_baseline(self):
        """Cheap in-iteration reset: re-read the pristine floorplan (no PDN yet,
        liberty stays linked) and reload the place proc."""
        self.s.read_db(self.cfg.floorplan_odb, timeout=120)
        self.s.send_command(R.SAFE_PLACE_PROC, timeout=10)

    def reload_clean(self):
        """Leak-proof per-try reset for the exploration loop. After a detail/route
        the OpenROAD process holds cut_rows / PDN stripes / a core voltage domain
        that ``read_db floorplan.odb`` does NOT clear; only a fresh process does.
        Kill it, re-source init (reloads LEF/lib/verilog/def + liberty + writes a
        pristine floorplan.odb) and reload the place proc. Liberty then stays
        linked through the in-process ``reload_baseline`` that placement does next."""
        self.ctx.note("[reload-clean] fresh process + re-source init")
        self.s.fresh_start()
        self.s.source(self.scripts["init"], timeout=300)
        self.s.set_init_script(self.scripts["init"])
        self.s.send_command(R.SAFE_PLACE_PROC, timeout=10)
        self.refresh_design_data("reinit")
        self.st.stage = "init"

    def unplace_std_cells(self):
        """Clear stdcell placement while preserving macro placements."""
        self.s.send_command(
            "set block [ord::get_db_block]\n"
            "foreach inst [$block getInsts] {\n"
            "    if {![[$inst getMaster] isBlock]} {\n"
            "        $inst setPlacementStatus NONE\n"
            "    }\n"
            "}\n",
            timeout=120)

    def clear_macro_placement_status(self):
        """Unlock all macros before issuing a fresh full macro placement."""
        self.s.send_command(
            "set block [ord::get_db_block]\n"
            "foreach inst [$block getInsts] {\n"
            "    if {[[$inst getMaster] isBlock]} {\n"
            "        $inst setPlacementStatus NONE\n"
            "    }\n"
            "}\n",
            timeout=120)

    # ------------------------------------------------------------------
    # Exploration placement sources (the candidate the loop will evaluate)
    # ------------------------------------------------------------------
    def load_entries(self, coords: list) -> dict:
        """Adopt a saved candidate's coordinates as the current placement (for
        fine refinement). Sets ``st.entries`` but does not place yet — the
        Refiner nudges then places."""
        st = self.st
        st.entries = [{"name": c["name"], "x": float(c["x"]), "y": float(c["y"]),
                       "orient": c.get("orient", "R0")} for c in coords]
        self.ctx.note(f"[load] adopted {len(st.entries)} macro coords from candidate")
        return {"loaded": len(st.entries)}

    def post_route_refine_variant(self, variant_index: int, total_variants: int,
                                  loop_index: int, tag: str | None = None) -> dict:
        """Create one deterministic post-route macro refinement variant.

        This starts from the current loaded best routed candidate and makes a
        small legal geometric move. Routing feedback is expensive, so diversity
        here comes from several local compaction anchors; the parent loop then
        keeps whichever routed candidate scores best.
        """
        st = self.st
        if not st.entries:
            raise RuntimeError("post-route refinement requires loaded macro coords")
        tag = tag or st.try_tag or f"post_r{loop_index:02d}_c{variant_index:02d}"
        cx0, cy0, cx1, cy1 = st.core
        anchors = [
            "center", "left", "right", "top", "bottom",
            "top-left", "top-right", "bottom-left", "bottom-right",
        ]
        # Keep post-route moves conservative: routed candidates are already close
        # to valid, and larger moves in this stage tend to create DRC-heavy routes.
        strengths = [1.0 / 96.0, 1.0 / 72.0, 1.0 / 48.0, 1.0 / 36.0]
        anchor = anchors[((loop_index - 1) * max(1, total_variants) + variant_index)
                         % len(anchors)]
        strength = strengths[min(max(0, loop_index - 1), len(strengths) - 1)]
        ents = geo.compact_entries([dict(e) for e in st.entries], strength, anchor,
                                   cx0, cy0, cx1, cy1)
        ents = geo.validate_entries(ents, cx0, cy0, cx1, cy1, mw=st.mw,
                                    mh=st.mh, macro_sizes=st.macro_sizes)
        ents = geo.legalize(ents, st.macro_sizes, cx0, cy0, cx1, cy1,
                            mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = ents
        st.best_entries = [dict(e) for e in ents]
        st.compaction_round += 1
        ov = self._place(ents, tag=tag)
        st.stage = "seeded"
        self.ctx.note("[post-route-refine] "
                      f"loop={loop_index} variant={variant_index + 1}/"
                      f"{total_variants} anchor={anchor} "
                      f"strength={strength:.4f} overlaps={ov}")
        return {
            "stage": st.stage,
            "loop": loop_index,
            "variant": variant_index,
            "total_variants": total_variants,
            "anchor": anchor,
            "strength": round(strength, 6),
            "overlaps": ov,
            "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                            stage="macro_placed")),
        }

    def nudge(self, plan_text: str, tag: str | None = None) -> dict:
        """Fine-grain move: apply ``#index -> (dx, dy[, orient])`` micron offsets
        onto the CURRENT entries, validate, legalize, place. Targeted, small moves
        (e.g. close a thin channel against the die edge)."""
        st = self.st
        tag = tag or st.try_tag or "fine"
        cx0, cy0, cx1, cy1 = st.core
        deltas = geo.parse_indexed_deltas(plan_text, len(st.macro_index))
        idx_by_name = {name: i for i, (name, _w, _h) in enumerate(st.macro_index)}
        ents = [dict(e) for e in st.entries]
        applied = 0
        for e in ents:
            i = idx_by_name.get(e["name"])
            if i is not None and i in deltas:
                dx, dy, orient = deltas[i]
                e["x"] += dx
                e["y"] += dy
                if orient:
                    e["orient"] = orient
                applied += 1
        ents = geo.validate_entries(ents, cx0, cy0, cx1, cy1, mw=st.mw,
                                    mh=st.mh, macro_sizes=st.macro_sizes)
        ents = geo.legalize(ents, st.macro_sizes, cx0, cy0, cx1, cy1,
                            mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = ents
        ov = self._place(ents, tag=tag)
        self.ctx.note(f"[nudge] applied {applied} move(s), overlaps after legalize={ov}")
        return {"applied": applied, "overlaps": ov,
                "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                stage="macro_placed"))}

    def snap_channels(self, tag: str | None = None, min_channel: float = 5.0) -> dict:
        """Deterministic fine fallback: flush any macro that leaves a thin sliver
        against a core edge, then legalize + place."""
        st = self.st
        tag = tag or st.try_tag or "fine"
        cx0, cy0, cx1, cy1 = st.core
        ents = [dict(e) for e in st.entries]
        ents = geo.snap_to_edges(ents, st.core, st.macro_sizes, min_channel=min_channel,
                                 mw=st.mw, mh=st.mh)
        ents = geo.legalize(ents, st.macro_sizes, cx0, cy0, cx1, cy1,
                            mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = ents
        ov = self._place(ents, tag=tag)
        self.ctx.note(f"[snap] flushed sub-{min_channel}um edge channels, overlaps={ov}")
        return {"overlaps": ov, "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                                stage="macro_placed"))}

    def place_shape(self, name: str, tag: str | None = None) -> dict:
        """Coarse-grain: place macros into a preset OUTLINE (ring / C / U / corner
        / split) via the region-distribution machinery, from a fresh ring base.
        Guarantees a genuinely different macro shape per coarse try."""
        st = self.st
        dist = geo.shape_distribution(name, st.n_macros, st.regions)
        if not dist:
            self.ctx.note(f"[shape:{name}] degenerate grid — falling back to ring seed")
            r = self.seed_macros()
            r["shape"] = "ring(seed)"
            return r
        r = self.place_in_regions(dist, orient="R0", base="ring", fast=False)
        r["shape"] = name
        self.ctx.note(f"[shape:{name}] distribution {dist}")
        return r

    def _place(self, entries: list[dict], tag: str) -> int:
        """Reload baseline + place all macros in size-sorted batches (largest
        first). Saves per-batch images to disk (for manual inspection) and one
        registered ``macro_placed`` milestone. Returns the overlap count."""
        if self.st.stage == "fast_placed":
            self.ctx.note(f"[place:{tag}] unplacing standard cells before macro "
                          "refinement (keep current macro state)")
            self.unplace_std_cells()
            self.ctx.note(f"[place:{tag}] clearing macro placement status before "
                          "full coordinate replacement")
            self.clear_macro_placement_status()
        elif self.st.stage in {"detailed", "routed"}:
            self.ctx.note(f"[place:{tag}] fresh reload before macro placement "
                          f"(discard previous {self.st.stage} state)")
            self.reload_clean()
        else:
            self.ctx.note(f"[place:{tag}] reloading floorplan.odb before macro "
                          "placement (discard previous stdcell placement)")
            self.reload_baseline()
        geo.assign_areas(entries, self.st.macro_sizes, self.st.mw, self.st.mh)
        if any("cluster_order" in e for e in entries):
            ents = sorted(entries, key=lambda e: (
                e.get("cluster_order", 10**9),
                -e.get("area", 0.0),
                e.get("name", ""),
            ))
        else:
            ents = sorted(entries, key=lambda e: e.get("area", 0.0), reverse=True)
        total = len(ents)
        nb = math.ceil(total / BATCH) if total else 0
        batch_dir = self.cfg.stage_image_dir("macro_placed", "batches")
        for b in range(nb):
            batch = ents[b * BATCH:(b + 1) * BATCH]
            self.s.save_image(batch_dir / f"{tag}_batch{b + 1:02d}_before.png",
                              area=self.st.image_area(self.cfg.save_image_margin))
            # Brace-quote names so escaped bus brackets reach findInst verbatim.
            lines = [f'safe_place {{{e["name"]}}} {e["x"]} {e["y"]} {e["orient"]}'
                     for e in batch]
            self.s.send_command("\n".join(lines), timeout=300)
            self.s.save_image(batch_dir / f"{tag}_batch{b + 1:02d}_after.png",
                              area=self.st.image_area(self.cfg.save_image_margin))
        ov = M.parse_overlaps(self.s.run_recipe(R.CHECK_OVERLAPS, timeout=60))
        self.s.run_recipe(R.LOCK_MACROS_FIRM, timeout=60)
        self.ctx.snapshot("macro_placed", f"{tag}_macro_placed.png",
                          f"{total} macros placed ({tag})",
                          overlaps=(ov if ov is not None else "?"))
        macro_db = self.cfg.save_dir / f"{tag}_macro_placed.odb"
        self.s.write_db(macro_db, timeout=60)
        self.st.last_macro_db = str(macro_db)
        self.ctx.update_view(macro_db)     # --gui: relaunch the live window
        return ov if ov is not None else -1

    def _fast(self, tag: str) -> float | None:
        """Fast global+detailed placement on the CURRENT session (no reload),
        register the fast-placed image, return stdcell density."""
        self._ensure_macro_overlap_free(tag)
        t0 = time.time()
        out_odb = self.cfg.save_dir / f"{tag}_fast_placed.odb"
        self.s.source(self.scripts["fast_place"], timeout=T_FAST)
        snap = self.ctx.snapshot("fast_placed", f"{tag}_fast_placed.png",
                                 f"fast-placed ({tag})")
        density = self.measure_density()
        if density is not None:
            snap.meta["density"] = round(density, 4)
        self.s.write_db(out_odb, timeout=60)
        self.st.last_fast_db = str(out_odb)
        self.ctx.update_view(out_odb)      # --gui: relaunch the live window at this stage
        self.ctx.note(f"[fast_place] {tag}: density={density} ({time.time() - t0:.0f}s)")
        return density

    def _ensure_macro_overlap_free(self, tag: str, max_rounds: int = 2) -> None:
        """Do not enter fast/detail placement with overlapping OpenROAD macro bboxes."""
        ov = self.check_overlaps()
        outside = self.check_outside_core()
        if (ov is None or ov <= 0) and (outside is None or outside <= 0):
            return
        self.ctx.note(f"[overlap-guard] {ov or 0} macro overlap(s), "
                      f"{outside or 0} outside-core macro(s) before fast_place; "
                      "repairing from OpenROAD readback")
        for round_idx in range(1, max_rounds + 1):
            self.refresh_design_data(f"{tag}_overlap_repair{round_idx}")
            ents = geo.parse_macro_positions(self.st.macros_text)
            if not ents:
                break
            cx0, cy0, cx1, cy1 = self.st.core
            ents = geo.validate_entries(ents, cx0, cy0, cx1, cy1, mw=self.st.mw,
                                        mh=self.st.mh,
                                        macro_sizes=self.st.macro_sizes)
            ents = geo.legalize(ents, self.st.macro_sizes, cx0, cy0, cx1, cy1,
                                mw=self.st.mw, mh=self.st.mh, gap=2.0)
            self.st.entries = ents
            ov = self._place(ents, tag=f"{tag}_overlap_fix{round_idx}")
            outside = self.check_outside_core()
            if (ov is None or ov <= 0) and (outside is None or outside <= 0):
                self.ctx.note("[overlap-guard] macro placement is overlap-free")
                return
        raise RuntimeError(
            f"macro placement remains illegal before fast_place: "
            f"overlaps={ov}, outside_core={outside}")

    def _base_tag(self) -> str:
        return self.st.try_tag or "iter00"

    def _macro_cluster_file(self) -> Path | None:
        # Canonical location is clusters/ (Get_claude_results.md:
        # CLUSTER = <project>/clusters/<design>_macro_clusters.txt). configs/ is
        # kept as a legacy fallback. Without this file the Placer cannot use the
        # cluster-aware deterministic placement and falls back to raw VLM
        # coordinates, which for a weak model collapses all macros onto one point.
        name = f"{self.cfg.design}_macro_clusters.txt"
        for sub in ("clusters", "configs"):
            path = self.cfg.project_root / sub / name
            if path.exists():
                return path
        return None

    def has_macro_cluster_file(self) -> bool:
        return self._macro_cluster_file() is not None

    # ==================================================================
    # Stage operations the agents invoke
    # ==================================================================
    def seed_macros(self) -> dict:
        """Deterministic perimeter-ring seed + legalize, then place. The Planner
        decides strategy separately; the seed itself is overlap-free by
        construction."""
        st = self.st
        entries, core, sizes, mw, mh = geo.build_ring_seed(
            st.macros_text, st.dims_text, gap=5.0,
            cluster_file=self._macro_cluster_file())
        st.entries = entries
        st.core, st.macro_sizes, st.mw, st.mh = core, sizes, mw, mh
        tag = self._base_tag()
        ov = self._place(entries, tag=tag)
        st.stage = "seeded"
        self.ctx.note(f"[seed] {len(entries)} macros to perimeter ring, overlaps={ov}")
        return {"stage": st.stage, "n_macros": len(entries), "overlaps": ov,
                "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                stage="macro_placed"))}

    def place_strategy(self, strategy: str) -> dict:
        """Expand Planner side/C-shape intent into direct legal coordinates."""
        st = self.st
        refining = bool(st.entries)
        cluster_file = self._macro_cluster_file()
        entries, core, sizes, mw, mh, sides = geo.build_strategy_seed(
            st.macros_text, st.dims_text, strategy, gap=5.0,
            cluster_file=cluster_file)
        st.entries = entries
        st.core, st.macro_sizes, st.mw, st.mh = core, sizes, mw, mh
        if refining:
            st.compaction_round += 1
        base_tag = self._base_tag()
        tag = base_tag if not refining else f"{base_tag}_r{st.compaction_round}"
        if self.st.macro_pin_sides:
            total_oriented = 0
            for _pass in range(3):
                oriented = self._orient_entries_away_from_edges(entries)
                total_oriented += oriented
                if not oriented and _pass > 0:
                    break
                entries = geo.validate_entries(entries, st.core[0], st.core[1],
                                               st.core[2], st.core[3], mw=st.mw,
                                               mh=st.mh, macro_sizes=st.macro_sizes)
                entries = geo.legalize(entries, st.macro_sizes, st.core[0],
                                       st.core[1], st.core[2], st.core[3],
                                       mw=st.mw, mh=st.mh, gap=2.0)
                st.entries = entries
            if total_oriented:
                self.ctx.note("[pin-orient] completed iterative orientation/"
                              "legalization to avoid outward-facing pin sides")
        ov = self._place(entries, tag=tag)
        outside = self.check_outside_core()
        st.stage = "seeded"
        cluster_note = f", clusters={cluster_file.name}" if cluster_file else ""
        self.ctx.note(f"[place_strategy] direct side-band coordinates sides={sides}"
                      f"{cluster_note}, "
                      f"overlaps={ov}, outside_core={outside}")
        return {"stage": st.stage, "n_macros": len(entries), "overlaps": ov,
                "outside_core": outside, "sides": sides,
                "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                stage="macro_placed"))}

    @staticmethod
    def _side_after_orient(base_side: str, orient: str) -> str | None:
        return ORIENT_SIDE_MAP.get(orient, ORIENT_SIDE_MAP["R0"]).get(base_side)

    @classmethod
    def _base_side_from_current(cls, current_side: str, orient: str) -> str | None:
        if current_side not in SIDES:
            return None
        for base in SIDES:
            if cls._side_after_orient(base, orient) == current_side:
                return base
        return current_side

    @classmethod
    def _orient_for_target_side(cls, base_side: str, target_side: str,
                                preserve_footprint: bool = False) -> str | None:
        if base_side not in SIDES or target_side not in SIDES:
            return "R0"
        orientations = (FOOTPRINT_PRESERVING_ORIENTS if preserve_footprint
                        else PREFERRED_ORIENTS)
        for orient in orientations:
            if cls._side_after_orient(base_side, orient) == target_side:
                return orient
        return None

    def _macro_center(self, e: dict) -> tuple[float, float]:
        w, h = self.st.macro_sizes.get(e["name"], (self.st.mw, self.st.mh))
        ow, oh = geo.oriented_size(w, h, e.get("orient", "R0"))
        return float(e["x"]) + ow / 2.0, float(e["y"]) + oh / 2.0

    def _outward_side(self, e: dict) -> str:
        """Nearest core edge side for a macro center."""
        cx0, cy0, cx1, cy1 = self.st.core
        mx, my = self._macro_center(e)
        distances = {
            "L": abs(mx - cx0),
            "R": abs(cx1 - mx),
            "B": abs(my - cy0),
            "T": abs(cy1 - my),
        }
        return min(distances, key=distances.get)

    def _orient_entries_away_from_edges(self, entries: list[dict]) -> int:
        """Use Gate-A pin-side detection to avoid outward-facing macro pins.

        The model is unreliable at fine pairwise pin-facing decisions. Keep the
        deterministic repair simple: if a macro's highlighted pin side points
        toward the nearest core edge, flip/rotate with footprint-preserving
        orientations so that side no longer faces outward.
        """
        if not self.st.macro_pin_sides or not entries:
            return 0
        changed = 0
        for e in entries:
            info = self.st.macro_pin_sides.get(e["name"])
            if not info:
                continue
            outward = self._outward_side(e)
            current = info.get("side")
            if current != outward:
                continue
            base = self._base_side_from_current(info.get("side", ""),
                                                info.get("orient", "R0"))
            target = OPPOSITE_SIDE[outward]
            orient = self._orient_for_target_side(base or "L", target,
                                                  preserve_footprint=True)
            if orient is None:
                for candidate in FOOTPRINT_PRESERVING_ORIENTS:
                    side = self._side_after_orient(base or "L", candidate)
                    if side != outward:
                        orient = candidate
                        break
            if orient is None:
                continue
            if orient != e.get("orient", "R0"):
                e["orient"] = orient
                changed += 1
        if changed:
            self.ctx.note(f"[pin-orient] oriented {changed}/{len(entries)} macros "
                          "so highlighted pin sides do not face outward")
        return changed

    def refine_large_feedback(self, feedback: str, strategy: str = "",
                              fallback_anchor: str | None = None,
                              fallback_strength: float | None = None) -> dict:
        """Apply Gate-B directional feedback for large macro-count designs.

        The large-design path avoids asking the Placer LLM for hundreds of
        coordinates. Instead, parse feedback such as "shift the right-side macro
        group left by 1/12 core width", move the matching macro subset, legalize,
        and re-place. Coarse requested moves are capped before execution. If no
        directional move is parseable, fall back to
        re-expanding the Planner strategy.
        """
        st = self.st
        if not st.entries:
            return self.place_strategy(strategy)

        cx0, cy0, cx1, cy1 = st.core
        width = cx1 - cx0
        height = cy1 - cy0
        text = (feedback or "").lower()
        requested_frac = self._feedback_fraction(text)
        frac = min(requested_frac, 1.0 / 8.0)
        if frac < requested_frac:
            self.ctx.note("[large-feedback] damping Gate-B move fraction "
                          f"{requested_frac:.3f} -> {frac:.3f} to avoid "
                          "creating a new large dead pocket")
        dx, dy = self._feedback_direction(text, width, height, frac)
        if dx == 0.0 and dy == 0.0 and "inward" in text:
            return self.refine_geometric(min(0.6, frac), "center")
        if dx == 0.0 and dy == 0.0:
            if fallback_anchor:
                try:
                    strength = float(fallback_strength)
                except (TypeError, ValueError):
                    strength = frac
                self.ctx.note("[large-feedback] no parseable shift direction; "
                              f"using Checker fallback compaction "
                              f"{strength:.2f}/{fallback_anchor}")
                return self.refine_geometric(strength, fallback_anchor)
            self.ctx.note("[large-feedback] no parseable shift direction; "
                          "re-expanding Planner strategy")
            return self.place_strategy(strategy)

        selected = self._feedback_selected_entries(text, st.entries)
        if not selected:
            selected = list(range(len(st.entries)))
        selected = self._expand_selection_to_clusters(selected, st.entries)
        ents = [dict(e) for e in st.entries]
        for i in selected:
            ents[i]["x"] += dx
            ents[i]["y"] += dy
        ents = geo.validate_entries(ents, cx0, cy0, cx1, cy1, mw=st.mw, mh=st.mh,
                                    macro_sizes=st.macro_sizes)
        ents = geo.legalize(ents, st.macro_sizes, cx0, cy0, cx1, cy1,
                            mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = ents
        st.compaction_round += 1
        tag = f"{self._base_tag()}_r{st.compaction_round}"
        ov = self._place(ents, tag=tag)
        outside = self.check_outside_core()
        st.stage = "seeded"
        self.ctx.note("[large-feedback] shifted "
                      f"{len(selected)}/{len(ents)} macros by "
                      f"dx={dx:.1f} dy={dy:.1f} from Gate-B feedback; "
                      f"overlaps={ov}, outside_core={outside}")
        return {"stage": st.stage, "n_macros": len(ents), "moved": len(selected),
                "dx": round(dx, 3), "dy": round(dy, 3), "overlaps": ov,
                "outside_core": outside,
                "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                stage="macro_placed"))}

    @staticmethod
    def _feedback_fraction(text: str) -> float:
        if re.search(r"\b1\s*/\s*16\b|one[- ]sixteenth|sixteenth", text):
            return 1.0 / 16.0
        if re.search(r"\b1\s*/\s*12\b|one[- ]twelfth|twelfth", text):
            return 1.0 / 12.0
        if re.search(r"\b1\s*/\s*8\b|one[- ]eighth|eighth", text):
            return 1.0 / 8.0
        if re.search(r"\b1\s*/\s*6\b|one[- ]sixth|sixth", text):
            return 1.0 / 6.0
        if re.search(r"\b1\s*/\s*4\b|one[- ]quarter|quarter", text):
            return 1.0 / 4.0
        return 1.0 / 12.0

    @staticmethod
    def _feedback_direction(text: str, width: float, height: float,
                            frac: float) -> tuple[float, float]:
        verb = r"(?:shift|move|pull|push|nudge|bring)"
        m = re.search(rf"\b{verb}\b(?P<clause>[^.:\n;]*)", text)
        clause = m.group("clause") if m else text
        dirs = re.findall(r"\b(left|west|right|east|down|lower|south|up|raise|north)\b",
                          clause)
        dx = dy = 0.0
        last_horizontal = next(
            (d for d in reversed(dirs) if d in ("left", "west", "right", "east")),
            "")
        last_vertical = next(
            (d for d in reversed(dirs)
             if d in ("down", "lower", "south", "up", "raise", "north")),
            "")
        if last_horizontal in ("left", "west"):
            dx = -frac * width
        elif last_horizontal in ("right", "east"):
            dx = frac * width
        if last_vertical in ("down", "lower", "south"):
            dy = -frac * height
        elif last_vertical in ("up", "raise", "north"):
            dy = frac * height
        return dx, dy

    def _feedback_selected_entries(self, text: str, entries: list[dict]) -> list[int]:
        cx0, cy0, cx1, cy1 = self.st.core
        width = cx1 - cx0
        height = cy1 - cy0
        band_x = 0.25 * width
        band_y = 0.25 * height
        wants = {
            "right": self._feedback_mentions_side(text, "right"),
            "left": self._feedback_mentions_side(text, "left"),
            "top": self._feedback_mentions_side(text, "top"),
            "bottom": self._feedback_mentions_side(text, "bottom"),
        }
        if not any(wants.values()):
            return []
        selected = []
        for i, e in enumerate(entries):
            w, h = self.st.macro_sizes.get(e["name"], (self.st.mw, self.st.mh))
            mx = float(e["x"]) + w / 2.0
            my = float(e["y"]) + h / 2.0
            want = True
            if wants["right"]:
                want = want and mx >= cx1 - band_x
            if wants["left"]:
                want = want and mx <= cx0 + band_x
            if wants["top"]:
                want = want and my >= cy1 - band_y
            if wants["bottom"]:
                want = want and my <= cy0 + band_y
            if want:
                selected.append(i)
        return selected

    @staticmethod
    def _feedback_mentions_side(text: str, side: str) -> bool:
        """Return true only when a feedback phrase names the macro group side.

        Direction words should not be treated as selection words. For example,
        "shift the left-side group right" selects the left group; "shift the
        top row down and right" selects the top row, not the right-side macros.
        """
        noun = r"(?:side|hand|macro|macros|group|cluster|row|column|bank|blocks?)"
        if re.search(rf"\b{side}[- ]{noun}\b", text):
            return True
        if re.search(rf"\b{side}\s+(?:macro|macros|group|cluster|row|column|bank|blocks?)\b",
                     text):
            return True
        if side in ("top", "bottom"):
            return bool(re.search(rf"\b{side}[- ](?:left|right)\b", text))
        if side in ("left", "right"):
            return bool(re.search(rf"\b(?:top|bottom)[- ]{side}\b", text))
        return False

    @staticmethod
    def _expand_selection_to_clusters(selected: list[int],
                                      entries: list[dict]) -> list[int]:
        """Move whole macro clusters during feedback refinement.

        Geometry-based feedback such as "shift the right-side group left" may
        select only part of a dataflow cluster. If cluster metadata is present,
        expand that selection so same-cluster macros stay close after Gate B
        refinement.
        """
        selected_set = set(selected)
        selected_clusters = {
            entries[i].get("cluster_order")
            for i in selected_set
            if i < len(entries) and entries[i].get("cluster_order") is not None
        }
        if not selected_clusters:
            return selected
        return [
            i for i, e in enumerate(entries)
            if i in selected_set or e.get("cluster_order") in selected_clusters
        ]

    # ------------------------------------------------------------------
    # LLM-coordinate placement (the Planner generates the plan)
    # ------------------------------------------------------------------
    def planner_context(self) -> str:
        """Core bounds + numbered macro list for direct coordinate planning."""
        st = self.st
        lines = [f"Core: x[{st.core[0]:.1f}..{st.core[2]:.1f}] "
                 f"y[{st.core[1]:.1f}..{st.core[3]:.1f}] um (origin bottom-left).",
                 "Coordinates are macro lower-left origins in microns and must stay "
                 "inside the core. For each macro, use x/y within the legal ranges "
                 "shown below; if using R90/R270/MXR90/MYR90, account for swapped "
                 "width/height.",
                 "The legal range endpoint is a boundary, not a preferred coordinate. "
                 "Do not place many macros at the same max x/y. In one column, "
                 "separate macro origins by at least that macro's bbox height + 2um; "
                 "in one row, separate origins by at least bbox width + 2um.",
                 "",
                 f"Macros to place ({st.n_macros}) — reference each by #index:"]
        cx0, cy0, cx1, cy1 = st.core
        for i, (name, w, h) in enumerate(st.macro_index):
            short = name.split("/")[-1][:34]
            x_max = max(cx0, cx1 - w)
            y_max = max(cy0, cy1 - h)
            rx_max = max(cx0, cx1 - h)
            ry_max = max(cy0, cy1 - w)
            lines.append(f"  #{i}  {w:.0f}x{h:.0f}um  {short}  "
                         f"R0 legal x[{cx0:.1f}..{x_max:.1f}] "
                         f"y[{cy0:.1f}..{y_max:.1f}]; rotated legal "
                         f"x[{cx0:.1f}..{rx_max:.1f}] y[{cy0:.1f}..{ry_max:.1f}]")
        return "\n".join(lines)

    def refine_context(self) -> str:
        """Numbered macro list WITH current positions — the canvas the Refiner
        reasons over to emit small ``#index -> (dx, dy)`` moves."""
        st = self.st
        pos_by_name = {e["name"]: e for e in st.entries}
        lines = [f"Core: x[{st.core[0]:.1f}..{st.core[2]:.1f}] "
                 f"y[{st.core[1]:.1f}..{st.core[3]:.1f}] um (origin bottom-left).",
                 "", f"Current macro placement ({st.n_macros}) — #index, size, position:"]
        for i, (name, w, h) in enumerate(st.macro_index):
            short = name.split("/")[-1][:34]
            e = pos_by_name.get(name)
            if e:
                lines.append(f"  #{i}  {w:.0f}x{h:.0f}um  {short}  "
                             f"at ({e['x']:.1f}, {e['y']:.1f}) {e.get('orient', 'R0')}")
            else:
                lines.append(f"  #{i}  {w:.0f}x{h:.0f}um  {short}  (position unknown)")
        return "\n".join(lines)

    def placement_refine_context(self) -> str:
        """Gate-A Placer retry context: current macro coordinates/sizes/names,
        so the Placer can fix the implementation
        without asking the Planner for a new strategy."""
        st = self.st
        lines = [f"Core: x[{st.core[0]:.1f}..{st.core[2]:.1f}] "
                 f"y[{st.core[1]:.1f}..{st.core[3]:.1f}] um (origin bottom-left).",
                 "",
                 "Current macro placement — #index, size, name, position:"]
        pos_by_name = {e["name"]: e for e in st.entries}
        for i, (name, w, h) in enumerate(st.macro_index):
            short = name.split("/")[-1][:34]
            e = pos_by_name.get(name)
            if e:
                lines.append(f"  #{i}  {w:.0f}x{h:.0f}um  {short}  "
                             f"at ({e['x']:.1f}, {e['y']:.1f}) {e.get('orient', 'R0')}")
            else:
                lines.append(f"  #{i}  {w:.0f}x{h:.0f}um  {short}  (position unknown)")
        pin_ctx = self.macro_pin_side_context()
        if pin_ctx:
            lines += ["", pin_ctx]
        return "\n".join(lines)

    def macro_pin_side_context(self, max_lines: int = 80) -> str:
        """Current dominant macro pin sides for Gate-A orientation checks."""
        st = self.st
        if not st.macro_pin_sides:
            return ""
        lines = [
            "Detected macro pin sides from current placed orientation "
            "(ruler-highlighted in the Gate-A image):"
        ]
        idx_by_name = {name: i for i, (name, _w, _h) in enumerate(st.macro_index)}
        ordered = sorted(st.macro_pin_sides.items(),
                         key=lambda kv: idx_by_name.get(kv[0], 10**9))
        for name, info in ordered[:max_lines]:
            i = idx_by_name.get(name, "?")
            short = name.split("/")[-1][:34]
            counts = info.get("counts") or {}
            lines.append(
                f"  #{i} {short}: pin_side={info.get('side')} "
                f"orient={info.get('orient')} dominant={info.get('dominant')} "
                f"counts L/R/T/B={counts.get('L', 0)}/{counts.get('R', 0)}/"
                f"{counts.get('T', 0)}/{counts.get('B', 0)}")
        remaining = len(ordered) - max_lines
        if remaining > 0:
            lines.append(f"  ... {remaining} more macros omitted")
        return "\n".join(lines)

    def macro_pin_outward_context(self, max_lines: int = 24) -> str:
        """Deterministic Gate-A check that macro pin sides do not face outward."""
        st = self.st
        if not st.entries or not st.macro_pin_sides:
            return ""
        idx_by_name = {name: i for i, (name, _w, _h) in enumerate(st.macro_index)}
        warnings = []
        ok = 0
        for e in st.entries:
            info = st.macro_pin_sides.get(e["name"])
            if not info:
                continue
            outward = self._outward_side(e)
            got = info.get("side")
            if got != outward:
                ok += 1
                continue
            n = idx_by_name.get(e["name"], "?")
            short = e["name"].split("/")[-1][:34]
            warnings.append(
                f"  #{n} {short}: pin_side={got} faces outward toward the "
                f"{outward} core edge; rotate/flip so the highlighted pin side "
                "faces inward or sideways while preserving row/column alignment")
        if not warnings:
            return f"Macro pin-side outward check: PASS ({ok} macro(s) checked)."
        lines = [
            "Macro pin-side outward check: FAIL. Gate A only requires that "
            "highlighted macro pin sides do not face outward toward the nearest "
            "core edge. Do not require pin-side-to-pin-side facing between "
            "neighboring macros. Repair by flipping/rotating the listed macros "
            "while preserving row/column alignment."
        ]
        lines.extend(warnings[:max_lines])
        remaining = len(warnings) - max_lines
        if remaining > 0:
            lines.append(f"  ... {remaining} more outward pin-side issue(s) omitted")
        return "\n".join(lines)

    def placement_region_summary(self) -> str:
        """Compact current macro coordinate summary for Gate-A checking."""
        st = self.st
        if not st.entries:
            return "(no current macro placement available)"
        centers = []
        for e in st.entries:
            w, h = st.macro_sizes.get(e["name"], (st.mw, st.mh))
            cx = float(e["x"]) + float(w) / 2.0
            cy = float(e["y"]) + float(h) / 2.0
            centers.append((cx, cy))
        if not centers:
            return "(no macro centers available)"
        min_x = min(x for x, _y in centers)
        max_x = max(x for x, _y in centers)
        min_y = min(y for _x, y in centers)
        max_y = max(y for _x, y in centers)
        cx0, cy0, cx1, cy1 = st.core
        width = max(1e-9, cx1 - cx0)
        height = max(1e-9, cy1 - cy0)
        edge_band_x = width * 0.15
        edge_band_y = height * 0.15
        left = sum(1 for x, _y in centers if x <= cx0 + edge_band_x)
        right = sum(1 for x, _y in centers if x >= cx1 - edge_band_x)
        bottom = sum(1 for _x, y in centers if y <= cy0 + edge_band_y)
        top = sum(1 for _x, y in centers if y >= cy1 - edge_band_y)
        lines = [
            f"Macro center bounding box: x[{min_x:.1f}..{max_x:.1f}] "
            f"y[{min_y:.1f}..{max_y:.1f}]",
            f"Macros near edges (15% core band): left={left}, right={right}, "
            f"bottom={bottom}, top={top}",
            f"Macro count summarized: {len(centers)}",
        ]
        return "\n".join(lines)

    def place_in_regions(self, assignments: dict, orient: str = "R0",
                         base: str = "ring", fast: bool = False) -> dict:
        """Region-distribution placement (scales to many / identical macros).

        ``assignments`` maps region label -> macro count (e.g. {"r0c3": 17, ...});
        Python assigns macros sequentially to the listed regions and packs each
        region's slice in a grid. The LLM chooses the distribution (which regions,
        how many each); Python computes the exact coordinates + legalizes.
        """
        st = self.st
        if base == "current" and st.entries:
            entries = [dict(e) for e in st.entries]
        else:
            entries, core, sizes, mw, mh = geo.build_ring_seed(
                st.macros_text, st.dims_text, gap=5.0,
                cluster_file=self._macro_cluster_file())
            st.core, st.macro_sizes, st.mw, st.mh = core, sizes, mw, mh
        cx0, cy0, cx1, cy1 = st.core
        region_by_label = {c["label"]: c for c in st.regions}
        names = [m[0] for m in st.macro_index]

        pos_by_index: dict[int, tuple[float, float]] = {}
        cursor = 0
        for label, count in assignments.items():
            region = region_by_label.get(str(label))
            try:
                count = int(count)
            except (TypeError, ValueError):
                continue
            if region is None or count <= 0:
                continue
            pts = geo.pack_region(region, count, st.mw, st.mh, gap=2.0)
            for k in range(count):
                if cursor >= len(names):
                    break
                pos_by_index[cursor] = pts[k]
                cursor += 1

        idx_by_name = {n: i for i, n in enumerate(names)}
        applied = 0
        for e in entries:
            i = idx_by_name.get(e["name"])
            if i is not None and i in pos_by_index:
                e["x"], e["y"] = pos_by_index[i]
                e["orient"] = orient
                applied += 1

        entries = geo.validate_entries(entries, cx0, cy0, cx1, cy1, mw=st.mw,
                                       mh=st.mh, macro_sizes=st.macro_sizes)
        entries = geo.legalize(entries, st.macro_sizes, cx0, cy0, cx1, cy1,
                               mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = entries
        st.compaction_round += 1 if base == "current" else 0
        base_tag = self._base_tag()
        tag = base_tag if base != "current" else f"{base_tag}_r{st.compaction_round}"
        ov = self._place(entries, tag=tag)
        st.stage = "seeded"
        self.ctx.note(f"[place_regions] distributed {applied}/{st.n_macros} macros "
                      f"over {len(assignments)} regions (base={base}), overlaps={ov}")
        result = {"stage": st.stage, "n_macros": st.n_macros,
                  "macros_distributed": applied, "overlaps": ov,
                  "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                  stage="macro_placed"))}
        if fast:
            density = self._fast(tag)
            st.last_density = density
            if density is not None and density > st.best_density:
                st.best_density = density
                st.best_entries = [dict(e) for e in entries]
            result["density"] = density
        return result

    def place_plan(self, plan_text: str, base: str = "ring", fast: bool = False) -> dict:
        """Apply the Planner's per-macro coordinates onto a complete base
        placement (ring seed for the first plan, current positions for a
        re-plan), then legalize and place. Unspecified macros keep their base
        position, so partial model output still yields a full legal placement."""
        st = self.st
        entries = self._plan_base_entries(base)
        cx0, cy0, cx1, cy1 = st.core

        overrides = geo.parse_indexed_plan(plan_text, len(st.macro_index))
        idx_by_name = {name: i for i, (name, _w, _h) in enumerate(st.macro_index)}
        applied = 0
        for e in entries:
            i = idx_by_name.get(e["name"])
            if i is not None and i in overrides:
                x, y, o = overrides[i]
                e["x"], e["y"], e["orient"] = x, y, o
                applied += 1

        entries = geo.validate_entries(entries, cx0, cy0, cx1, cy1, mw=st.mw,
                                       mh=st.mh, macro_sizes=st.macro_sizes)
        entries = geo.legalize(entries, st.macro_sizes, cx0, cy0, cx1, cy1,
                               mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = entries

        st.compaction_round += 1 if base == "current" else 0
        base_tag = self._base_tag()
        tag = base_tag if base != "current" else f"{base_tag}_r{st.compaction_round}"
        ov = self._place(entries, tag=tag)
        st.stage = "seeded"
        self.ctx.note(f"[place_plan] applied {applied}/{st.n_macros} LLM coords "
                      f"(base={base}), overlaps after legalize={ov}")
        result = {"stage": st.stage, "n_macros": st.n_macros,
                  "llm_coords_applied": applied, "overlaps": ov,
                  "macro_image": str(self.ctx.img(f"{tag}_macro_placed.png",
                                                  stage="macro_placed"))}
        if fast:
            density = self._fast(tag)
            st.last_density = density
            if density is not None and density > st.best_density:
                st.best_density = density
                st.best_entries = [dict(e) for e in entries]
            result["density"] = density
            result["fast_image"] = str(self.ctx.img(f"{tag}_fast_placed.png",
                                                    stage="fast_placed"))
        return result

    def _plan_base_entries(self, base: str) -> list[dict]:
        """Return mutable entries for a direct coordinate plan without placing."""
        st = self.st
        if base == "current" and st.entries:
            return [dict(e) for e in st.entries]
        entries, core, sizes, mw, mh = geo.build_ring_seed(
            st.macros_text, st.dims_text, gap=5.0, legalize_seed=False,
            cluster_file=self._macro_cluster_file())
        st.core, st.macro_sizes, st.mw, st.mh = core, sizes, mw, mh
        return entries

    def preview_plan_quality(self, plan_text: str, base: str = "ring") -> dict:
        """Validate a raw LLM coordinate plan before legalization/OpenROAD."""
        st = self.st
        entries = self._plan_base_entries(base)
        overrides = geo.parse_indexed_plan(plan_text, len(st.macro_index))
        idx_by_name = {name: i for i, (name, _w, _h) in enumerate(st.macro_index)}
        applied = 0
        for e in entries:
            i = idx_by_name.get(e["name"])
            if i is not None and i in overrides:
                x, y, o = overrides[i]
                e["x"], e["y"], e["orient"] = x, y, o
                applied += 1

        outside_tol = max(st.mw, st.mh)
        raw_stats = geo.placement_stats(entries, st.macro_sizes, st.core,
                                        st.mw, st.mh, gap=0.0,
                                        outside_tol=outside_tol)
        # PURE-MODEL PLACEMENT CONTRACT: the model owns every macro coordinate.
        # Geometry quality is NOT judged here — overlaps are removed by
        # ``legalize.py`` and out-of-core coordinates are clamped by
        # ``_clamp_to_core``, both of which run on every accepted plan. The only
        # thing this gate enforces is COVERAGE: a plan that leaves macros
        # unplaced would silently keep their base (non-model) positions, which is
        # exactly the contamination the pure-model run must not have. An
        # incomplete plan is therefore rejected, and if the Placer never produces
        # a complete one the candidate fails outright (see Specialists.place) —
        # no ring seed, no geometric repair, no partial placement.
        reject_reasons: list[str] = []
        if applied == 0:
            reject_reasons.append("no parseable #index coordinate lines")
        elif applied < st.n_macros:
            reject_reasons.append(
                f"incomplete plan: coordinates for only {applied} of "
                f"{st.n_macros} macros")

        return {
            "applied": applied,
            "n_macros": st.n_macros,
            "reject": bool(reject_reasons),
            "reasons": reject_reasons,
            **raw_stats,
        }

    def fast_place(self, tag: str = "base") -> dict:
        """Fast-place the CURRENT macro arrangement to get a stdcell density +
        image. ``tag`` distinguishes the baseline from per-round refinements."""
        base_tag = self._base_tag()
        fast_tag = f"{base_tag}_{tag}"
        density = self._fast(fast_tag)
        st = self.st
        st.last_density = density
        if density is not None and density > st.best_density:
            st.best_density = density
            st.best_entries = [dict(e) for e in st.entries]
        st.stage = "fast_placed"
        return {"stage": st.stage, "density": density,
                "fast_image": str(self.ctx.img(f"{fast_tag}_fast_placed.png",
                                               stage="fast_placed"))}

    def refine_geometric(self, strength: float, anchor: str) -> dict:
        """Deterministic compaction fallback (used when the Placer agent does not
        re-emit a plan): contract current macros toward an anchor + legalize +
        place (no fast-place — the orchestrator does that)."""
        st = self.st
        st.compaction_round += 1
        cx0, cy0, cx1, cy1 = st.core
        ents = geo.compact_entries([dict(e) for e in st.entries], strength, anchor,
                                   cx0, cy0, cx1, cy1)
        ents = geo.validate_entries(ents, cx0, cy0, cx1, cy1, mw=st.mw,
                                    mh=st.mh, macro_sizes=st.macro_sizes)
        ents = geo.legalize(ents, st.macro_sizes, cx0, cy0, cx1, cy1,
                            mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = ents
        self._place(ents, tag=f"{self._base_tag()}_r{st.compaction_round}")
        self.ctx.note(f"[refine-fallback] geometric compaction {strength}/{anchor}")
        return {"round": st.compaction_round}

    def compact(self, strength: float, anchor: str) -> dict:
        """Geometric compaction toward an anchor (the agent's placement decision),
        then legalize, place, fast-place and measure. Keeps the densest result."""
        st = self.st
        st.compaction_round += 1
        n = st.compaction_round
        cx0, cy0, cx1, cy1 = st.core
        ents = [dict(e) for e in st.entries]
        ents = geo.compact_entries(ents, strength, anchor, cx0, cy0, cx1, cy1)
        ents = geo.validate_entries(ents, cx0, cy0, cx1, cy1, mw=st.mw,
                                    mh=st.mh, macro_sizes=st.macro_sizes)
        ents = geo.legalize(ents, st.macro_sizes, cx0, cy0, cx1, cy1,
                            mw=st.mw, mh=st.mh, gap=2.0)
        st.entries = ents
        tag = f"{self._base_tag()}_c{n}"
        self._place(ents, tag=tag)
        density = self._fast(tag)
        d = density or 0.0
        st.last_density = d
        st.history.append({"round": n, "strength": strength, "anchor": anchor,
                           "density": round(d, 4)})
        improved = d > st.best_density
        if improved:
            st.best_density = d
            st.best_entries = [dict(e) for e in ents]
        st.stage = "fast_placed"
        self.ctx.note(f"[compact] round {n}: strength={strength} anchor={anchor} "
                      f"density={d:.4f} best={st.best_density:.4f} "
                      f"{'(improved)' if improved else '(no gain)'}")
        return {"stage": st.stage, "round": n, "strength": strength, "anchor": anchor,
                "density": round(d, 4), "best_density": round(st.best_density, 4),
                "improved": improved,
                "fast_image": str(self.ctx.img(f"{tag}_fast_placed.png",
                                               stage="fast_placed"))}

    def detail_place(self, entries: list | None = None, tag: str = "iter00") -> dict:
        """Full detailed placement (FIRM macros, cut_rows, 5um halo, PDN, RePlAce)
        on a macro arrangement. ``entries`` defaults to the density-best (classic
        single pass); the exploration loop passes the *current candidate's* entries
        and a per-try ``tag``. Produces the routable checkpoint."""
        st = self.st
        ents = entries if entries is not None else (st.best_entries or st.entries)
        self._place(ents, tag=f"{tag}_detail")
        t0 = time.time()
        out = self.s.source(self.scripts["detail_place"], timeout=T_DETAIL)
        (self.cfg.stage_report_dir("detail_placed") / f"{tag}_detail_place.log").write_text(out)
        # Bail out cleanly if the script timed out / the process died: continuing
        # would write a half-built db and the route would fail with the opaque
        # "no network has been linked". Surface the real cause instead.
        if "command timed out" in out or "exited unexpectedly" in out:
            reason = ("timed out (exceeded T_DETAIL)" if "timed out" in out
                      else "OpenROAD process died")
            self.ctx.note(f"[detail_place] {tag} FAILED: {reason}")
            st.stage = "detail_failed"
            return {"stage": "detail_failed",
                    "error": f"detailed placement {reason}; no routable checkpoint",
                    "density": None, "detail_db": None}
        snap = self.ctx.snapshot("detail_placed", f"{tag}_detail_placed.png",
                                 "detailed placement")
        db = self.cfg.save_dir / f"{tag}_detail_placed.odb"
        self.s.write_db(db, timeout=120)
        self.ctx.update_view(db)           # --gui: relaunch the live window
        st.best_detail_db = str(db)
        density = self.measure_density()
        if density is not None:
            snap.meta["density"] = round(density, 4)
        st.stage = "detailed"
        self.ctx.note(f"[detail_place] {tag} done in {time.time() - t0:.0f}s, "
                      f"density={density}, db={db.name}")
        return {"stage": st.stage, "density": density, "detail_db": str(db),
                "detail_image": snap.path}

    def route_finalize(self, tag: str = "iter00", out_odb=None) -> dict:
        """CTS + light route on the detailed checkpoint, in a FRESH process so it
        can't inherit leaked PDN/route state. Reads post-route ``wns/tns`` (the PPA
        signal the goal loop checks). ``out_odb`` defaults to ``best_overall.odb``
        (classic single pass); the exploration loop routes to a per-try odb and
        promotes the goal-best afterwards."""
        st = self.st
        if not st.best_detail_db:
            return {"error": "no detailed-placement checkpoint yet — run detail_place first"}
        out_odb = Path(out_odb) if out_odb else self.cfg.best_overall_odb
        self.ctx.note(f"[route] {tag}: fresh process + preload + "
                      f"{self.cfg.route_mode_effective} CTS/route")
        self.s.fresh_start()
        preload = R.ROUTE_PRELOAD.format(data_dir=self.cfg.data_dir,
                                         tech_dir=self.cfg.tech_dir,
                                         best_db=st.best_detail_db)
        self.s.send_command(preload, timeout=300)
        t0 = time.time()
        out = self.s.send_command(
            f"catch {{source {self.scripts['post_route']}}} _e\n"
            f'if {{$_e ne ""}} {{ puts "CTS_ROUTE_ERROR: $_e" }}\n'
            f"catch {{report_tns}}\ncatch {{report_wns}}\n"
            f"catch {{report_design_area}}\ncatch {{report_wire_length}}\n"
            f"catch {{report_power}}\n",
            timeout=T_ROUTE)
        (self.cfg.stage_report_dir("routed") / f"{tag}_cts_route.log").write_text(out)
        # Hard-fail if CTS/route aborted. The wrapper prints "CTS_ROUTE_ERROR:"
        # when `source` of the template raised (e.g. CTS-0114 clock overlap).
        # Without this guard the subsequent report_tns/report_wns return a
        # vacuous WNS=TNS=0 (no clock tree -> no constrained paths) that looks
        # like perfectly clean timing — a false pass on a design that never
        # actually routed.
        if "CTS_ROUTE_ERROR" in out:
            err_line = next((ln.strip() for ln in out.splitlines()
                             if "CTS_ROUTE_ERROR" in ln), "CTS_ROUTE_ERROR")
            self.ctx.note(f"[route] {tag} ABORTED: {err_line}")
            st.routed_metrics = None
            st.stage = "route_failed"
            return {"stage": "route_failed",
                    "error": f"CTS/route aborted ({err_line}); "
                             "PPA metrics are not valid for this candidate",
                    "metrics": {}, "log": str(
                        self.cfg.stage_report_dir("routed") / f"{tag}_cts_route.log")}
        # A crash/OOM mid-route, or any non-completion, leaves the route script's
        # final ROUTE_ALL_DONE marker absent. Treat that as a failure rather than
        # parsing a truncated set of reports (which would record misleading PPA).
        if "ROUTE_ALL_DONE" not in out:
            reason = ("OpenROAD process exited unexpectedly"
                      if "exited unexpectedly" in out
                      else "route did not reach completion (no ROUTE_ALL_DONE)")
            self.ctx.note(f"[route] {tag} INCOMPLETE: {reason}")
            st.routed_metrics = None
            st.stage = "route_failed"
            return {"stage": "route_failed",
                    "error": f"CTS/route incomplete ({reason}); PPA not valid",
                    "metrics": {}, "log": str(
                        self.cfg.stage_report_dir("routed") / f"{tag}_cts_route.log")}
        metrics = M.parse_metrics(out)
        drc_rpt = self.cfg.report_dir / f"04_{self.cfg.design}.drc.rpt"
        try:
            if drc_rpt.exists():
                metrics["drc"] = drc_rpt.read_text().count("violation type:")
        except OSError:
            pass
        metrics.setdefault("drc", 0)
        snap = self.ctx.snapshot("routed", f"{tag}_routed.png", "final routed")
        self.s.write_db(out_odb, timeout=120)
        self.ctx.update_view(out_odb)      # --gui: show final routed
        st.routed_metrics = metrics
        st.stage = "routed"
        self.ctx.note(f"[route] {tag} done in {time.time() - t0:.0f}s -> {metrics}")
        return {"stage": st.stage, "metrics": metrics, "deliverable": str(out_odb),
                "routed_image": snap.path, "routed_odb": str(out_odb)}

    # ------------------------------------------------------------------
    def run_stage(self, stage_key: str) -> dict:
        """Generic stage executor — render + source any stage's TCL template and
        snapshot the result. The hook that lets agents run *other* flow stages
        from the skills directory without bespoke wiring."""
        sk = self.ctx.skills
        if not sk.has_stage(stage_key):
            return {"error": f"unknown stage '{stage_key}'"}
        stage = sk.stage(stage_key)
        if stage.kind != "script" or not stage.template:
            return {"error": f"stage '{stage_key}' is not a runnable script stage"}
        script = self.scripts.get(stage_key)
        if not script:
            script = str(self.cfg.render(sk.template_path(stage_key)))
        timeout = {"detail_place": T_DETAIL, "post_route": T_ROUTE}.get(stage_key, T_FAST)
        self.ctx.note(f"[stage:{stage_key}] sourcing {script}")
        out = self.s.source(script, timeout=timeout)
        snap = self.ctx.snapshot(stage_key, f"stage_{stage_key}.png", stage.title)
        return {"stage": stage_key, "image": snap.path,
                "output_tail": out[-1200:]}

    # ==================================================================
    # Measurements / queries
    # ==================================================================
    def measure_density(self) -> float | None:
        return M.parse_density(self.s.run_recipe(R.STDCELL_DENSITY, timeout=120))

    def check_overlaps(self) -> int | None:
        return M.parse_overlaps(self.s.run_recipe(R.CHECK_OVERLAPS, timeout=60))

    def check_outside_core(self) -> int | None:
        return M.parse_outside_core(
            self.s.run_recipe(R.CHECK_MACROS_IN_CORE, timeout=60))

    def measure_congestion(self) -> dict:
        density = self.measure_density()
        overlaps = self.check_overlaps()
        return {
            "skipped_global_route": True,
            "reason": ("Gate-B congestion is approximated from fast-placement "
                       "density/overlap checks; final route computes real "
                       "routing/timing metrics."),
            "density": density,
            "overlaps": overlaps,
        }

    # ------------------------------------------------------------------
    def _core_str(self) -> str:
        if not self.st.core:
            return "?"
        cx0, cy0, cx1, cy1 = self.st.core
        return f"({cx0:.1f},{cy0:.1f})-({cx1:.1f},{cy1:.1f})"

    def design_state_summary(self) -> dict:
        st = self.st
        util = st.util_estimate()
        sizes = sorted({(round(w, 1), round(h, 1)) for w, h in st.macro_sizes.values()})
        return {
            "stage": st.stage,
            "n_macros": st.n_macros,
            "core_bounds": self._core_str(),
            "macro_footprints": [f"{w}x{h}" for w, h in sizes][:8],
            "macro_area_util": None if util is None else round(util, 3),
            "compaction_rounds_done": st.compaction_round,
            "current_density": st.last_density,
            "best_density": round(st.best_density, 4) if st.best_density else 0.0,
            "have_detail_checkpoint": st.best_detail_db is not None,
            "routed": st.routed_metrics is not None,
            "snapshots": len(self.ctx.registry.all()),
            "compaction_history": st.history[-6:],
        }
