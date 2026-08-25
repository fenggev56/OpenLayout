"""Shared flow state + context — the single source of truth the agents reason
about and the operations mutate.

:class:`FlowState` is plain data (the canonical placement + bookkeeping).
:class:`FlowContext` bundles the infra (config, OpenROAD session, VLM, skill
library, snapshot registry) and the live state, plus tiny helpers for naming and
registering images. Heavy EDA behaviour lives in :mod:`macroflow.operations`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from . import geometry as geo
from .config import FlowConfig
from .skills import SkillLibrary
from .snapshots import SnapshotRegistry
from .telemetry import Telemetry


def draw_pin_overlay(path, area, pin_boxes) -> None:
    """Draw top-level IO pin boxes onto an OpenROAD PNG.

    OpenROAD rasterizes thin boundary pins to near-invisibility at the die zoom
    level, so the checker (and humans) lose the pin sides. This redraws each
    bterm box as a small cyan marker on top of the rendered image.

    ``area`` is the ``(x0, y0, x1, y1)`` micron window the PNG was rendered with
    (it MUST match the ``save_image -area`` that produced ``path``) — it sets the
    micron->pixel mapping. No-op when there is no area, no pins, or the file
    can't be opened. Shared by :meth:`FlowContext.snapshot` and the standalone
    ``save_die_image`` tool so both overlays are identical.
    """
    if not area or not pin_boxes:
        return
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return
    ax0, ay0, ax1, ay1 = area
    aw = max(1e-9, ax1 - ax0)
    ah = max(1e-9, ay1 - ay0)
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def px(x: float) -> float:
        return (x - ax0) / aw * (w - 1)

    def py(y: float) -> float:
        return (ay1 - y) / ah * (h - 1)

    for x0, y0, x1, y1 in pin_boxes:
        left = px(x0)
        right = px(x1)
        top = py(y1)
        bottom = py(y0)
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        half = 2.0
        left = min(left, cx - half)
        right = max(right, cx + half)
        top = min(top, cy - half)
        bottom = max(bottom, cy + half)
        draw.rectangle([left, top, right, bottom],
                       fill=(0, 220, 255, 150),
                       outline=(255, 255, 255, 230),
                       width=1)
    Image.alpha_composite(img, overlay).convert("RGB").save(path)


@dataclass
class FlowState:
    """Canonical design state, mutated only through :mod:`macroflow.operations`."""

    stage: str = "init"                 # init|seeded|fast_placed|detailed|routed|done
    entries: list = field(default_factory=list)   # current macro plan [{name,x,y,orient,area}]
    die: tuple | None = None            # (dx0, dy0, dx1, dy1) microns
    core: tuple | None = None           # (cx0, cy0, cx1, cy1) microns
    macro_sizes: dict = field(default_factory=dict)
    mw: float = geo.DEFAULT_MW
    mh: float = geo.DEFAULT_MH
    n_macros: int = 0

    dims_text: str = ""
    macros_text: str = ""
    pin_boxes: list = field(default_factory=list)
    macro_pin_sides: dict = field(default_factory=dict)
    macro_index: list = field(default_factory=list)  # [(name, w, h)] in DB order
    regions: list = field(default_factory=list)       # NxN region grid cells

    last_density: float | None = None
    best_density: float = 0.0
    best_entries: list = field(default_factory=list)
    best_detail_db: str | None = None
    routed_metrics: dict | None = None
    last_macro_db: str | None = None
    last_fast_db: str | None = None

    compaction_round: int = 0
    history: list = field(default_factory=list)  # [{round,strength,anchor,density}]

    try_tag: str = ""                  # per-exploration-try tag (e.g. "t01"); set by the explorer

    def util_estimate(self) -> float | None:
        """Rough macro-area / core-area fraction (sanity only)."""
        if not self.core or not self.macro_sizes:
            return None
        cx0, cy0, cx1, cy1 = self.core
        core_area = max(1e-9, (cx1 - cx0) * (cy1 - cy0))
        macro_area = sum(w * h for w, h in self.macro_sizes.values())
        return macro_area / core_area

    def image_area(self, margin: float = 0.03):
        """Bounds for visualization: die/core plus pin boxes, with margin."""
        bounds = [self.die or self.core]
        bounds.extend(self.pin_boxes)
        bounds = [b for b in bounds if b]
        if not bounds:
            return None
        x0 = min(b[0] for b in bounds)
        y0 = min(b[1] for b in bounds)
        x1 = max(b[2] for b in bounds)
        y1 = max(b[3] for b in bounds)
        pad = max(x1 - x0, y1 - y0) * max(0.0, margin)
        return x0 - pad, y0 - pad, x1 + pad, y1 + pad


class FlowContext:
    """Infra + state holder shared by operations, tools and agents."""

    def __init__(self, cfg: FlowConfig, session, vlm, skills: SkillLibrary,
                 registry: SnapshotRegistry):
        self.cfg = cfg
        self.session = session
        self.vlm = vlm
        self.skills = skills
        self.registry = registry
        self.state = FlowState()
        self.telemetry = Telemetry()
        self.gui = False                  # --gui: live OpenROAD GUI window
        self.view_tcl = self.cfg.save_dir / "_gui_show.tcl"   # kill+relaunch spec
        self.transcript: list[str] = []   # human-readable op log for the summary

    # ------------------------------------------------------------------
    def img(self, name: str, stage: str | None = None) -> Path:
        if stage:
            return self.cfg.stage_image_dir(stage) / name
        return self.cfg.image_dir / name

    def db(self, name: str) -> Path:
        return self.cfg.save_dir / name

    def note(self, msg: str) -> None:
        self.transcript.append(msg)
        print(f"  {msg}", flush=True)

    def snapshot(self, stage: str, name: str, desc: str = "",
                 timeout: float = 30.0, **meta):
        """Render a PNG of the current layout and register it for the agents."""
        path = self.img(name, stage=stage)
        area = self.state.image_area(self.cfg.save_image_margin)
        self.session.save_image(path, timeout=timeout, area=area)
        if self.cfg.save_pin_overlay:
            self._overlay_pin_boxes(path, area)
        return self.registry.register(stage, str(path), desc, **meta)

    def _overlay_pin_boxes(self, path: Path, area) -> None:
        draw_pin_overlay(path, area, self.state.pin_boxes)

    def update_view(self, odb_path) -> None:
        """--gui: point the live OpenROAD window at the latest checkpoint. Rewrites
        the view TCL; the GuiWatcher thread notices and relaunches the window."""
        if not self.gui:
            return
        from . import gui
        try:
            gui.write_view_tcl(self.view_tcl, odb_path)
        except OSError as e:
            self.note(f"[gui] could not update view: {e}")
