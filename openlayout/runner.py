"""OpenROAD batch runner — renders a parameterized TCL template (guarded
``if {![info exists X]} {set X ...}`` vars, same convention as macroflow) and
executes it in a fresh ``openroad -exit`` process.  Fresh processes per stage
are deliberate: they are restartable, parallelizable (optimizer suggestions),
and avoid the stale-state problems of a long-lived interactive session.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def _tcl_quote(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ' \t"[]{}$\\;'):
        return '{' + s + '}'
    return s if s else '{}'


def render(template: str | Path, params: dict, out_path: Path) -> Path:
    """Prepend a ``set var value`` header to a template and write it out."""
    tpl = Path(template)
    if not tpl.is_absolute():
        tpl = SCRIPTS_DIR / tpl
    header = ["# ---- rendered by openlayout.runner ----"]
    for k, v in params.items():
        header.append(f"set {k} {_tcl_quote(v)}")
    header.append("# ---- end header ----\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header) + tpl.read_text())
    return out_path


class RunResult:
    def __init__(self, ok: bool, log_path: Path, seconds: float,
                 returncode: int, timed_out: bool = False):
        self.ok = ok
        self.log_path = log_path
        self.seconds = seconds
        self.returncode = returncode
        self.timed_out = timed_out

    @property
    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def __repr__(self):
        return (f"RunResult(ok={self.ok}, rc={self.returncode}, "
                f"t={self.seconds:.0f}s, log={self.log_path})")


def run_openroad(script: Path, log_path: Path, *, exe: str = "openroad",
                 timeout_s: int = 7200, telemetry=None,
                 stage: str = "") -> RunResult:
    """Run one rendered TCL script in batch mode, capturing stdout+stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # headless image saving
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    env.pop("QT_QPA_FONTDIR", None)
    env["QT_QPA_PLATFORM"] = "minimal"
    t0 = time.monotonic()
    timed_out = False
    with open(log_path, "w") as lf:
        try:
            proc = subprocess.run(
                [exe, "-no_init", "-exit", str(script)],
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=timeout_s, env=env, cwd=str(script.parent))
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            timed_out = True
            lf.write(f"\n[openlayout] TIMEOUT after {timeout_s}s\n")
    dt = time.monotonic() - t0
    if telemetry is not None:
        telemetry.add_tool_run(stage=stage, seconds=dt, label=script.name)
    ok = (rc == 0) and not timed_out
    return RunResult(ok, log_path, dt, rc, timed_out)
