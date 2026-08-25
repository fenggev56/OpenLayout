"""Live OpenROAD GUI display for a macroflow run (the ``--gui`` path).

macroflow's agent loop drives a *headless* OpenROAD session; you can't show a
live GUI inside it (the Qt event loop is incompatible with the prompt-sync). Two
in-process tricks also fail: ``openroad -gui`` does NOT fire Tcl ``after`` timers
(so a self-refreshing window can't reload on its own), and a *second* ``read_db``
in one live GUI doesn't swap the design.

So we use the **kill-and-relaunch** design: macroflow writes a tiny "what to show
now" TCL (``write_view_tcl`` — a ``read_db`` of the latest checkpoint) every time
it reaches a new stage, and :class:`GuiWatcher` (a Python thread, NOT a Tcl
timer) notices the file change and relaunches a fresh ``openroad -gui`` on it.
Each launch is a clean process whose single ``read_db`` works. The window
updates per stage; it never touches the headless flow.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path


def display_available() -> bool:
    """True if an X display is set (needed to open a GUI window)."""
    return bool(os.environ.get("DISPLAY"))


def write_view_tcl(tcl_path, odb_path) -> None:
    """(Over)write the 'display this now' TCL: load + fit the given checkpoint and
    drop a sentinel. Run WITHOUT ``-exit`` so the window stays open; the watcher
    relaunches when this file changes."""
    tcl = Path(tcl_path)
    state = tcl.parent / ".gui_view_state"
    tcl.write_text(
        f"read_db {odb_path}\n"
        f"gui::fit\n"
        f'catch {{ set _f [open "{state}" w]; '
        f'puts $_f "showing [file tail {odb_path}]"; close $_f }}\n'
    )


class GuiWatcher(threading.Thread):
    """Watches the view TCL and kills+relaunches ``openroad -gui`` on each change.

    A Python thread (not a Tcl ``after`` timer, which doesn't fire in -gui). Each
    relaunch is a fresh process — so its single ``read_db`` works. Daemon thread;
    call :meth:`stop` to tear the window down with the run.
    """

    def __init__(self, tcl_path, poll: float = 1.0):
        super().__init__(daemon=True)
        self.tcl = Path(tcl_path)
        self.poll = poll
        self._stop_evt = threading.Event()
        self.proc: subprocess.Popen | None = None
        self._mtime: float | None = None

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                if self.tcl.exists():
                    m = self.tcl.stat().st_mtime
                    if m != self._mtime:
                        self._mtime = m
                        self._relaunch()
            except Exception as e:                       # never let the viewer kill a run
                print(f"  [gui] watcher error: {e}", flush=True)
            self._stop_evt.wait(self.poll)
        self._kill()

    def _relaunch(self) -> None:
        self._kill()
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "xcb"                   # real X display
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        env.pop("QT_QPA_FONTDIR", None)
        try:
            # No -exit: the script runs (read_db + fit), then the window persists.
            self.proc = subprocess.Popen(["openroad", "-gui", str(self.tcl)], env=env)
            print("  [gui] (re)launched live OpenROAD window", flush=True)
        except OSError as e:
            print(f"  [gui] could not launch openroad -gui: {e}", flush=True)
            self.proc = None

    def _kill(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=8)
