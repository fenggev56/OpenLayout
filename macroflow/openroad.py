"""OpenROAD session — the Python side of "system operation".

A persistent headless ``openroad`` subprocess with prompt-synchronized command
I/O, auto-restart on death, a bulletproof ``fresh_start`` (kill + relaunch — the
only reliable way to clear cut_rows / PDN stripes / the core voltage domain that
``read_db`` leaves behind), and small convenience wrappers (save_image, write_db,
read_db, run_recipe). Self-contained: no dependency on ``openlayout``.

If ``openroad`` is not on PATH it runs in *simulation mode* (logs commands,
returns placeholders) so the orchestration logic can be exercised without the
tool installed.
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import threading
import time
from pathlib import Path


class _OutputReader(threading.Thread):
    """Reads tool stdout in chunks and signals when the per-command sync token
    (a unique end-of-command sentinel appended by :class:`ORSession`) appears.

    Using a unique token per command — instead of a generic ``openroad>`` prompt —
    avoids desync on builds where the prompt is emitted more than once or a stale
    prompt from the previous command lingers in the pipe.
    """

    def __init__(self, stream, prompt_pattern: str):
        super().__init__(daemon=True)
        self.stream = stream
        self.fd = stream.fileno()
        self.prompt_re = re.compile(prompt_pattern)
        self.buffer: list[str] = []
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.closed = threading.Event()   # set when stdout hits EOF (process died)
        self.prompt_seen = False          # did the last get_output see the token?
        self._stop = False
        self._token: str | None = None    # current end-of-command sentinel
        self._tail = ""                   # carry-over so a token split across
                                          # read chunks is still detected

    def arm(self, token: str) -> None:
        """Set the sentinel to wait for and reset the ready/tail state."""
        with self.lock:
            self._token = token
            self._tail = ""
        if not self.closed.is_set():
            self.ready.clear()

    def run(self):
        try:
            while not self._stop:
                rlist, _, _ = select.select([self.fd], [], [], 1.0)
                if not rlist:
                    continue
                try:
                    chunk = os.read(self.fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                with self.lock:
                    self.buffer.append(text)
                    tok = self._token
                    if tok:
                        self._tail = (self._tail + text)[-(len(tok) + 128):]
                        hit = tok in self._tail
                    else:
                        hit = bool(self.prompt_re.search(text))
                if hit:
                    self.ready.set()
        except (ValueError, OSError):
            pass
        finally:
            # stdout closed → the openroad process exited (cleanly or crashed).
            # Unblock any waiter so a dead child can't hang the flow for the full
            # (multi-hour) command timeout.
            self.closed.set()
            self.ready.set()

    def get_output(self, timeout: float = 60.0) -> str:
        """Wait for the next prompt, or return early if the process died.

        Sets ``self.prompt_seen`` so the caller can distinguish a completed
        command (prompt reappeared) from a timeout (command still running) —
        the latter must NOT be treated as success or the next command desyncs.
        """
        deadline = time.monotonic() + timeout
        self.prompt_seen = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self.ready.wait(timeout=min(1.0, remaining)):
                self.prompt_seen = True
                break
            if self.closed.is_set():
                break
        with self.lock:
            out = "".join(self.buffer)
            self.buffer.clear()
            # Only clear ready while the process is alive; once closed it must
            # stay set so subsequent waiters also return immediately.
            if not self.closed.is_set():
                self.ready.clear()
        return out

    def stop(self):
        self._stop = True


class ORSession:
    """Headless OpenROAD session with auto-restart and multi-line TCL."""

    # Unique end-of-command sentinel prefix. A per-command counter is appended so
    # every command has a distinct token; the reader waits for exactly that token.
    # Needed because newer OpenROAD builds (26Q3+) don't print the ``openroad>``
    # prompt for piped stdin, and a generic prompt is race-prone (stale prompts).
    _SYNC_PREFIX = "__MF_SYNC_"

    def _sync_token(self) -> str:
        return f"{self._SYNC_PREFIX}{self._cmd_count}__"

    def _emit_tcl(self, token: str) -> str:
        # Print the sentinel on its own line, then flush so it isn't buffered.
        return f'puts "\\n{token}" ; flush stdout\n'

    def __init__(self, session_dir: Path, executable: str = "openroad",
                 shell_args: tuple[str, ...] = ("-no_init",),
                 prompt_pattern: str = r"openroad>\s*",
                 save_image_width: int | None = None,
                 save_image_resolution: float | None = None):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.executable = executable
        self.shell_args = list(shell_args)
        self.prompt_pattern = prompt_pattern
        self.log_path = self.session_dir / "openroad_session.log"
        self.process: subprocess.Popen | None = None
        self._reader: _OutputReader | None = None
        self._sim = False
        self._init_script: str | None = None
        self._cmd_count = 0
        self.save_image_width = save_image_width
        self.save_image_resolution = save_image_resolution

    # ------------------------------------------------------------------
    @property
    def is_simulation(self) -> bool:
        return self._sim

    def check_available(self) -> bool:
        try:
            r = subprocess.run(["which", self.executable], capture_output=True,
                               text=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start(self) -> bool:
        """Start (or restart) the subprocess. Returns True if real, False if sim."""
        # Qt/offscreen hygiene so save_image works headless.
        for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            os.environ.pop(key, None)
        os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

        if not self.check_available():
            self._sim = True
            self._log(f"[SIMULATION] {self.executable} not on PATH.\n")
            return False
        cmd = [self.executable] + self.shell_args
        self._log(f"[SESSION] Starting: {' '.join(cmd)}\n")
        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, cwd=str(self.session_dir),
            )
            self._reader = _OutputReader(self.process.stdout, self.prompt_pattern)
            self._reader.start()
            # Emit an initial sentinel so the startup wait resolves promptly even
            # on builds that don't print ``openroad>`` for piped stdin.
            token = f"{self._SYNC_PREFIX}start__"
            self._reader.arm(token)
            try:
                self.process.stdin.write(self._emit_tcl(token).encode("utf-8"))
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            self._log(self._reader.get_output(timeout=120.0))
            return True
        except (FileNotFoundError, OSError) as e:
            self._sim = True
            self._log(f"[SIMULATION] Failed to start: {e}\n")
            return False

    def set_init_script(self, path: str):
        """Remember the init script so auto-restart can re-source the design."""
        self._init_script = str(path)

    # ------------------------------------------------------------------
    def is_alive(self) -> bool:
        if self._sim:
            return True
        return self.process is not None and self.process.poll() is None

    def _restart(self) -> bool:
        """Restart after an unexpected death and re-source the init script."""
        print("  [OpenROAD] session died — restarting...", flush=True)
        self.close(graceful=False)
        ok = self.start()
        if ok and self._init_script:
            print("  [OpenROAD] re-loading design...", flush=True)
            self.send_command(f"source {self._init_script}", timeout=300)
        return ok

    def fresh_start(self) -> bool:
        """Kill the process and start a brand-new one WITHOUT auto-sourcing.

        The leak-proof reset: ``read_db`` into a live process does not clear
        cut_rows / PDN stripes / the core voltage domain. A new process does.
        The caller decides what to load next.
        """
        print("  [OpenROAD] fresh process restart (clean state)...", flush=True)
        self.close(graceful=False)
        return self.start()

    def send_command(self, command: str, timeout: float = 120.0) -> str:
        """Send a (possibly multi-line) TCL command; return output up to prompt."""
        if not self._sim and self.process is not None and not self.is_alive():
            self._restart()
        command = command.strip()
        self._cmd_count += 1
        if self._sim:
            self._log(f"[SIM-CMD #{self._cmd_count}] {command[:200]}\n")
            return f"[SIMULATION] command #{self._cmd_count} logged."
        if "\n" in command:
            tmp = self.session_dir / "_mf_cmd.tcl"
            tmp.write_text(command)
            return self._send_line(f"source {tmp}", timeout)
        return self._send_line(command, timeout)

    def _send_line(self, line: str, timeout: float) -> str:
        if not self.is_alive():
            return "[ERROR] OpenROAD session is not running."
        self._log(f"[CMD #{self._cmd_count}] {line}\n")
        # Newer OpenROAD builds (26Q3+) do NOT print the ``openroad>`` prompt when
        # stdin is a pipe (non-tty), so the reader could never detect command
        # completion and every command timed out. Piped OpenROAD does continue past
        # command errors, so we append a unique sentinel after each command and
        # wait for it; the sentinel is stripped from the returned output.
        token = self._sync_token()
        self._reader.arm(token)
        try:
            payload = line + "\n" + self._emit_tcl(token)
            self.process.stdin.write(payload.encode("utf-8"))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return f"[ERROR] failed to send command: {e}"
        out = self._reader.get_output(timeout=timeout)
        # Drop the sentinel line (and any trailing prompt fragment) from output.
        out = re.sub(r"\n?" + re.escape(token) + r"[^\n]*\n?", "", out)
        # If stdout hit EOF during the command the process exited (crash / OOM);
        # send_command never issues a clean "exit" through here, so this is always
        # unexpected. Surface it so callers (e.g. route_finalize) treat it as a
        # failure instead of parsing a truncated, misleading result.
        if self._reader is not None and self._reader.closed.is_set():
            try:
                self.process.wait(timeout=2)
            except Exception:
                pass
            rc = self.process.poll() if self.process is not None else None
            out += (f"\n[ERROR] OpenROAD process exited unexpectedly "
                    f"(returncode={rc}) during command.\n")
        elif self._reader is not None and not self._reader.prompt_seen:
            # The command did not finish within `timeout` (no prompt, process
            # still alive). Returning now would desync the session: the next
            # command's output would interleave with this still-running one
            # (e.g. detail_place overrunning T_DETAIL -> a stray write_db hits a
            # half-built db -> ORD-0007 -> route "no network linked"). Force a
            # clean reset and surface the timeout so callers fail this candidate
            # explicitly instead of silently corrupting downstream stages.
            out += (f"\n[ERROR] command timed out after {timeout:.0f}s "
                    f"(still running); session reset.\n")
            self.close(graceful=False)
        self._log(out)
        return out

    # --- convenience wrappers --------------------------------------------------
    def run_recipe(self, recipe_tcl: str, timeout: float = 120.0) -> str:
        return self.send_command(recipe_tcl, timeout=timeout)

    @staticmethod
    def _area_arg(area) -> str:
        if not area:
            return ""
        try:
            x0, y0, x1, y1 = [float(v) for v in area]
        except (TypeError, ValueError):
            return ""
        return f" -area {{{x0:g} {y0:g} {x1:g} {y1:g}}}"

    def save_image(self, path, timeout: float = 30.0, area=None) -> str:
        area_arg = self._area_arg(area)
        if self.save_image_width:
            return self.send_command(
                f"save_image{area_arg} -width {int(self.save_image_width)} {path}",
                timeout=timeout)
        if self.save_image_resolution:
            return self.send_command(
                f"save_image{area_arg} -resolution {self.save_image_resolution:g} {path}",
                timeout=timeout)
        return self.send_command(f"save_image{area_arg} {path}", timeout=timeout)

    def write_db(self, path, timeout: float = 120.0) -> str:
        return self.send_command(f"write_db {path}", timeout=timeout)

    def read_db(self, path, timeout: float = 120.0) -> str:
        return self.send_command(f"read_db {path}", timeout=timeout)

    def source(self, path, timeout: float = 300.0) -> str:
        return self.send_command(f"source {path}", timeout=timeout)

    # ------------------------------------------------------------------
    def close(self, graceful: bool = True):
        if self._sim:
            return
        if self.process and self.is_alive():
            try:
                if graceful:
                    self.process.stdin.write(b"exit\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=30)
                else:
                    raise subprocess.TimeoutExpired("exit", 0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        if self._reader:
            self._reader.stop()
            self._reader = None
        self.process = None

    def _log(self, text: str):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
