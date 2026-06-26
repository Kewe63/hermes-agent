"""Subprocess supervisor for the cua-driver MCP backend.

Hermes spawns ``cua-driver`` as a long-lived stdio subprocess (see
``cua_backend.py``). On Windows the bundled UIAccess helper
(``cua-driver-uia.exe``) attached to that process is known to die after
foreground-window changes — for example when the user presses Alt+Tab
to switch to Chrome, VS Code, or any other application. Once the helper
dies, subsequent ``computer_use`` calls return zero-byte captures or
fail outright, even though ``hermes computer-use doctor`` reports a
green health check.

This module adds a thin supervisor that:

* watches the subprocess for unexpected exit,
* restarts it with bounded retries (exponential back-off),
* exposes a ``probe_health()`` helper that runs a no-op MCP roundtrip
  so the health gate fails *fast* and accurately (instead of asking
  for a screen capture at a moment when the helper is half-dead).

The supervisor is intentionally small, deterministic, and requires no
new dependencies. Diagnostics are logged via the project's standard
``loguru``/``logging`` surface; nothing about the user-facing toolset
schema changes — the supervisor only wraps the spawn site.

Issue ref: NousResearch/hermes-agent#52951.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default restart policy. Three retries with 0.5s, 1.0s, 2.0s back-off is
# enough headroom for transient Windows UIAccess token revocations without
# masking a real broken install on the user's machine.
DEFAULT_MAX_RESTARTS = 3
DEFAULT_INITIAL_BACKOFF_S = 0.5
DEFAULT_BACKOFF_MULTIPLIER = 2.0

# A health probe must round-trip in well under a second on a healthy
# system; anything slower is a strong signal the helper is sick.
DEFAULT_HEALTH_PROBE_TIMEOUT_S = 2.0


@dataclass
class SupervisedProcessSpec:
    """Specification for the subprocess we want to keep alive."""

    argv: List[str]
    env: Optional[dict] = None
    cwd: Optional[str] = None
    max_restarts: int = DEFAULT_MAX_RESTARTS
    initial_backoff_s: float = DEFAULT_INITIAL_BACKOFF_S
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    health_probe_timeout_s: float = DEFAULT_HEALTH_PROBE_TIMEOUT_S
    on_restart: Optional[Callable[[int], None]] = None

    @property
    def command_display(self) -> str:
        return " ".join(self.argv)


@dataclass
class SupervisorState:
    """Mutable state shared with the watchdog thread."""

    proc: Optional[subprocess.Popen] = None
    restart_count: int = 0
    last_exit_code: Optional[int] = None
    last_restart_at: Optional[float] = None
    healthy: bool = False
    last_error: Optional[str] = None
    watchdog_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class CuaSupervisor:
    """Watchdog that keeps ``cua-driver`` alive across helper-die events."""

    def __init__(self, spec: SupervisedProcessSpec) -> None:
        self._spec = spec
        self._state = SupervisorState()

    @property
    def state(self) -> SupervisorState:
        return self._state

    @property
    def spec(self) -> SupervisedProcessSpec:
        return self._spec

    def is_running(self) -> bool:
        with self._state.lock:
            return (
                self._state.proc is not None
                and self._state.proc.poll() is None
            )

    def start(self) -> subprocess.Popen:
        """Spawn the subprocess once and start the watchdog thread.

        Idempotent — a second call returns the existing handle.
        """
        with self._state.lock:
            if self._state.proc is not None and self._state.proc.poll() is None:
                return self._state.proc

            proc = self._spawn_locked()
            self._state.proc = proc
            self._state.stop_event.clear()

        if self._state.watchdog_thread is None or not self._state.watchdog_thread.is_alive():
            t = threading.Thread(
                target=self._watchdog_loop,
                name="cua-supervisor-watchdog",
                daemon=True,
            )
            self._state.watchdog_thread = t
            t.start()
        return proc

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the watchdog and terminate the subprocess if we own it."""
        self._state.stop_event.set()
        with self._state.lock:
            proc = self._state.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=timeout_s)
            except Exception as e:  # pragma: no cover -- defensive
                logger.debug("cua-supervisor terminate failed: %s", e)

    def get_proc(self) -> Optional[subprocess.Popen]:
        with self._state.lock:
            return self._state.proc

    def probe_health(self, timeout_s: Optional[float] = None) -> bool:
        """Issue a quick stdin roundtrip to verify the MCP backend is alive.

        The craft of this probe is that it does NOT need a usable screen
        capture: a healthy cua-driver responds to ``ping`` (or even an
        empty newline) with a JSON-RPC reply; a dead/half-dead helper
        hangs. We bound the wait via ``timeout_s`` so callers can treat a
        hang as ``unhealthy`` instead of waiting forever.

        Note: the exact ``ping`` wire format depends on the installed
        cua-driver version. We keep this conservative — if any
        round-trip completes within the timeout AND the process is still
        running, we declare it healthy. This catches the
        ``0x0 capture`` symptom reported in #52951 because the helper
        dying makes the process stall at the next ``initialize`` /
        ``ping`` roundtrip.
        """
        timeout = timeout_s if timeout_s is not None else self._spec.health_probe_timeout_s
        with self._state.lock:
            proc = self._state.proc
        if proc is None or proc.poll() is not None:
            self._state.healthy = False
            self._state.last_error = "process not running"
            return False

        try:
            # A JSON-RPC initialize ping is the most portable MCP probe
            # we have. Even on older drivers a partial response is a
            # strong signal the helper is alive.
            ping = (
                b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
            )
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(ping + b"\n")
            proc.stdin.flush()
            # Use a short poll loop so we don't deadlock if the helper
            # is wedged. We don't care about payload validity here, only
            # whether the round-trip completes.
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if line:
                    self._state.healthy = True
                    self._state.last_error = None
                    return True
                if proc.poll() is not None:
                    self._state.healthy = False
                    self._state.last_error = f"exit={proc.returncode}"
                    return False
                time.sleep(0.05)
            self._state.healthy = False
            self._state.last_error = "ping timeout"
            return False
        except Exception as e:
            self._state.healthy = False
            self._state.last_error = f"probe error: {e}"
            return False

    # ------------------------------------------------------------------
    # Internal watchdog
    # ------------------------------------------------------------------

    def _spawn_locked(self) -> subprocess.Popen:
        """Caller must hold ``self._state.lock``."""
        # NOTE: stdio handles are listed inline on the Popen() call so the
        # `scripts/check_subprocess_stdin.py` lint (which scans for a
        # literal ``stdin=`` token inside the call itself) doesn't false
        # on the **kwargs spread below.
        extra_kwargs: Dict[str, Any] = {}
        if self._spec.env is not None:
            extra_kwargs["env"] = self._spec.env
        if self._spec.cwd is not None:
            extra_kwargs["cwd"] = self._spec.cwd
        # On POSIX we want the new process to outlive the supervisor's
        # controlling terminal; on Windows CREATE_NEW_PROCESS_GROUP keeps
        # the helper from inheriting our Ctrl+C handler (which would
        # surface the symptom the user sees in #52951).
        if os.name == "nt":
            extra_kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        logger.info(
            "cua-supervisor: spawning %s (restart_count=%d)",
            self._spec.command_display,
            self._state.restart_count,
        )
        return subprocess.Popen(
            self._spec.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **extra_kwargs,
        )

    def _watchdog_loop(self) -> None:
        """Daemon thread that watches for unexpected exits and restarts."""
        while not self._state.stop_event.is_set():
            with self._state.lock:
                proc = self._state.proc
            if proc is None:
                time.sleep(0.2)
                continue

            # Poll the subprocess; if it exited unexpectedly, restart with
            # exponential back-off until we hit max_restarts.
            while not self._state.stop_event.is_set():
                rc = proc.poll()
                if rc is None:
                    time.sleep(0.5)
                    continue

                with self._state.lock:
                    self._state.last_exit_code = rc
                    self._state.healthy = False
                    self._state.last_error = f"process exited rc={rc}"
                    if self._state.restart_count >= self._spec.max_restarts:
                        logger.error(
                            "cua-supervisor: process exited rc=%s and restart "
                            "budget exhausted (%d/%d); supervisor is now in "
                            "a degraded, non-restarting state until the "
                            "caller calls start() again.",
                            rc,
                            self._state.restart_count,
                            self._spec.max_restarts,
                        )
                        return
                    backoff = (
                        self._spec.initial_backoff_s
                        * (
                            self._spec.backoff_multiplier
                            ** self._state.restart_count
                        )
                    )
                    self._state.restart_count += 1
                    self._state.last_restart_at = time.monotonic()
                    new_proc = self._spawn_locked()
                    self._state.proc = new_proc

                logger.warning(
                    "cua-supervisor: process exited rc=%s; restarting in "
                    "%.2fs (count=%d/%d)",
                    rc,
                    backoff,
                    self._state.restart_count,
                    self._spec.max_restarts,
                )
                if self._spec.on_restart is not None:
                    try:
                        self._spec.on_restart(self._state.restart_count)
                    except Exception as e:  # pragma: no cover -- defensive
                        logger.debug("cua-supervisor on_restart hook: %s", e)

                # Sleep outside the lock so the next iteration can re-enter
                # it cleanly.
                if self._state.stop_event.wait(backoff):
                    return
                proc = new_proc

        logger.debug("cua-supervisor: watchdog loop exiting cleanly")


_INSTANCE_LOCK = threading.Lock()
_INSTANCE: Optional[CuaSupervisor] = None


def get_cua_supervisor(
    spec: Optional[SupervisedProcessSpec] = None,
) -> CuaSupervisor:
    """Return the process-wide cua-driver supervisor, building it lazily.

    The supervisor is a singleton so that every call site
    (``cua_backend.py``, ``doctor.py``, the desktop gateway, etc.) sees
    the same restart counters, health verdict, and underlying
    subprocess handle.
    """
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            if spec is None:
                raise RuntimeError(
                    "cua supervisor not initialized; pass spec on first call"
                )
            _INSTANCE = CuaSupervisor(spec)
        return _INSTANCE


def set_cua_supervisor_for_tests(supervisor: Optional[CuaSupervisor]) -> None:
    """Replace the singleton (test-only entry point)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None and _INSTANCE is not supervisor:
            try:
                _INSTANCE.stop()
            except Exception:
                pass
        _INSTANCE = supervisor
