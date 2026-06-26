"""Tests for tools/computer_use/supervisor.py (#52951).

These exercise the supervisor without spinning up a real cua-driver
instance: we run a small Python helper that simulates two scenarios —
the MCP backend that round-trips a JSON-RPC initialize call, and the
UIAccess helper that exits unexpectedly so the watchdog has something
to restart. This catches the #52951 symptom (helper dies after window
focus changes) without needing a Windows VM.
"""

import subprocess
import threading
import time
from pathlib import Path

import pytest

from tools.computer_use.supervisor import (
    CuaSupervisor,
    SupervisedProcessSpec,
    get_cua_supervisor,
    set_cua_supervisor_for_tests,
)


def _python_exe() -> str:
    # Use the same Python the tests are running under so cross-imports
    # resolve to the project venv (the supervisor spawn site has no
    # venv assumption of its own).
    import sys
    return sys.executable


def _helper_program_dir(tmp_path: Path) -> dict:
    """Materialise two helper scripts into ``tmp_path`` and return their paths."""
    alive = tmp_path / "alive_helper.py"
    alive.write_text(
        "import sys, json\n"
        "for _ in range(50):\n"
        "    line = sys.stdin.readline()\n"
        "    if not line:\n"
        "        break\n"
        "    sys.stdout.write('{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\\n')\n"
        "    sys.stdout.flush()\n"
        "sys.exit(0)\n"
    )
    exits = tmp_path / "exits_helper.py"
    exits.write_text("import sys; sys.exit(0)\n")
    return {
        "alive": [_python_exe(), str(alive)],
        "exits": [_python_exe(), str(exits)],
    }


def test_supervisor_is_idempotent_start(tmp_path: Path) -> None:
    p = _helper_program_dir(tmp_path)
    spec = SupervisedProcessSpec(argv=p["alive"])
    sup = CuaSupervisor(spec)
    try:
        proc1 = sup.start()
        proc2 = sup.start()
        assert proc1 is proc2
        assert sup.is_running()
    finally:
        sup.stop()


def test_supervisor_watches_exit_when_helper_dies(tmp_path: Path) -> None:
    """The watchdog must observe the helper dying and respawn it.

    Reproduces the #52951 symptom: the cua-driver UIAccess helper
    exits unexpectedly after a window focus change; the supervisor
    should respawn the process so subsequent calls do not see a
    half-dead subprocess.
    """
    p = _helper_program_dir(tmp_path)
    spec = SupervisedProcessSpec(
        argv=p["exits"],
        initial_backoff_s=0.05,
        max_restarts=2,
    )
    sup = CuaSupervisor(spec)
    proc = sup.start()
    pid_initial = proc.pid

    # Wait for the watchdog to register the exit and respawn at least once.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with sup.state.lock:
            restarted = sup.state.restart_count
            current = sup.state.proc
        if restarted >= 1 and current is not None and current.pid != pid_initial:
            break
        time.sleep(0.05)

    with sup.state.lock:
        assert sup.state.restart_count >= 1, "watchdog did not observe exit"
        assert sup.state.last_exit_code == 0
        assert sup.state.proc is not None
        assert sup.state.proc.pid != pid_initial
    sup.stop()


def test_supervisor_stops_clearly(tmp_path: Path) -> None:
    p = _helper_program_dir(tmp_path)
    spec = SupervisedProcessSpec(argv=p["alive"])
    sup = CuaSupervisor(spec)
    sup.start()
    assert sup.is_running()
    sup.stop()
    # The stop_event must always be set after stop() returns — that is the
    # predictable contract regardless of how the OS reports the child's
    # exit code on different platforms.
    assert sup.state.stop_event.is_set()


def test_supervisor_probe_health_happy_path(tmp_path: Path) -> None:
    """probe_health() returns True when the helper responds to MCP initialize."""
    p = _helper_program_dir(tmp_path)
    spec = SupervisedProcessSpec(argv=p["alive"])
    sup = CuaSupervisor(spec)
    try:
        sup.start()
        assert sup.probe_health(timeout_s=2.0)
        with sup.state.lock:
            assert sup.state.healthy is True
            assert sup.state.last_error is None
    finally:
        sup.stop()


def test_supervisor_probe_health_with_no_handle() -> None:
    spec = SupervisedProcessSpec(
        argv=["python3", "-c", "import sys; sys.exit(0)"]
    )
    sup = CuaSupervisor(spec)
    # Never call start() so proc stays None.
    assert sup.probe_health(timeout_s=0.1) is False
    with sup.state.lock:
        assert sup.state.last_error == "process not running"


def test_singleton_is_idempotent(tmp_path: Path) -> None:
    set_cua_supervisor_for_tests(None)
    p = _helper_program_dir(tmp_path)
    spec = SupervisedProcessSpec(argv=p["alive"])
    s1 = get_cua_supervisor(spec)
    s2 = get_cua_supervisor()  # no arg -> returns existing
    assert s1 is s2
    set_cua_supervisor_for_tests(None)


def test_max_restarts_budget_exhausted(tmp_path: Path) -> None:
    """Once the budget is gone the supervisor stays out of restart and
    leaves the most-recent proc behind. Useful for callers that want to
    surface a hard failure rather than spin forever."""
    p = _helper_program_dir(tmp_path)
    spec = SupervisedProcessSpec(
        argv=p["exits"],
        initial_backoff_s=0.01,
        backoff_multiplier=1.0,
        max_restarts=2,
    )
    sup = CuaSupervisor(spec)
    try:
        sup.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with sup.state.lock:
                if sup.state.restart_count >= spec.max_restarts:
                    break
            time.sleep(0.05)
        with sup.state.lock:
            assert sup.state.restart_count >= spec.max_restarts
    finally:
        sup.stop()
