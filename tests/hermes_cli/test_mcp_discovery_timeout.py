"""Regression tests for wait_for_mcp_discovery timeout override — fixes #37013.

Before the fix, hermes -z used the hardcoded 0.75 s default which is too short
for container/stdio-backed MCP servers with a real process cold-start (~1-2 s).
The fix adds HERMES_MCP_DISCOVERY_WAIT env var and mcp.discovery_wait config key,
and raises the headless caller default to 5.0 s.
"""

import threading
import time

import pytest

import hermes_cli.mcp_startup as mcp_startup


def _reset():
    mcp_startup._mcp_discovery_thread = None
    mcp_startup._mcp_discovery_started = False


def _make_slow_thread(delay: float):
    """Return a started thread that sleeps for delay seconds."""
    def _work():
        time.sleep(delay)
    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t


class TestEnvVarOverride:

    def test_env_var_extends_timeout(self, monkeypatch):
        """HERMES_MCP_DISCOVERY_WAIT must override the caller's default."""
        _reset()
        monkeypatch.setenv("HERMES_MCP_DISCOVERY_WAIT", "3.0")
        # fast thread finishes well within 3 s
        mcp_startup._mcp_discovery_thread = _make_slow_thread(0.05)
        t0 = time.monotonic()
        mcp_startup.wait_for_mcp_discovery(timeout=0.75)
        elapsed = time.monotonic() - t0
        # thread finished; we should NOT have waited the full 3 s
        assert elapsed < 2.0

    def test_env_var_zero_skips_wait(self, monkeypatch):
        """HERMES_MCP_DISCOVERY_WAIT=0 must return immediately."""
        _reset()
        monkeypatch.setenv("HERMES_MCP_DISCOVERY_WAIT", "0")
        mcp_startup._mcp_discovery_thread = _make_slow_thread(5.0)
        t0 = time.monotonic()
        mcp_startup.wait_for_mcp_discovery(timeout=0.75)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5

    def test_invalid_env_var_falls_back_to_caller_default(self, monkeypatch):
        """Non-float HERMES_MCP_DISCOVERY_WAIT must be ignored silently."""
        _reset()
        monkeypatch.setenv("HERMES_MCP_DISCOVERY_WAIT", "notanumber")
        mcp_startup._mcp_discovery_thread = _make_slow_thread(5.0)
        t0 = time.monotonic()
        mcp_startup.wait_for_mcp_discovery(timeout=0.1)
        elapsed = time.monotonic() - t0
        # should use 0.1 s fallback, not hang
        assert elapsed < 1.0

    def test_no_env_var_uses_caller_default(self, monkeypatch):
        """Without env var, caller's timeout argument must be respected."""
        _reset()
        monkeypatch.delenv("HERMES_MCP_DISCOVERY_WAIT", raising=False)
        mcp_startup._mcp_discovery_thread = _make_slow_thread(5.0)
        t0 = time.monotonic()
        mcp_startup.wait_for_mcp_discovery(timeout=0.1)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0


class TestHeadlessDefault:

    def test_headless_caller_passes_5s(self, monkeypatch):
        """cli.py get_tool_definitions headless path must call wait_for_mcp_discovery(timeout=5.0)
        while a discovery thread is in-flight, proving the real caller is exercised."""
        _reset()
        monkeypatch.delenv("HERMES_MCP_DISCOVERY_WAIT", raising=False)

        # Put an in-flight discovery thread in place so wait_for_mcp_discovery
        # actually blocks and the timeout value is meaningful.
        mcp_startup._mcp_discovery_thread = _make_slow_thread(0.05)
        mcp_startup._mcp_discovery_started = True

        received = []

        def _fake_wait(timeout=0.75):
            received.append(timeout)

        # Patch on the module that cli.py imports from at call time.
        monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", _fake_wait)

        # Also patch model_tools so get_tool_definitions doesn't need the full stack.
        import types
        fake_model_tools = types.ModuleType("model_tools")
        fake_model_tools.get_tool_definitions = lambda *a, **kw: []
        monkeypatch.setitem(__import__("sys").modules, "model_tools", fake_model_tools)

        import cli
        # Reload so cli.py picks up the patched sys.modules entry.
        import importlib
        importlib.reload(cli)

        cli.get_tool_definitions()

        assert received == [5.0], (
            f"cli.get_tool_definitions() headless path should pass timeout=5.0, got {received}"
        )
