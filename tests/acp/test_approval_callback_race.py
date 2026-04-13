"""Tests for task-scoped approval callback registry.

Verifies that concurrent ACP sessions cannot overwrite each other's
approval callbacks (the race condition fixed in this PR).
"""

import threading
import time
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Unit tests: terminal_tool task-scoped registry
# ---------------------------------------------------------------------------

class TestTaskApprovalRegistry:
    """register/unregister/get_task_approval_callback API."""

    def setup_method(self):
        from tools import terminal_tool as tt
        # Clear registry between tests
        with tt._get_task_approval_lock():
            tt._task_approval_callbacks.clear()

    def test_register_and_get(self):
        from tools import terminal_tool as tt
        cb = MagicMock()
        tt.register_task_approval_callback("session-A", cb)
        assert tt.get_task_approval_callback("session-A") is cb

    def test_unregister_removes_entry(self):
        from tools import terminal_tool as tt
        cb = MagicMock()
        tt.register_task_approval_callback("session-B", cb)
        tt.unregister_task_approval_callback("session-B")
        # Falls back to global _approval_callback (None by default)
        assert tt.get_task_approval_callback("session-B") is None

    def test_unknown_task_returns_global_fallback(self):
        from tools import terminal_tool as tt
        global_cb = MagicMock()
        original = tt._approval_callback
        try:
            tt._approval_callback = global_cb
            result = tt.get_task_approval_callback("nonexistent-session")
            assert result is global_cb
        finally:
            tt._approval_callback = original

    def test_sessions_are_isolated(self):
        """Two sessions get their own callbacks, not each other's."""
        from tools import terminal_tool as tt
        cb_a = MagicMock()
        cb_b = MagicMock()
        tt.register_task_approval_callback("session-A", cb_a)
        tt.register_task_approval_callback("session-B", cb_b)

        assert tt.get_task_approval_callback("session-A") is cb_a
        assert tt.get_task_approval_callback("session-B") is cb_b
        assert cb_a is not cb_b

    def test_second_register_overwrites_for_same_task(self):
        """Re-registering for the same task_id replaces the old callback."""
        from tools import terminal_tool as tt
        cb1 = MagicMock()
        cb2 = MagicMock()
        tt.register_task_approval_callback("session-X", cb1)
        tt.register_task_approval_callback("session-X", cb2)
        assert tt.get_task_approval_callback("session-X") is cb2

    def test_unregister_nonexistent_is_noop(self):
        from tools import terminal_tool as tt
        # Should not raise
        tt.unregister_task_approval_callback("does-not-exist")


# ---------------------------------------------------------------------------
# Concurrency tests: no race condition under parallel sessions
# ---------------------------------------------------------------------------

class TestApprovalCallbackConcurrency:
    """Parallel sessions must not interfere with each other's callbacks."""

    def setup_method(self):
        from tools import terminal_tool as tt
        with tt._get_task_approval_lock():
            tt._task_approval_callbacks.clear()

    def test_parallel_sessions_see_own_callback(self):
        """Spawn N threads each registering their own callback; they must
        always read back their own, never another session's."""
        from tools import terminal_tool as tt

        errors = []
        barrier = threading.Barrier(10)

        def session_worker(session_id: str):
            cb = MagicMock(name=f"cb-{session_id}")
            tt.register_task_approval_callback(session_id, cb)
            barrier.wait()  # all register before any reads
            # Read 100 times under contention
            for _ in range(100):
                result = tt.get_task_approval_callback(session_id)
                if result is not cb:
                    errors.append(f"{session_id}: expected own cb, got {result}")
            tt.unregister_task_approval_callback(session_id)

        threads = [
            threading.Thread(target=session_worker, args=(f"sess-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], "\n".join(errors)

    def test_global_callback_not_polluted_by_task_callbacks(self):
        """Registering task-scoped callbacks must not change _approval_callback."""
        from tools import terminal_tool as tt
        original_global = tt._approval_callback
        tt.register_task_approval_callback("s1", MagicMock())
        tt.register_task_approval_callback("s2", MagicMock())
        assert tt._approval_callback is original_global


# ---------------------------------------------------------------------------
# Server.py integration: register/unregister called, not set_approval_callback
# ---------------------------------------------------------------------------

class TestServerUsesTaskScopedAPI:
    """server.py prompt() must use register/unregister, not set_approval_callback."""

    def test_server_does_not_call_set_approval_callback(self):
        """AST check: prompt() in server.py must not call set_approval_callback."""
        import ast

        with open("acp_adapter/server.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "prompt":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute):
                            assert func.attr != "set_approval_callback", (
                                "prompt() must not call set_approval_callback() — "
                                "use register_task_approval_callback() instead"
                            )
                break

    def test_server_calls_register_task_approval_callback(self):
        """AST check: prompt() must call register_task_approval_callback."""
        import ast

        with open("acp_adapter/server.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found_register = False
        found_unregister = False

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "prompt":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute):
                            if func.attr == "register_task_approval_callback":
                                found_register = True
                            if func.attr == "unregister_task_approval_callback":
                                found_unregister = True
                break

        assert found_register, (
            "prompt() must call register_task_approval_callback()"
        )
        assert found_unregister, (
            "prompt() must call unregister_task_approval_callback() in finally"
        )

    def test_unregister_is_in_finally_block(self):
        """AST check: unregister_task_approval_callback must be in a finally block."""
        import ast

        with open("acp_adapter/server.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "prompt":
                for child in ast.walk(node):
                    if isinstance(child, ast.Try) and child.finalbody:
                        finally_text = "\n".join(
                            ast.dump(n) for n in child.finalbody
                        )
                        if "unregister_task_approval_callback" in finally_text:
                            return  # found in finally — test passes
                break

        raise AssertionError(
            "unregister_task_approval_callback must be called inside a finally block "
            "to guarantee cleanup even when the agent raises"
        )


# ---------------------------------------------------------------------------
# Functional: check_all_guards uses task-scoped callback
# ---------------------------------------------------------------------------

class TestCheckAllGuardsTaskScoped:
    """_check_all_guards must route through get_task_approval_callback."""

    def setup_method(self):
        from tools import terminal_tool as tt
        with tt._get_task_approval_lock():
            tt._task_approval_callbacks.clear()

    def test_check_all_guards_uses_task_callback(self):
        """_check_all_guards(task_id=X) must call session X's callback, not
        the global one."""
        from tools import terminal_tool as tt
        from tools.approval import check_all_command_guards as _impl

        session_cb = MagicMock(return_value="once")
        global_cb = MagicMock(return_value="deny")

        tt.register_task_approval_callback("my-session", session_cb)
        original_global = tt._approval_callback
        try:
            tt._approval_callback = global_cb
            # Patch impl so it doesn't do real detection; just call cb
            with patch("tools.terminal_tool._check_all_guards_impl") as mock_impl:
                mock_impl.return_value = {"approved": True}
                tt._check_all_guards("echo hi", "local", task_id="my-session")
                # The callback passed to the impl should be session_cb
                _, kwargs = mock_impl.call_args
                assert kwargs.get("approval_callback") is session_cb
        finally:
            tt._approval_callback = original_global
            tt.unregister_task_approval_callback("my-session")

    def test_check_all_guards_falls_back_for_unknown_task(self):
        """When task_id has no registered callback, fall back to global."""
        from tools import terminal_tool as tt

        global_cb = MagicMock(return_value="once")
        original_global = tt._approval_callback
        try:
            tt._approval_callback = global_cb
            with patch("tools.terminal_tool._check_all_guards_impl") as mock_impl:
                mock_impl.return_value = {"approved": True}
                tt._check_all_guards("echo hi", "local", task_id="unknown-session")
                _, kwargs = mock_impl.call_args
                assert kwargs.get("approval_callback") is global_cb
        finally:
            tt._approval_callback = original_global

    def test_check_all_guards_no_task_id_uses_global(self):
        """When task_id is empty string, global callback is used."""
        from tools import terminal_tool as tt

        global_cb = MagicMock(return_value="once")
        original_global = tt._approval_callback
        try:
            tt._approval_callback = global_cb
            with patch("tools.terminal_tool._check_all_guards_impl") as mock_impl:
                mock_impl.return_value = {"approved": True}
                tt._check_all_guards("echo hi", "local")  # no task_id
                _, kwargs = mock_impl.call_args
                assert kwargs.get("approval_callback") is global_cb
        finally:
            tt._approval_callback = original_global
