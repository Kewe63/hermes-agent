"""Tests for agent/post_response_hooks.py — post-response hook extension point.

Covers hook loading, system prompt injection, response validation,
security checks (world-writable refusal), max_nudges config, and
context schema — without hitting the network (all I/O is mocked).
"""

import os
import platform
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.post_response_hooks import (
    DEFAULT_MAX_NUDGES,
    Hook,
    build_system_prompt_additions,
    load_hooks,
    run_post_response_checks,
)


# ---------------------------------------------------------------------------
# Hook dataclass unit tests
# ---------------------------------------------------------------------------


class TestHookCheck:
    def test_check_passes_when_no_check_fn(self):
        hook = Hook(module_name="noop")
        assert hook.check("any response", {}) is True

    def test_check_delegates_to_check_fn(self):
        hook = Hook(module_name="gate", _check_fn=lambda r, c: "good" in r)
        assert hook.check("this is good", {}) is True
        assert hook.check("this is bad", {}) is False

    def test_check_catches_exception_and_returns_true(self):
        """Broken hook never crashes the agent — exception is swallowed."""
        def _boom(r, c):
            raise RuntimeError("boom")

        hook = Hook(module_name="broken", _check_fn=_boom)
        assert hook.check("anything", {}) is True

    def test_check_catches_type_error(self):
        """Hook returning non-bool-coercible value doesn't crash."""
        def _bad_return(r, c):
            raise TypeError("bad coercion")

        hook = Hook(module_name="bad_type", _check_fn=_bad_return)
        assert hook.check("anything", {}) is True


# ---------------------------------------------------------------------------
# load_hooks
# ---------------------------------------------------------------------------


class TestLoadHooks:
    def test_empty_config_returns_empty(self):
        assert load_hooks([]) == []

    def test_disabled_hook_is_skipped(self):
        configs = [{"module": "my_hook", "enabled": False}]
        assert load_hooks(configs) == []

    def test_missing_module_file_is_skipped(self, tmp_path):
        configs = [{"module": "nonexistent"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            (tmp_path / "hooks").mkdir()
            result = load_hooks(configs)
        assert result == []

    def test_module_without_hook_class_is_skipped(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "empty_mod.py").write_text("x = 1\n")

        configs = [{"module": "empty_mod"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)
        assert result == []

    def test_valid_hook_is_loaded(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "quality.py").write_text(
            "class Hook:\n"
            "    module_name = 'quality'\n"
            "    system_prompt_addition = 'Be thorough.'\n"
            "    nudge_message = 'Please elaborate.'\n"
            "    def check(self, response, context):\n"
            "        return len(response) > 10\n"
        )

        configs = [{"module": "quality"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        assert result[0].module_name == "quality"
        assert result[0].system_prompt_addition == "Be thorough."
        assert result[0].nudge_message == "Please elaborate."
        assert result[0].max_nudges == DEFAULT_MAX_NUDGES
        assert result[0].check("short", {}) is False
        assert result[0].check("this is a long enough response", {}) is True

    def test_invalid_config_entries_are_skipped(self):
        configs = ["not_a_dict", {"no_module_key": True}, None]
        assert load_hooks(configs) == []

    def test_multiple_hooks_ordering(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        for name in ("alpha", "beta"):
            (hooks_dir / f"{name}.py").write_text(
                f"class Hook:\n"
                f"    module_name = '{name}'\n"
                f"    system_prompt_addition = ''\n"
                f"    nudge_message = ''\n"
            )

        configs = [{"module": "alpha"}, {"module": "beta"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 2
        assert result[0].module_name == "alpha"
        assert result[1].module_name == "beta"

    def test_max_nudges_from_config(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "retry_hook.py").write_text(
            "class Hook:\n"
            "    module_name = 'retry_hook'\n"
        )

        configs = [{"module": "retry_hook", "max_nudges": 3}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        assert result[0].max_nudges == 3

    def test_max_nudges_invalid_value_uses_default(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "bad_cfg.py").write_text(
            "class Hook:\n"
            "    module_name = 'bad_cfg'\n"
        )

        configs = [{"module": "bad_cfg", "max_nudges": "not_a_number"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        assert result[0].max_nudges == DEFAULT_MAX_NUDGES

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix file permissions only")
    def test_world_writable_hook_is_refused(self, tmp_path):
        """Security: world-writable hook files are not loaded."""
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "unsafe.py"
        hook_file.write_text("class Hook:\n    module_name = 'unsafe'\n")
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IWOTH)

        configs = [{"module": "unsafe"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert result == []

    def test_hook_with_import_error_is_skipped(self, tmp_path):
        """Hook that raises on import doesn't crash the loader."""
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "crasher.py").write_text("raise ImportError('missing dep')\n")

        configs = [{"module": "crasher"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert result == []


# ---------------------------------------------------------------------------
# build_system_prompt_additions
# ---------------------------------------------------------------------------


class TestBuildSystemPromptAdditions:
    def test_empty_hooks(self):
        assert build_system_prompt_additions([]) == ""

    def test_aggregates_additions(self):
        hooks = [
            Hook(module_name="a", system_prompt_addition="Rule A."),
            Hook(module_name="b", system_prompt_addition=""),
            Hook(module_name="c", system_prompt_addition="Rule C."),
        ]
        result = build_system_prompt_additions(hooks)
        assert result == "Rule A.\n\nRule C."

    def test_single_addition(self):
        hooks = [Hook(module_name="solo", system_prompt_addition="Only rule.")]
        assert build_system_prompt_additions(hooks) == "Only rule."


# ---------------------------------------------------------------------------
# run_post_response_checks
# ---------------------------------------------------------------------------


class TestRunPostResponseChecks:
    def test_all_hooks_pass_returns_none(self):
        hooks = [
            Hook(module_name="a", _check_fn=lambda r, c: True),
            Hook(module_name="b", _check_fn=lambda r, c: True),
        ]
        assert run_post_response_checks(hooks, "response", {}) is None

    def test_first_failing_hook_wins(self):
        hooks = [
            Hook(module_name="pass", _check_fn=lambda r, c: True),
            Hook(module_name="fail1", nudge_message="Fix from fail1", _check_fn=lambda r, c: False),
            Hook(module_name="fail2", nudge_message="Fix from fail2", _check_fn=lambda r, c: False),
        ]
        result = run_post_response_checks(hooks, "response", {})
        assert result == "Fix from fail1"

    def test_failing_hook_without_nudge_uses_default(self):
        hooks = [
            Hook(module_name="strict", nudge_message="", _check_fn=lambda r, c: False),
        ]
        result = run_post_response_checks(hooks, "response", {})
        assert "strict" in result
        assert "quality check" in result

    def test_empty_hooks_returns_none(self):
        assert run_post_response_checks([], "response", {}) is None

    def test_context_contains_expected_keys(self):
        """Context dict schema: user_message, messages, model."""
        received = {}

        def _capture(r, c):
            received.update(c)
            return True

        hooks = [Hook(module_name="spy", _check_fn=_capture)]
        ctx = {
            "user_message": "hello",
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
        }
        run_post_response_checks(hooks, "response", ctx)
        assert received["user_message"] == "hello"
        assert received["model"] == "test-model"
        assert isinstance(received["messages"], list)

    def test_hook_exception_does_not_block_others(self):
        """Crashing hook is skipped (treated as pass), next hook still runs."""
        def _boom(r, c):
            raise ValueError("kaboom")

        hooks = [
            Hook(module_name="crasher", _check_fn=_boom),
            Hook(module_name="checker", nudge_message="check failed", _check_fn=lambda r, c: False),
        ]
        result = run_post_response_checks(hooks, "response", {})
        assert result == "check failed"
