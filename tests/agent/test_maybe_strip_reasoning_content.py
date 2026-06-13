"""Tests for ``_maybe_strip_reasoning_content`` in chat_completion_helpers.py.

This function strips ``reasoning_content`` from assistant messages when the
current provider does not support reasoning fields, preventing HTTP 400/422
on cross-provider fallback from thinking to non-thinking models.
"""

from unittest.mock import Mock, MagicMock, patch

import pytest

from agent.chat_completion_helpers import _maybe_strip_reasoning_content


def _make_agent(
    needs_thinking_pad: bool = False,
    supports_reasoning_extra: bool = False,
) -> MagicMock:
    """Build a mock AIAgent with the reasoning-capability knobs."""
    agent = MagicMock()
    agent._needs_thinking_reasoning_pad = Mock(return_value=needs_thinking_pad)
    agent._supports_reasoning_extra_body = Mock(return_value=supports_reasoning_extra)
    return agent


class TestStripReasoningContent:
    """Unit tests for _maybe_strip_reasoning_content."""

    def test_passthrough_when_no_reasoning_content(self):
        """Messages without reasoning_content pass through unchanged."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=False)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
        result = _maybe_strip_reasoning_content(agent, messages)
        assert result is messages  # same object, no deepcopy

    def test_strips_reasoning_content_for_non_reasoning_provider(self):
        """Non-reasoning provider gets reasoning_content stripped from assistant msgs."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=False)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "let me think", "reasoning_content": "I need to consider..."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}], "reasoning_content": " "},
        ]
        result = _maybe_strip_reasoning_content(agent, messages)
        assert result is not messages  # deep-copied
        assert "reasoning_content" not in result[1]
        assert "reasoning_content" not in result[2]
        assert result[1]["content"] == "let me think"
        assert result[2]["tool_calls"] == [{"id": "call_1"}]

    def test_keeps_reasoning_content_for_thinking_pad_provider(self):
        """DeepSeek/Kimi/MiMo (needs_thinking_reasoning_pad) keep reasoning_content."""
        agent = _make_agent(needs_thinking_pad=True, supports_reasoning_extra=False)
        messages = [
            {"role": "assistant", "content": "thinking...", "reasoning_content": "step by step"},
        ]
        result = _maybe_strip_reasoning_content(agent, messages)
        assert result is messages  # passthrough
        assert result[0]["reasoning_content"] == "step by step"

    def test_keeps_reasoning_content_for_reasoning_extra_body_provider(self):
        """Providers with reasoning extra_body (OpenAI o-series, Anthropic, etc.) keep it."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=True)
        messages = [
            {"role": "assistant", "content": "thinking...", "reasoning_content": "reasoning..."},
        ]
        result = _maybe_strip_reasoning_content(agent, messages)
        assert result is messages  # passthrough
        assert result[0]["reasoning_content"] == "reasoning..."

    def test_user_and_tool_messages_unaffected(self):
        """Only assistant messages get reasoning_content stripped."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=False)
        messages = [
            {"role": "user", "content": "hello", "reasoning_content": "nope"},
            {"role": "tool", "content": "result", "reasoning_content": "also nope"},
            {"role": "assistant", "content": "hi", "reasoning_content": "yes"},
        ]
        result = _maybe_strip_reasoning_content(agent, messages)
        assert result[0]["reasoning_content"] == "nope"  # user untouched
        assert result[1]["reasoning_content"] == "also nope"  # tool untouched
        assert "reasoning_content" not in result[2]  # assistant stripped

    def test_empty_messages_list(self):
        """Empty list returns unchanged."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=False)
        result = _maybe_strip_reasoning_content(agent, [])
        assert result == []

    def test_mixed_content_types_preserved(self):
        """Non-dict items in message list are left alone."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=False)
        messages = [
            {"role": "assistant", "content": "a", "reasoning_content": "r"},
            "not a dict",
            42,
        ]
        result = _maybe_strip_reasoning_content(agent, messages)
        assert "reasoning_content" not in result[0]
        assert result[1] == "not a dict"
        assert result[2] == 42

    def test_original_messages_not_mutated(self):
        """The input list and its dicts are never mutated in place."""
        agent = _make_agent(needs_thinking_pad=False, supports_reasoning_extra=False)
        original = [
            {"role": "assistant", "content": "hi", "reasoning_content": "thinking"},
        ]
        _maybe_strip_reasoning_content(agent, original)
        assert original[0]["reasoning_content"] == "thinking"  # original untouched


class TestStripReasoningContentThroughForwarder:
    """Integration-style: verify forwarder in run_agent.py routes correctly."""

    def test_forwarder_matches_implementation(self):
        """AIAgent._maybe_strip_reasoning_content calls the right helper."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        # Set minimal attrs the forwarder needs
        agent._needs_thinking_reasoning_pad = Mock(return_value=False)
        agent._supports_reasoning_extra_body = Mock(return_value=False)

        messages = [
            {"role": "assistant", "content": "hi", "reasoning_content": "thinking"},
        ]
        result = agent._maybe_strip_reasoning_content(messages)
        assert "reasoning_content" not in result[0]
        assert result[0]["content"] == "hi"
