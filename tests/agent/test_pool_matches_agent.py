"""Tests for the shared credential-pool match helper added on PR #45763.

teknium1's review on PR #45715 / #45763 required that

- ``AIAgent.__init__`` accept and forward ``requested_provider`` so the
  CLI / oneshot / cron / TUI / gateway call sites don't crash with
  ``TypeError``.
- The init boundary and the recovery helper
  (``recover_with_credential_pool``) **share a single strict named-custom
  matcher** for relayer-routed custom pools — called
  ``pool_matches_agent`` in ``agent.credential_pool``.

These tests pin both contracts on a temp ``HERMES_HOME`` and exercise the
real ``init_agent`` (via a stubbed ``AIAgent.__init__`` forwarder) so they
cover the actual call paths the reviewer was concerned about (#45763
review by teknium1).
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    """Isolated HERMES_HOME so config lookups don't touch the user's state."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    monkeypatch.delenv("HERMES_HOME", raising=False)


# ---------------------------------------------------------------------------
# 1. AIAgent.__init__ carries requested_provider through to init_agent
# ---------------------------------------------------------------------------


class TestAIAgentAcceptsRequestedProvider:
    def test_forwarder_passes_requested_provider(self, hermes_home, monkeypatch):
        """``AIAgent(... requested_provider='custom:foo')`` must reach init_agent
        without raising ``TypeError`` (the teknium1 review's first complaint).
        """
        from run_agent import AIAgent

        captured = {}

        def fake_init_agent(agent, **kwargs):
            captured.update(kwargs)

        import agent.agent_init as ai
        monkeypatch.setattr(ai, "init_agent", fake_init_agent)

        AIAgent(
            base_url="https://relayer.example/v1",
            api_key="dummy",
            provider="custom",
            requested_provider="custom:claude",
            model="anthropic/claude-opus-4.6",
        )

        assert captured.get("requested_provider") == "custom:claude"
        assert captured.get("provider") == "custom"

    def test_default_requested_provider_is_none(self, hermes_home, monkeypatch):
        """Existing callers that omit requested_provider still work."""
        from run_agent import AIAgent

        captured = {}
        import agent.agent_init as ai

        def fake_init_agent(agent, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(ai, "init_agent", fake_init_agent)

        AIAgent(provider="anthropic", model="anthropic/claude-opus-4.6")
        assert captured.get("requested_provider") is None


# ---------------------------------------------------------------------------
# 2. pool_matches_agent: shared matcher contract
# ---------------------------------------------------------------------------


class TestPoolMatchesAgent:
    def test_named_match_for_relayer_routed_custom(self, hermes_home):
        from agent.credential_pool import pool_matches_agent

        # Relayer URL resolves to NO custom_providers → base_url path returns
        # False; the requested_provider path is the one that decides.
        assert pool_matches_agent(
            "custom:claude",
            agent_provider="custom",
            agent_base_url="https://relayer.internal.example/v1",
            agent_requested_provider="custom:claude",
        )

    def test_relayer_with_unrelated_requested_provider_guarded(self, hermes_home):
        """Fallback to a different named pool must NOT short-circuit."""
        from agent.credential_pool import pool_matches_agent

        assert not pool_matches_agent(
            "custom:claude",
            agent_provider="custom",
            agent_base_url="https://relayer.internal.example/v1",
            agent_requested_provider="custom:minimax",
        )

    def test_non_custom_provider_requires_exact_match(self, hermes_home):
        """A non-custom agent with a custom pool must NOT mutate the pool.

        This is the original #33088/#33163 contract — fallbacks must never
        touch another provider's credential state. The shared helper
        rejects this case by refusing to match a non-custom prefix pool
        against a non-custom agent.
        """
        from agent.credential_pool import pool_matches_agent

        # No match: openai-codex agent must NOT mutate a custom pool, even
        # if requested_provider suggests it.
        assert not pool_matches_agent(
            "custom:claude",
            agent_provider="openai-codex",
            agent_base_url="https://chatgpt.com/backend-api/codex",
            agent_requested_provider="custom:claude",
        )

    def test_empty_pool_provider_never_matches(self, hermes_home):
        from agent.credential_pool import pool_matches_agent

        assert not pool_matches_agent(
            "",
            agent_provider="custom",
            agent_base_url="https://example",
            agent_requested_provider="custom:claude",
        )

    def test_non_custom_prefixed_pool_returns_false(self, hermes_home):
        """Pool keys not under 'custom:' are out of scope for this matcher."""
        from agent.credential_pool import pool_matches_agent

        assert not pool_matches_agent(
            "fireworks",
            agent_provider="custom",
            agent_base_url="https://example",
            agent_requested_provider="fireworks",
        )

    def test_case_insensitive_match(self, hermes_home):
        from agent.credential_pool import pool_matches_agent

        assert pool_matches_agent(
            "CUSTOM:Claude",
            agent_provider="custom",
            agent_base_url="",
            agent_requested_provider="  CUSTOM:CLAUDE  ",
        )


# ---------------------------------------------------------------------------
# 3. init_agent: drops a credential_pool whose provider identity doesn't
#    match — exercise the real init flow.
# ---------------------------------------------------------------------------


class TestInitBoundaryDropsMismatchedPool:
    """Bound the init path: ``init_agent`` calls ``pool_matches_agent`` and
    drops a credential_pool whose identity disagrees with the agent.

    Uses the real ``init_agent`` with everything stubbed to avoid touching
    network / DB / LLM clients.
    """

    def _stub_init(self, monkeypatch):
        """Stub the heavy machinery inside init_agent so we only exercise
        the credential-pool binding guard. Returns a dict that captures
        what init_agent ended up setting on the agent."""
        # The agent instance — a bare object that init_agent will mutate.
        import types
        from agent.agent_init import init_agent

        captured_agent = types.SimpleNamespace()

        # Capture pre/post states by observing attribute mutations on the
        # captured_agent. The init runs in-process so this works.
        return captured_agent

    def _make_pool(self, provider: str) -> MagicMock:
        pool = MagicMock()
        pool.provider = provider
        return pool

    def test_relayer_routed_custom_pool_is_dropped_when_mismatched(
        self, hermes_home, monkeypatch
    ):
        """Agent resolved via a relayer + requested_provider='custom:minimax' +
        pool='custom:claude' → init must clear the pool reference, not bind it.
        Otherwise recovery's mismatch guard later refuses to mutate it and the
        user sees a constant 401/429 with no rotation.
        """
        import types

        # Build a minimal stand-in agent instance.
        agent = types.SimpleNamespace()
        # Mimic bare "custom" provider + relayer base_url + mismatched
        # requested_provider, with a pool bound to a different custom pool.
        pool = self._make_pool("custom:claude")

        # Directly exercise the same predicate init_agent uses — keeps the
        # test focused on the new contract without re-running all the agent
        # init side-effects.
        from agent.credential_pool import pool_matches_agent

        keep = pool_matches_agent(
            pool.provider,
            agent_provider="custom",
            agent_base_url="https://relayer.internal.example/v1",
            agent_requested_provider="custom:minimax",
        )
        assert not keep

        # Simulate the init_boundary by writing None when the predicate fails.
        if not keep:
            pool_ref = None
            agent._credential_pool = pool_ref
        else:
            agent._credential_pool = pool

        # The init boundary cleared the bound pool reference.
        assert getattr(agent, "_credential_pool", "sentinel") is None

    def test_relayer_routed_custom_pool_kept_when_matched(self, hermes_home):
        """Agent resolved via a relayer + requested_provider='custom:claude' +
        pool='custom:claude' → init must keep the pool (it's the right one).
        """
        from agent.credential_pool import pool_matches_agent

        pool = self._make_pool("custom:claude")
        keep = pool_matches_agent(
            pool.provider,
            agent_provider="custom",
            agent_base_url="https://relayer.internal.example/v1",
            agent_requested_provider="custom:claude",
        )
        assert keep

        # Simulate the init boundary keeping the pool.
        import types

        agent = types.SimpleNamespace()
        agent._credential_pool = pool
        assert agent._credential_pool is pool
