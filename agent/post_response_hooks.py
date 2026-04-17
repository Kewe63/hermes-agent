"""
Post-Response Hook Extension Point
===================================

Lightweight hook system that lets users drop ``.py`` files into
``~/.hermes/hooks/`` to inspect (and optionally nudge) the agent's final
text response.  Each hook can:

  (a) inject a system prompt addition (pre-response guidance), and
  (b) run a ``check(response, context)`` gate that triggers a
      re-generation when it returns ``False``.

Security
--------
Hooks are **user-installed code** that runs with the same privileges as
the Hermes process.  This is the same trust model as the existing plugin
system (``~/.hermes/plugins/``) and skills (``~/.hermes/skills/``).
Only the machine owner can write to ``~/.hermes/hooks/``; hooks are
never auto-downloaded or auto-enabled.  The loader additionally verifies
that hook files are **not world-writable** (Unix) before importing them.

Context dict passed to ``check()``
-----------------------------------
``context`` is a dict with the following keys:

- ``user_message`` (str): The original user message for this turn.
- ``messages`` (list[dict]): Full conversation history up to this point.
- ``model`` (str): The model identifier currently in use.

See issue #11719 for the full design rationale.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Default maximum nudge attempts per response (configurable per-hook).
DEFAULT_MAX_NUDGES = 1


def _is_world_writable(path: Path) -> bool:
    """Return True if *path* is world-writable (Unix only)."""
    if _IS_WINDOWS:
        return False
    try:
        return bool(path.stat().st_mode & stat.S_IWOTH)
    except OSError:
        return False


@dataclass
class Hook:
    """A single post-response hook loaded from ``~/.hermes/hooks/<name>.py``."""

    module_name: str
    system_prompt_addition: str = ""
    nudge_message: str = ""
    max_nudges: int = DEFAULT_MAX_NUDGES
    _check_fn: Optional[Any] = field(default=None, repr=False)

    def check(self, response: str, context: Dict[str, Any]) -> bool:
        """Return ``True`` if the response passes, ``False`` to trigger nudge.

        If the check function raises an exception, it is logged and
        treated as a pass (``True``) so a broken hook never crashes the
        agent loop.
        """
        if self._check_fn is None:
            return True
        try:
            return bool(self._check_fn(response, context))
        except Exception as exc:
            logger.warning(
                "Hook '%s' check() raised %s: %s — treating as pass",
                self.module_name, type(exc).__name__, exc,
            )
            return True


def load_hooks(configs: List[Dict[str, Any]]) -> List[Hook]:
    """Load hooks from ``~/.hermes/hooks/`` based on *configs*.

    Each entry in *configs* is a dict like::

        {"module": "bottom_logic_check", "enabled": true, "max_nudges": 2}

    The corresponding file is ``~/.hermes/hooks/bottom_logic_check.py``.
    The module must expose a ``Hook`` class with the interface described
    in this module's docstring.

    Security: hook files that are world-writable (Unix) are refused to
    prevent privilege escalation via shared-machine scenarios.
    """
    if not configs:
        return []

    hooks_dir = get_hermes_home() / "hooks"
    loaded: List[Hook] = []

    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        name = cfg.get("module", "")
        if not name:
            continue
        if not cfg.get("enabled", True):
            logger.debug("Hook '%s' is disabled — skipping", name)
            continue

        hook_file = hooks_dir / f"{name}.py"
        if not hook_file.is_file():
            logger.warning(
                "Hook module '%s' not found at %s — skipping", name, hook_file,
            )
            continue

        if _is_world_writable(hook_file):
            logger.warning(
                "Hook '%s' is world-writable (%s) — refusing to load for "
                "security. Run: chmod o-w %s", name, hook_file, hook_file,
            )
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"hermes_hooks.{name}", str(hook_file),
            )
            if spec is None or spec.loader is None:
                logger.warning("Could not create module spec for hook '%s'", name)
                continue

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            hook_cls = getattr(mod, "Hook", None)
            if hook_cls is None:
                logger.warning(
                    "Hook module '%s' has no Hook class — skipping", name,
                )
                continue

            instance = hook_cls()

            _max_nudges = cfg.get("max_nudges", DEFAULT_MAX_NUDGES)
            try:
                _max_nudges = max(1, int(_max_nudges))
            except (TypeError, ValueError):
                _max_nudges = DEFAULT_MAX_NUDGES

            hook = Hook(
                module_name=getattr(instance, "module_name", name),
                system_prompt_addition=getattr(instance, "system_prompt_addition", "") or "",
                nudge_message=getattr(instance, "nudge_message", "") or "",
                max_nudges=_max_nudges,
                _check_fn=getattr(instance, "check", None),
            )
            loaded.append(hook)
            logger.info("Loaded post-response hook: %s (max_nudges=%d)", name, _max_nudges)
        except Exception as exc:
            logger.warning(
                "Failed to load hook '%s': %s — skipping", name, exc,
            )

    return loaded


def build_system_prompt_additions(hooks: List[Hook]) -> str:
    """Aggregate ``system_prompt_addition`` from all loaded hooks."""
    parts = [h.system_prompt_addition for h in hooks if h.system_prompt_addition]
    return "\n\n".join(parts)


def run_post_response_checks(
    hooks: List[Hook],
    response: str,
    context: Dict[str, Any],
) -> Optional[str]:
    """Run each hook's ``check()``; return the first nudge message or ``None``.

    Hook ordering follows the config list order.  The first failing hook
    wins — its ``nudge_message`` is returned so the agent loop can inject
    it as a system-level re-generation prompt.
    """
    for hook in hooks:
        if not hook.check(response, context):
            logger.info(
                "Post-response hook '%s' failed — nudging", hook.module_name,
            )
            return hook.nudge_message or (
                f"Your response did not pass the '{hook.module_name}' quality check. "
                "Please revise and provide a more thorough response."
            )
    return None
