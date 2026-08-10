"""Regression test for the asyncio.get_event_loop() deprecation fix in
hermes_cli/web_server.py.

Python 3.10+ deprecates ``asyncio.get_event_loop()`` when no running event
loop exists (DeprecationWarning; scheduled for removal). The single
remaining call site in this file is the MiniMax PKCE device-flow boot
at line ~L6173, where the synchronous closure ``_do_minimax_request`` is
handed to ``loop.run_in_executor`` from inside
``async def _start_device_code_flow``.

Drop-in replacement: ``asyncio.get_running_loop()``. The outer function
is async, so the running loop is guaranteed. The inner closure is
correctly sync — ``run_in_executor`` is the bridge from async caller to
sync worker.

These tests pin:
  - zero remaining ``asyncio.get_event_loop()`` sites in this file;
  - the migrated ``_start_device_code_flow`` body uses
    ``get_running_loop`` (presence);
  - the migrated site is inside ``async def _start_device_code_flow``
    (we walk back to the *outer* async def, ignoring inner sync
    closures like ``_do_minimax_request`` whose ``def`` is
    structurally nested but execution-wise irrelevant for the
    async-running-loop question);
  - the ``import time`` is still present (used by time.time() for
    expires_at math).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_SERVER = REPO_ROOT / "hermes_cli" / "web_server.py"


def _read_web_server() -> str:
    assert WEB_SERVER.is_file(), f"missing {WEB_SERVER}"
    return WEB_SERVER.read_text(encoding="utf-8")


def test_get_event_loop_removed() -> None:
    """No ``asyncio.get_event_loop()`` should remain in
    hermes_cli/web_server.py — the deprecation must be fully resolved
    in this file before siblings follow."""
    content = _read_web_server()
    matches = re.findall(r"asyncio\.get_event_loop\(", content)
    assert matches == [], (
        f"Found {len(matches)} remaining `asyncio.get_event_loop(` "
        f"call(s) in hermes_cli/web_server.py. The 3.10+ deprecation "
        f"must be fully resolved."
    )


def _find_outer_async_def(
    content: str, line_idx: int, max_lookback: int = 500
) -> tuple[int | None, str | None]:
    """Walk backwards from ``line_idx`` to find the *outer* ``async def``.
    Inner ``def`` (sync closures) are skipped — they are nested
    structural blocks but the loop-policy question is decided by the
    nearest enclosing async def.

    Returns (line_number_1based, function_name) or (None, None) if not
    found within ``max_lookback`` lines.
    """
    lines = content.split("\n")
    for i in range(line_idx - 1, max(0, line_idx - max_lookback), -1):
        m = re.match(r"^(\s*)(async )?def (\w+)", lines[i])
        if m:
            async_kw = m.group(2)
            name = m.group(3)
            if async_kw:
                return i + 1, name
    return None, None


def test_migrated_site_in_async_def() -> None:
    """The migrated ``asyncio.get_running_loop().run_in_executor(...)``
    call site at the MiniMax PKCE device-flow boot must sit inside
    ``async def _start_device_code_flow``. If the call moves to a
    sync caller the call would raise RuntimeError because there is
    no running loop in a sync context."""
    content = _read_web_server()
    lines = content.split("\n")

    # The migrated site is multi-line: `await ... .run_in_executor(`
    # spans two lines, and ``_do_minimax_request`` is the function arg
    # on the next line. Walk forward from each ``run_in_executor(`` start
    # and accept the call if a MiniMax marker appears in the next 5
    # lines.
    migrated_sites = []
    for i, line in enumerate(lines):
        if "await asyncio.get_running_loop().run_in_executor(" not in line:
            continue
        ctx = "\n".join(lines[i : min(i + 6, len(lines))])
        if "minimax" not in ctx.lower():
            continue
        migrated_sites.append(i)
    assert len(migrated_sites) >= 1, (
        "expected at least one `await asyncio.get_running_loop().run_in_executor` "
        "site whose next 5 lines mention `minimax` (the MiniMax PKCE boot — "
        "the only deprecation fix in this PR)."
    )

    idx = migrated_sites[0]
    owner_line, owner_name = _find_outer_async_def(content, idx)
    assert owner_name is not None and owner_line is not None, (
        f"migrated MiniMax site at line {idx + 1} has no outer async def "
        f"within 500 lines. _start_device_code_flow is async — this "
        f"site must live inside it."
    )
    assert owner_name == "_start_device_code_flow", (
        f"migrated MiniMax site at line {idx + 1} is inside `{owner_name}()`, "
        f"expected `_start_device_code_flow`."
    )
    decl = lines[owner_line - 1]
    assert decl.lstrip().startswith("async def"), (
        f"{owner_name} at line {owner_line} is not declared async: {decl.rstrip()!r}"
    )


def test_time_module_already_imported() -> None:
    """Sanity guard: ``time`` is used elsewhere in this file
    (e.g. time.time() at line ~6091 in _start_device_code_flow) so
    the import should already exist."""
    content = _read_web_server()
    assert re.search(r"^import time\b", content, re.MULTILINE), (
        "hermes_cli/web_server.py no longer imports `time` — the "
        "device-code flow uses time.time() to compute expires_at; "
        "removing the import would crash the flow."
    )
