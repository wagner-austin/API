"""Exactly the queue tools tools/fleet calls (MCPs board tasks cca52c93, 2b06dd4b).

On 2026-10-16 the hub tick held the hub lane 290 s calling a tool fleet-mcp
had stopped serving. MCPs' deploy now refuses to remove a tool a running
caller names; this suite is the caller's side of that contract. It reads
every ``call_mcp_tool`` call in this package's source from its syntax tree,
resolves the tool each one names, a literal or a module-level constant, and
holds the set to :data:`CALLED`. A call to any other tool, a removed one
included, fails here, and so does a tool name the scan cannot resolve, so a
computed name cannot hide a call.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import pytest

#: This package's source.
SOURCE: Final[Path] = Path(__file__).resolve().parents[1] / "src" / "fleet"

#: The function every queue call goes through.
CALL: Final[str] = "call_mcp_tool"

#: Every tool this package calls. WHEN THIS FAILS: a call was added or
#: removed. A new tool must be served by the deployed fleet-mcp before the
#: roll that calls it; a tool dropped here may then be removed from fleet-mcp,
#: whose deploy searches the rolled tree for it first.
CALLED: Final[frozenset[str]] = frozenset(
    {
        "dispatch_claim",
        "dispatch_get",
        "dispatch_list",
        "dispatch_publisher_tick",
        "dispatch_registry_publish",
        "dispatch_report",
        "dispatch_tick",
        "task_post",
        "task_session_observe",
        "task_sessions_list",
    }
)

#: How many call sites the source holds, so a scan that found none cannot pass.
#: The 17th is :func:`fleet.core.queue_lost.report_lost`'s ``dispatch_report``
#: with action ``lost``, which fleet-mcp serves since MCPs 60a8dbb05.
CALL_SITES: Final[int] = 18


def constants(tree: ast.Module) -> dict[str, str]:
    """The module's top-level names bound to a string literal."""
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            first = node.targets[0]
            if not isinstance(first, ast.Name):
                continue
            target, value = first.id, node.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found[target] = value.value
    return found


def called_tools(source: str, where: str) -> list[str]:
    """The tool each ``call_mcp_tool`` call in one module names.

    Raises:
        ValueError: When a call's tool is neither a literal nor a module constant.
    """
    tree = ast.parse(source)
    bound = constants(tree)
    tools: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        match node.func:
            case ast.Name(id=name) | ast.Attribute(attr=name) if name == CALL:
                pass
            case _:
                continue
        given = [kw.value for kw in node.keywords if kw.arg == "tool"] or node.args[2:3]
        match given:
            case [ast.Constant(value=str() as literal)]:
                tools.append(literal)
            case [ast.Name(id=constant)] if constant in bound:
                tools.append(bound[constant])
            case _:
                unread = f"{CALL} names a tool this scan cannot read"
                raise ValueError(f"{where}:{node.lineno}: {unread}")
    return tools


def test_the_package_calls_exactly_the_declared_tools() -> None:
    modules = sorted(SOURCE.rglob("*.py"))
    sites = [
        tool
        for module in modules
        for tool in called_tools(module.read_text(encoding="utf-8"), str(module))
    ]

    assert len(modules) > 100
    assert len(sites) == CALL_SITES
    assert set(sites) == CALLED


def test_a_literal_a_constant_and_a_keyword_are_each_read() -> None:
    source = (
        'PUBLISH: Final = "dispatch_registry_publish"\n'
        'OTHER = "dispatch_get"\n'
        "x, y = 1, 2\n"
        "LIMIT = 3\n"
        "def f(post, creds):\n"
        '    call_mcp_tool(post, creds, "dispatch_claim", {})\n'
        "    mcp_client.call_mcp_tool(post, creds, PUBLISH, {})\n"
        '    call_mcp_tool(post, creds, arguments={}, tool="task_post")\n'
        "    other(post, creds, PUBLISH, {})\n"
    )

    assert called_tools(source, "m.py") == [
        "dispatch_claim",
        "dispatch_registry_publish",
        "task_post",
    ]


@pytest.mark.parametrize(
    "call",
    [
        "call_mcp_tool(post, creds, tool_name, {})",
        "call_mcp_tool(post, creds, PREFIX + 'claim', {})",
        "call_mcp_tool(post, creds)",
        "call_mcp_tool(post, creds, LIMIT, {})",
    ],
)
def test_a_tool_the_scan_cannot_read_is_refused(call: str) -> None:
    with pytest.raises(ValueError, match=f"m.py:3: {CALL} names a tool this scan cannot read"):
        called_tools(f"LIMIT = 3\nPREFIX = 'dispatch_'\n{call}\n", "m.py")
