"""A deploy ships only a commit its host code ran at, on both platforms.

MCPs board task 465689f5, b6eb30c8's R3. MCPs' maketools publish-executed
is tested there against the fleet queue's answers. What only this
repository can pin is the chain that makes its answer mean something here:
the deploy asks it for both execution projects, and each project's check
runs tools/fleet's host cases rather than something that passes without
them. That the fleet declares the two projects on their platforms is pinned
beside the registry's decoder, in tools/fleet's test_execution_projects. A
link lost from that chain passes every other check and stops the gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

#: The repository root: tools/maketools/tests/ is three levels below it.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def _recipe(path: str, target: str) -> list[str]:
    """The recipe lines of one target, in order.

    Args:
        path: Repository-relative Makefile path.
        target: The target whose rule line opens the recipe.

    Returns:
        Each tab-indented line after the rule, tab removed, up to the first
        line that is not one.
    """
    lines = (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    opening = [index for index, line in enumerate(lines) if line.startswith(f"{target}:")]
    assert len(opening) == 1, f"{path} has {len(opening)} rules for {target}"
    recipe: list[str] = []
    for line in lines[opening[0] + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line[1:])
    return recipe


def test_every_deploy_runs_the_gate_for_both_platforms() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "\ninfra: commit-tasks executed\n" in makefile
    assert _recipe("Makefile", "executed") == [
        "$(PYTHON) .githooks/published_maketools.py publish-executed ../MCPs "
        "tools/fleet-execution .",
        "$(PYTHON) .githooks/published_maketools.py publish-executed ../MCPs "
        "tools/fleet-execution-linux .",
    ]


def test_a_fleet_roll_moves_the_ref_only_after_both_suites_passed_at_head() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "\nfleet-roll: commit-tasks executed\n" in makefile
    assert _recipe("Makefile", "fleet-roll") == ["git update-ref refs/fleet/rolled HEAD"]


@pytest.mark.parametrize(
    ("script", "agent"),
    [("run-agent-tick.ps1", "fleet-agent"), ("run-node-agent-tick.ps1", "fleet-node-agent")],
)
def test_every_scheduled_tick_runs_the_rolled_commit(script: str, agent: str) -> None:
    text = (ROOT / "tools" / "fleet" / "scripts" / script).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    start = lines.index("$agentArguments = @(")
    assert lines[start + 1 : start + 5] == [
        "'run', '--', 'python', '-m', 'fleet.cli.rolled',",
        "'--repo-root', $apiRoot,",
        f"'--agent', '{agent}',",
        "'--',",
    ]
    # The registry is the roll's own; a tick naming one would run pinned code
    # against an unpinned configuration.
    assert "'--config'" not in text


def test_the_execution_project_runs_tools_fleet_host_cases() -> None:
    makefile = (ROOT / "tools" / "fleet-execution" / "Makefile").read_text(encoding="utf-8")
    assert "\ncheck: lint test\n" in makefile.replace("\r\n", "\n")
    assert _recipe("tools/fleet-execution/Makefile", "test") == ["$(MAKE) -C ../fleet execution"]
    assert _recipe("tools/fleet/Makefile", "execution") == [
        "poetry sync --with dev",
        "$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --no-cov",
    ]
