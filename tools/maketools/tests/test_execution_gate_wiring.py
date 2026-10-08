"""A deploy ships only a commit its host code ran at, on both platforms.

MCPs board task 465689f5, b6eb30c8's R3. MCPs' maketools publish-executed
is tested there against the fleet queue's answers. What only this
repository can pin is the chain that makes its answer mean something here:
the deploy asks it for both execution projects, and each project's check
runs tools/fleet's host cases rather than something that passes without
them. That the fleet declares the two projects on their platforms is pinned
beside the registry's decoder, in tools/fleet's test_execution_projects. A
service deploy also asks for tools/maketools-execution, whose check runs the
real compose deploy of each covered service. A link lost from that chain
passes every other check and stops the gate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_dict, require_int

from tests.test_service_entry import SERVICES

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
    assert "\ninfra: commit-tasks executed services-executed\n" in makefile
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


def test_every_scheduled_tick_runs_the_rolled_commit() -> None:
    # Every fleet task's action is poetry running fleet.cli.tick (MCPs board
    # task 94ac1c4f), and the tick runs its agent only through the rolled
    # launcher, for both the hub's agent and a node's.
    schedule = (ROOT / "tools" / "fleet" / "scripts" / "FleetSchedule.ps1").read_text(
        encoding="utf-8"
    )
    assert '$line = "run -- python -m fleet.cli.tick --api-root ' in schedule
    tick = (ROOT / "tools" / "fleet" / "src" / "fleet" / "cli" / "tick.py").read_text(
        encoding="utf-8"
    )
    assert '(sys.executable, "-m", "fleet.cli.rolled", *plan["arguments"])' in tick
    for agent in ("fleet-agent", "fleet-node-agent"):
        assert len(list(re.finditer(rf'rolled_cli\.AGENT_FLAG,\s+"{agent}",', tick))) == 1
    # The registry is the roll's own; a tick naming one would run pinned code
    # against an unpinned configuration.
    assert "--config" not in tick


def test_every_service_deploy_asks_for_a_passed_deploy_run_at_head() -> None:
    assert _recipe("Makefile", "services-executed") == [
        "$(PYTHON) .githooks/published_maketools.py publish-executed ../MCPs "
        "tools/maketools-execution .",
    ]
    makefile = (ROOT / "tools" / "maketools-execution" / "Makefile").read_text(encoding="utf-8")
    assert "\n_check-unbudgeted: lint test\n" in makefile.replace("\r\n", "\n")
    assert _recipe("tools/maketools-execution/Makefile", "test") == [
        "make -C ../maketools execution"
    ]
    assert _recipe("tools/maketools/Makefile", "execution") == [
        "poetry sync --with dev",
        "$(PYTHON) scripts/run.py test --host-execution --no-cov --dist load --durations=0",
    ]


def test_the_execution_project_runs_tools_fleet_host_cases() -> None:
    makefile = (ROOT / "tools" / "fleet-execution" / "Makefile").read_text(encoding="utf-8")
    assert "\n_check-unbudgeted: lint test\n" in makefile.replace("\r\n", "\n")
    assert _recipe("tools/fleet-execution/Makefile", "test") == ["make -C ../fleet execution"]
    assert _recipe("tools/fleet/Makefile", "execution") == [
        "poetry sync --with dev",
        "$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --no-cov",
    ]


def test_the_deploy_project_asks_the_fleet_for_one_worker_per_case() -> None:
    """Every deploy case runs at once, so the check takes the slowest one.

    MCPs board task 90c135ac. Each case is its own image build, mostly its
    pip install and image export: turkic-api took 190 s built alone on
    lavender-wsl's rootless daemon and 188 s beside three others, so the
    cases do not slow each other and the check's wall is the sum of its
    rounds. Four workers made two rounds, 374 s against the 300 s budget at
    API 9686df19d. The dispatcher grants at most the larger of a project's
    minimum_workers and half a node's spare cores, so the minimum is what
    puts every case in one round, and a node that cannot give that many
    refuses the job rather than running it into the budget.
    """
    workspace = narrow_json_to_dict(
        load_json_str((ROOT / "tools" / "fleet" / "fleet.json").read_text(encoding="utf-8"))
    )
    project = require_dict(require_dict(workspace, "projects"), "tools/maketools-execution")
    assert require_int(project, "minimum_workers") == len(SERVICES)
