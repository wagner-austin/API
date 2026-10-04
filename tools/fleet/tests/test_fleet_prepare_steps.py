"""The registry's MCPs npm projects prepare through MCPs' fleet-prepare
(MCPs board task 74b13c20), decoded from the real fleet.json.

Every MCPs project that installs the npm workspace runs its install,
workspace build and (when it needs the test database) its database as three
``make -s -C scripts/fleet-prepare <phase>`` steps, each timed as the phase
it names, so a row of a tree the node has already prepared restores it
instead of rebuilding it. The directory each step names is the path the
runner's ``INSTALL_PATH_NOT_IN_COMMIT`` refusal asks the commit for, so a
commit from before the directory existed is refused by name rather than
failing inside make.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str

from fleet.contracts.source import InstallStep
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import export

PREPARE = ("make", "-s", "-C", "scripts/fleet-prepare")


def _steps() -> dict[str, tuple[InstallStep, ...]]:
    """Every sourced project's install steps, from the real registry."""
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    return {
        name: project["source"]["install"]
        for name, project in workspace["projects"].items()
        if project["source"] is not None
    }


def test_every_mcps_workspace_install_goes_through_fleet_prepare() -> None:
    prepared = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][:4] == PREPARE for step in steps)
    }
    unprepared = sorted(
        name
        for name, steps in _steps().items()
        if name.startswith("MCPs/")
        and any(step["argv"][-1] == "scripts/ci-build-consumed-workspaces.mjs" for step in steps)
        and name not in prepared
    )

    assert len(prepared) == 53
    assert unprepared == [
        "MCPs/execution",
        "MCPs/execution-deploy",
        "MCPs/execution-elevated",
        "MCPs/execution-linux",
    ]
    for steps in prepared.values():
        assert [step["phase"] for step in steps[:2]] == ["install", "workspace-build"]
        assert [step["argv"][4] for step in steps[:2]] == ["install", "workspace-build"]


def test_a_test_database_step_names_its_container_and_its_phase() -> None:
    databases = [
        step
        for steps in _steps().values()
        for step in steps
        if step["argv"][:4] == PREPARE and step["argv"][4] == "test-database"
    ]

    assert len(databases) == 27
    assert {step["phase"] for step in databases} == {"test-database"}
    assert {step["argv"][5:] for step in databases} == {("CONTAINER=corvis-fleet-testdb",)}


def test_each_prepare_step_names_the_directory_an_older_commit_lacks() -> None:
    named = {
        export.install_paths(step["argv"])
        for steps in _steps().values()
        for step in steps
        if step["argv"][:4] == PREPARE
    }

    assert named == {("scripts/fleet-prepare",)}
