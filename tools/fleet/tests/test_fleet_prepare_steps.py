"""The registry's MCPs npm projects prepare through MCPs' fleet-prepare
(MCPs board task 74b13c20), decoded from the real registry.

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

from fleet.contracts.source import InstallStep
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import export
from tests._registry import shipped_document

PREPARE = ("make", "-s", "-C", "scripts/fleet-prepare")


def _steps() -> dict[str, tuple[InstallStep, ...]]:
    """Every sourced project's install steps, from the real registry."""
    workspace = decode_fleet_workspace(shipped_document())
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

    # 54 since MCPs/sms-gateway joined (MCPs board task 1da15750); 55 since
    # MCPs/execution-testdb, the execution suite's testdb lane (MCPs board
    # task c04519f9, A2); 56 since MCPs/login-probe, the login real-use
    # check (MCPs board task 2ce5fc24); 55 since MCPs/dx-mod was archived
    # (MCPs board task 5d8b114b); 57 since MCPs/packages/wiki-check and
    # MCPs/brief-generator, which end with the peer-wikis phase (MCPs board
    # task b1089284); 58 since MCPs/print-mcp, the hub's print tools (MCPs
    # board task 9896d6b2); 59 since MCPs/execution-served, the execution
    # suite's served lane (MCPs board task b456a3cb).
    assert len(prepared) == 59
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

    # 28 since MCPs/execution-testdb (MCPs board task c04519f9, A2); 29 since
    # MCPs/brief-generator (MCPs board task b1089284).
    assert len(databases) == 29
    assert {step["phase"] for step in databases} == {"test-database"}
    assert {step["argv"][5:] for step in databases} == {("CONTAINER=corvis-fleet-testdb",)}


def test_a_peer_wikis_step_runs_last_beside_the_companions_it_places() -> None:
    """MCPs fleet-prepare's peer-wikis phase copies three companions from
    beside the export into the places docker-compose mounts them (MCPs board
    task b1089284). It must run after install and workspace-build, which store
    the tree's untracked files, and its row must declare the three, or the
    phase refuses FLEET_COMPANION_MISSING on the node."""
    workspace = decode_fleet_workspace(shipped_document())
    placing = {
        name: project["source"]
        for name, project in workspace["projects"].items()
        if project["source"] is not None
        and any(step["argv"][4:] == ("peer-wikis",) for step in project["source"]["install"])
    }

    assert sorted(placing) == ["MCPs/brief-generator", "MCPs/packages/wiki-check"]
    for source in placing.values():
        assert source["install"][-1]["phase"] == "peer-wikis"
        assert {"irvine-civic-wiki", "bookshelf", "chau"} <= {
            companion["directory"] for companion in source["companions"]
        }


def test_a_served_lane_step_runs_last_and_only_for_the_served_project() -> None:
    """MCPs fleet-prepare's served-lane phase writes the marker that makes the
    execution workspace collect tests/served alone (MCPs board task b456a3cb).
    Those cases read the public site, so no project a deploy waits on may
    carry the step, and it must follow install and workspace-build, which
    store the tree's untracked files for the node's next row."""
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:] == ("served-lane",) for step in steps)
    }

    assert sorted(marking) == ["MCPs/execution-served"]
    assert [step["phase"] for step in marking["MCPs/execution-served"]] == [
        "install",
        "workspace-build",
        "served-lane",
    ]


def test_each_prepare_step_names_the_directory_an_older_commit_lacks() -> None:
    named = {
        export.install_paths(step["argv"])
        for steps in _steps().values()
        for step in steps
        if step["argv"][:4] == PREPARE
    }

    assert named == {("scripts/fleet-prepare",)}
