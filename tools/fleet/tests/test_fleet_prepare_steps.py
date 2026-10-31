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

from fleet.contracts.source import InstallStep, ProjectSource
from fleet.contracts.tags import NodeTag
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
        and any("scripts/ci-build-consumed-workspaces.mjs" in step["argv"] for step in steps)
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
    # suite's served lane (MCPs board task b456a3cb); 60 since that lane is
    # one project per search host, MCPs/execution-served-diphtheria and
    # -colossus (MCPs board task bf7a9460); 61 since session-audit's real
    # PostgreSQL review-routing cases (MCPs board task 55189157); 60 since
    # MCPs/pc, which named a directory MCPs deleted, was dropped (MCPs board
    # task 60458a21); 61 since MCPs/packages/session-audit-linux, the
    # package's Linux lane (MCPs board task 31e0a1eb); 62 since
    # MCPs/execution-netboot-diphtheria, the execution suite's netboot lane
    # (MCPs board task d387271d); 63 since MCPs/execution-twin, the
    # execution suite's twin lane (MCPs board task 83b5c38f); 62 since
    # session-audit's Windows lane, MCPs/packages/session-audit, was dropped
    # with the package's Windows code (MCPs board task c8cb0ac5); 63 since
    # MCPs/kvm-mcp, the hardware KVM tools (MCPs board task c8f09479); 64
    # since MCPs/instagram-mcp, the creator tools (MCPs board task f986db61);
    # 65 since MCPs/execution-deployed-diphtheria, the execution suite's
    # deployed lane (MCPs board task 03b683d2), named for demure since the
    # database moved there (MCPs board task 43fa3a6c); 66 since
    # MCPs/execution-egress-diphtheria, the execution suite's egress lane
    # (MCPs board task 523a7445); 67 since MCPs/execution-served-demure,
    # the served lane that reads demure's provisioning as the stack host
    # (MCPs board task c036582e); 68 since MCPs/execution-lead-lacrymal,
    # the lead lane that runs one real planning pass (MCPs board task
    # 770837b7). The unprepared lose MCPs/execution-elevated, whose
    # installer cases MCPs board task 42fb4e26 deleted, and MCPs/execution,
    # the Windows lane MCPs board task 5b040fdb deleted.
    # 70 since execution-deploy and execution-linux share cached multi-root
    # preparation (MCPs board task 007bfc7e).
    assert len(prepared) == 70
    assert prepared["MCPs/instagram-mcp"] == (
        {"phase": "install", "argv": (*PREPARE, "install")},
        {"phase": "workspace-build", "argv": (*PREPARE, "workspace-build")},
    )
    assert unprepared == ["MCPs/phone-release-execution"]
    # Board task 007bfc7e: these lanes run execution and its nested proxy
    # twin. The existing planner includes their dependencies and generated
    # outputs; a bare invocation builds consumers from the entire repository.
    for name in ("MCPs/execution-deploy", "MCPs/execution-linux"):
        assert prepared[name] == (
            InstallStep(phase="install", argv=(*PREPARE, "install")),
            InstallStep(
                phase="workspace-build",
                argv=(*PREPARE, "workspace-build", "ROOTS=execution mcp-proxy"),
            ),
        )
    assert prepared["MCPs/packages/wiki-check"] == (
        InstallStep(phase="install", argv=(*PREPARE, "install")),
        InstallStep(
            phase="workspace-build",
            argv=(*PREPARE, "workspace-build", "ROOTS=packages/wiki-check packages/wiki-search"),
        ),
        InstallStep(phase="peer-wikis", argv=(*PREPARE, "peer-wikis")),
    )
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
    # MCPs/brief-generator (MCPs board task b1089284); 30 since
    # MCPs/packages/session-audit (MCPs board task 55189157); 31 since its
    # Linux lane, MCPs/packages/session-audit-linux (MCPs board task 31e0a1eb);
    # 30 again since the package dropped its Windows lane (MCPs board task
    # c8cb0ac5); 31 since MCPs/execution-lead-lacrymal (MCPs board task
    # 770837b7); 32 since MCPs/execution-served-diphtheria, for its
    # usage-limit case (MCPs board task ea46a7a4).
    assert len(databases) == 32
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


def test_a_served_lane_step_runs_last_and_only_for_the_served_projects() -> None:
    """MCPs fleet-prepare's served-lane phase writes the marker that makes the
    execution workspace collect tests/served alone (MCPs board tasks b456a3cb
    and bf7a9460). Those cases read the host's own search container, so no
    project a deploy waits on may carry the step, and it must follow install
    and workspace-build, which store the tree's untracked files for the
    node's next row."""
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:] == ("served-lane",) for step in steps)
    }

    assert sorted(marking) == [
        "MCPs/execution-served-colossus",
        "MCPs/execution-served-demure",
        "MCPs/execution-served-diphtheria",
    ]
    # The gateway host's lane migrates corvis_test first, for its usage-limit
    # case (MCPs board task ea46a7a4); the marker still comes last.
    assert {name: [step["phase"] for step in steps] for name, steps in marking.items()} == {
        "MCPs/execution-served-colossus": ["install", "workspace-build", "served-lane"],
        "MCPs/execution-served-demure": ["install", "workspace-build", "served-lane"],
        "MCPs/execution-served-diphtheria": [
            "install",
            "workspace-build",
            "test-database",
            "served-lane",
        ],
    }


def test_a_twin_lane_step_runs_last_and_only_for_the_docker_lane_twin_project() -> None:
    """MCPs fleet-prepare's twin-lane phase writes the marker that makes the
    execution workspace collect tests/passkey-twin.test.ts alone (MCPs board
    task 83b5c38f): a push touching one of mcp-proxy's Keycloak twins queues
    that row and reads the twins' own cases from its log. The case starts
    Keycloak and mailpit, so its row requires the docker tag, which runs its
    build as execdocker against that user's rootless daemon, and the step
    follows install and workspace-build, which store the tree's untracked
    files for the node's next row."""
    workspace = decode_fleet_workspace(shipped_document())
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:] == ("twin-lane",) for step in steps)
    }

    assert sorted(marking) == ["MCPs/execution-twin"]
    project = workspace["projects"]["MCPs/execution-twin"]
    assert (project["required_tags"], project["node"]) == ((NodeTag.LINUX, NodeTag.DOCKER), None)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/MCPs.git",
        path="execution",
        install=tuple(
            InstallStep(phase=phase, argv=(*PREPARE, phase))
            for phase in ("install", "workspace-build", "twin-lane")
        ),
        companions=(),
    )


def test_a_deployed_lane_step_runs_last_and_only_for_the_deployed_project() -> None:
    """MCPs fleet-prepare's deployed-lane phase writes the marker that makes
    the execution workspace collect tests/deployed alone (MCPs board task
    03b683d2). That case reads the database host's newest deploy against
    what it runs, so no project a deploy waits on may carry the step, and it
    follows install and workspace-build, which store the tree's untracked
    files for the node's next row."""
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:] == ("deployed-lane",) for step in steps)
    }

    assert sorted(marking) == ["MCPs/execution-deployed-demure"]
    for steps in marking.values():
        assert [step["phase"] for step in steps] == ["install", "workspace-build", "deployed-lane"]


def test_an_egress_lane_step_runs_last_and_only_for_the_egress_project() -> None:
    """MCPs fleet-prepare's egress-lane phase writes the marker that makes
    the execution workspace collect tests/egress alone (MCPs board task
    523a7445). That case reaches ggcity.org from inside the stack's
    containers and stays red while the site drops the host's address, so no
    other project may carry the step, and it follows install and
    workspace-build, which store the tree's untracked files for the node's
    next row."""
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:] == ("egress-lane",) for step in steps)
    }

    assert sorted(marking) == ["MCPs/execution-egress-diphtheria"]
    for steps in marking.values():
        assert [step["phase"] for step in steps] == ["install", "workspace-build", "egress-lane"]


def test_a_lead_lane_step_runs_last_and_only_for_the_lead_project() -> None:
    """MCPs fleet-prepare's lead-lane phase builds session-audit's venv and
    writes the marker that makes the execution workspace collect tests/lead
    alone (MCPs board task 770837b7, A1). That case runs one real planning
    pass, a ``claude -p`` on the session host that spends model usage, so no
    other project may carry the step, and it follows the test database the
    pass's board reads."""
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:] == ("lead-lane",) for step in steps)
    }

    assert sorted(marking) == ["MCPs/execution-lead-lacrymal"]
    for steps in marking.values():
        assert [step["phase"] for step in steps] == [
            "install",
            "workspace-build",
            "test-database",
            "lead-lane",
        ]


def test_a_netboot_lane_step_runs_last_and_only_for_the_netboot_project() -> None:
    """MCPs fleet-prepare's netboot-lane phase copies the host's Tailscale API
    key into the export and writes the marker that makes the execution
    workspace collect tests/netboot alone (MCPs board task d387271d). That
    case mints a tailnet key and installs a VM, so no project a deploy waits
    on may carry the step, and it runs after install and workspace-build,
    which store the tree's untracked files for the node's next row, so the
    key is never stored with them."""
    marking = {
        name: steps
        for name, steps in _steps().items()
        if any(step["argv"][4:5] == ("netboot-lane",) for step in steps)
    }

    assert sorted(marking) == ["MCPs/execution-netboot-diphtheria"]
    for steps in marking.values():
        assert [step["phase"] for step in steps] == ["install", "workspace-build", "netboot-lane"]


def test_each_prepare_step_names_the_directory_an_older_commit_lacks() -> None:
    named = {
        export.install_paths(step["argv"])
        for steps in _steps().values()
        for step in steps
        if step["argv"][:4] == PREPARE
    }

    assert named == {("scripts/fleet-prepare",)}
