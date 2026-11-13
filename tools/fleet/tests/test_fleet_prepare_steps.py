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

from collections.abc import Mapping

from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
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
    # preparation (MCPs board task 007bfc7e). 71 since MCPs/participant-mcp,
    # the chat app's subscription participants (MCPs board task 42512a6b).
    # 72 since the served lane follows control onto Lacrymal (MCPs board task 927bbd25).
    # 73 since media-review-mcp joined (MCPs board task c33e4c11).
    # 74 since the CI contracts joined (MCPs board task cf418418).
    # 75 since mediawiki-mcp joined (MCPs board task 9a49a297).
    assert len(prepared) == 75
    assert prepared["MCPs/scripts/ci-contracts"] == (
        InstallStep(phase="install", argv=(*PREPARE, "install")),
        InstallStep(
            phase="workspace-build", argv=(*PREPARE, "workspace-build", "ROOTS=mcp-shared")
        ),
    )
    assert prepared["MCPs/media-review-mcp"] == (
        {"phase": "install", "argv": (*PREPARE, "install")},
        {"phase": "workspace-build", "argv": (*PREPARE, "workspace-build")},
    )
    assert prepared["MCPs/instagram-mcp"] == (
        {"phase": "install", "argv": (*PREPARE, "install")},
        {"phase": "workspace-build", "argv": (*PREPARE, "workspace-build")},
        InstallStep(
            phase="test-database",
            argv=(*PREPARE, "test-database", "CONTAINER=corvis-fleet-testdb"),
        ),
    )
    assert unprepared == ["MCPs/phone-release-execution"]
    # Board task 007bfc7e: these lanes run execution and its nested proxy
    # twin. The existing planner includes their dependencies and generated
    # outputs; a bare invocation builds consumers from the entire repository.
    lanes_build = InstallStep(
        phase="workspace-build", argv=(*PREPARE, "workspace-build", "ROOTS=execution:mcp-proxy")
    )
    assert prepared["MCPs/execution-linux"] == (
        InstallStep(phase="install", argv=(*PREPARE, "install")),
        lanes_build,
    )
    # MCPs board task 1e0ee821: the deploy lane starts its rehearsals' image
    # build between the two and goes on while it builds, so the build runs
    # beside the workspace build, the lint and the suite's first files.
    assert prepared["MCPs/execution-deploy"] == (
        InstallStep(phase="install", argv=(*PREPARE, "install")),
        InstallStep(phase="rehearsal-images", argv=(*PREPARE, "rehearsal-images")),
        lanes_build,
    )
    assert prepared["MCPs/packages/wiki-check"] == (
        InstallStep(phase="install", argv=(*PREPARE, "install")),
        InstallStep(
            phase="workspace-build",
            argv=(*PREPARE, "workspace-build", "ROOTS=packages/wiki-check:packages/wiki-search"),
        ),
        InstallStep(phase="peer-wikis", argv=(*PREPARE, "peer-wikis")),
    )
    # Every prepared tree is installed, then built; a phase that only starts
    # a child beside the build changes neither.
    for steps in prepared.values():
        built = [step for step in steps if step["phase"] != "rehearsal-images"]
        assert [step["phase"] for step in built[:2]] == ["install", "workspace-build"]
        assert [step["argv"][4] for step in built[:2]] == ["install", "workspace-build"]


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
    # usage-limit case (MCPs board task ea46a7a4); 33 since
    # MCPs/instagram-mcp, for its scheduled publications (MCPs board task 4fd6d24a).
    assert len(databases) == 33
    assert {step["phase"] for step in databases} == {"test-database"}
    assert {step["argv"][5:] for step in databases} == {("CONTAINER=corvis-fleet-testdb",)}


def test_a_peer_wikis_step_runs_last_beside_the_companions_it_places() -> None:
    """MCPs fleet-prepare's peer-wikis phase moves the civic wiki's companion
    from beside the export into the place docker-compose mounts it (MCPs board
    task b1089284). It must run after install and workspace-build, which store
    the tree's untracked files, and its row must declare that companion, or the
    phase refuses FLEET_COMPANION_MISSING on the node. The bookshelf and chau
    companions it also placed left with those wikis' deregistration (MCPs board
    task 6ba9c531): one still declared would land beside the export, where
    wiki-check's guard reads its wiki as unregistered (fleet job b8d91553)."""
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
        directories = {companion["directory"] for companion in source["companions"]}
        assert "irvine-civic-wiki" in directories
        assert not {"bookshelf", "chau"} & directories


#: The repositories of the wikis MCPs deregistered on 2026-11-06 (MCPs board
#: task d549ae86), named without their ``.git`` suffix.
DEREGISTERED_WIKI_REPOSITORIES = ("wagner-austin/bookshelf", "wagner-austin/chau")


def _deregistered_wiki_companions(
    companions: Mapping[str, tuple[ProjectCompanion, ...]],
) -> list[str]:
    """Each MCPs project companion whose remote is a deregistered wiki's repository.

    Args:
        companions: Each project's companions, by project name.

    Returns:
        ``<project>: <remote>`` for every such companion, sorted.
    """
    return sorted(
        f"{name}: {companion['remote']}"
        for name, declared in companions.items()
        if name.startswith("MCPs/")
        for companion in declared
        if companion["remote"].removesuffix(".git").endswith(DEREGISTERED_WIKI_REPOSITORIES)
    )


def test_no_mcps_project_brings_a_deregistered_wikis_companion() -> None:
    """No MCPs row, placing or not, exports bookshelf or chau beside itself
    (MCPs board task a422ac68). A companion lands at ``../<directory>`` beside
    the export whatever its row's install steps are, and wiki-check's guard
    reads a wiki there that the registry no longer holds as unregistered."""
    workspace = decode_fleet_workspace(shipped_document())
    companions = {
        name: project["source"]["companions"]
        for name, project in workspace["projects"].items()
        if project["source"] is not None
    }

    assert _deregistered_wiki_companions(companions) == []
    # The negative: a row with no peer-wikis step, given a chau companion.
    assert companions["MCPs/mcp-shared"] == ()
    planted = ProjectCompanion(
        remote="https://github.com/wagner-austin/chau.git", ref="main", directory="chau"
    )
    assert _deregistered_wiki_companions({**companions, "MCPs/mcp-shared": (planted,)}) == [
        "MCPs/mcp-shared: https://github.com/wagner-austin/chau.git"
    ]


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
        "MCPs/execution-served-lacrymal",
    ]
    # The gateway host's lane migrates corvis_test first, for its usage-limit
    # case (MCPs board task ea46a7a4); the marker still comes last.
    assert {name: [step["phase"] for step in steps] for name, steps in marking.items()} == {
        "MCPs/execution-served-colossus": ["install", "workspace-build", "served-lane"],
        "MCPs/execution-served-demure": ["install", "workspace-build", "served-lane"],
        "MCPs/execution-served-lacrymal": ["install", "workspace-build", "served-lane"],
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
    files for the node's next row.

    Its workspace build names mcp-proxy as a root beside execution, as the
    deploy and linux lanes' do: the twin's check page names the relay bundle
    by the hash of mcp-proxy/assets/relay.js, which only mcp-proxy's build
    writes, and execution's own plan never reaches mcp-proxy. Without the
    root, fleet job b5ef14b9 at MCPs 266eece7e failed reading that file
    (MCPs board task f9074c8f)."""
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
        install=(
            InstallStep(phase="install", argv=(*PREPARE, "install")),
            InstallStep(
                phase="workspace-build",
                argv=(*PREPARE, "workspace-build", "ROOTS=execution:mcp-proxy"),
            ),
            InstallStep(phase="twin-lane", argv=(*PREPARE, "twin-lane")),
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
