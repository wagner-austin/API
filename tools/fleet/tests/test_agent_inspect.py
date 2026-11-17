"""The inspect lane at the tick level (MCPs board task 3e9c99af): a claimed
``inspect`` job verifies the installed fleet-audit release, runs its
``fleet-inspect`` with only the job's machine and probe, and closes with the
inspection it printed through ``dispatch_report``'s ``inspected``.

Split by role like ``test_agent_netboot.py``: the queue speaks the real wire
shape, and the verifier and the CLI go through the same command hook every
dispatch test uses, faked at that seam; ``test_fleet_inspect.py`` runs the
command against a real process.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, load_json_str

from fleet.cli import agent, agent_lanes
from fleet.contracts.dispatch import DispatchLane
from fleet.core import _test_hooks, fleet_inspect, sealed_release
from tests._agent_cli import (
    HELD_THEN_CLAIMED,
    HUB_HOST,
    NOTHING_HELD,
    agent_argv,
    bind_hub_tick,
    placed_argv,
    tick_run_id,
)
from tests._hub_tick import UNROLLED_TICK, unrolled_calls
from tests._queue_fakes import FakeQueue, queue_job
from tests.conftest import FakeRun, failed, ok

RELEASE_ID: Final = "5d0e6b1c2a3f4e5d6c7b8a9f0e1d2c3b"
REVISION: Final = "c3f1e2d4b5a6978869a7b6c5d4e3f2a1b0c9d8e7"

#: The line the CLI prints for pendragon's services, as fleet-mcp encodes it.
ANSWER: Final = (
    '{"node":"pendragon","hostname":"pendragon","probe":"services",'
    '"startedAt":"2026-11-17T06:00:00.000Z","finishedAt":"2026-11-17T06:00:01.200Z",'
    '"exitStatus":0,"data":{"systemState":"running","failedUnits":{"lines":[],"total":0}},'
    '"error":null}'
)

SUBMITTER: Final = "opus-mcps-1117-eeab"


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """The queue's variables, and the machine and unit the tick runs in."""
    bind_hub_tick()


def place(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Lay out the installed release and the workspace, keyed by the flag naming each.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The release checkout and the workspace.
    """
    release = tmp_path / "releases" / "fleet-audit" / RELEASE_ID / "MCPs"
    release.mkdir(parents=True)
    workspace = tmp_path / "MCPs"
    workspace.mkdir()
    return {agent_lanes.INSPECT_RELEASE_FLAG: release, agent_lanes.WORKSPACE_FLAG: workspace}


def verify_reply(release: pathlib.Path) -> _test_hooks.CommandResult:
    """The release's verifier, answering that its seal holds.

    Args:
        release: The release's checkout.

    Returns:
        Exit 0 with the line MCPs maketools prints.
    """
    return ok(f"{sealed_release.verified_line('fleet-audit', RELEASE_ID, REVISION, release)}\n")


def expected_inspection(paths: dict[str, pathlib.Path]) -> tuple[str, ...]:
    """The command the agent runs for pendragon's services from the planted release.

    Args:
        paths: What :func:`place` laid out.

    Returns:
        The argv :func:`fleet.core.fleet_inspect.inspect_argv` composes.
    """
    release = paths[agent_lanes.INSPECT_RELEASE_FLAG]
    return fleet_inspect.inspect_argv(
        fleet_inspect.InspectPlace(
            release=release, nodejs="node", workspace=paths[agent_lanes.WORKSPACE_FLAG]
        ),
        sealed_release.SealedRelease(release=RELEASE_ID, revision=REVISION, root=release),
        fleet_inspect.InspectTarget(node="pendragon", probe="services"),
    )


def inspect_row(**overrides: JSONValue) -> JSONObject:
    """A claimed inspect row in the real wire shape.

    Args:
        **overrides: Fields to vary.

    Returns:
        The wire object: the probe as the project, the machine to read as
        the requested node, and no commit (MCPs mig 987).
    """
    fields: dict[str, JSONValue] = {
        "status": "claimed",
        "command": "inspect",
        "project": "probes/services",
        "requestedNode": "pendragon",
        "submittedBy": SUBMITTER,
        "sha": None,
    }
    fields.update(overrides)
    return queue_job(**fields)


def ran_queue(closed: str) -> FakeQueue:
    """The queue's three answers for an inspection that ran.

    Args:
        closed: The status it closes with.

    Returns:
        The claim, the start report and the closing report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": inspect_row()}),
            dump_json_str({"job": inspect_row(status="running", node=HUB_HOST)}),
            dump_json_str({"job": inspect_row(status=closed, node=HUB_HOST)}),
        ]
    )


def refused_queue(**claimed: JSONValue) -> FakeQueue:
    """The queue's two answers for an inspection refused before it ran.

    Args:
        **claimed: Fields to vary on the claimed row.

    Returns:
        The claim and the refusal report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": inspect_row(**claimed)}),
            dump_json_str({"job": inspect_row(status="refused")}),
        ]
    )


def inspect_argv(
    config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
) -> list[str]:
    """One inspect tick's command line with the planted place.

    Args:
        config_path: The workspace document.
        repo: The monorepo root.
        tmp_path: Where :func:`place` lays the release and workspace out.

    Returns:
        The argument list.
    """
    return placed_argv(DispatchLane.INSPECT, config_path, repo, place(tmp_path))


class TestTheInspectLane:
    def test_an_answered_probe_closes_the_job_with_its_inspection(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        release = paths[agent_lanes.INSPECT_RELEASE_FLAG]
        runner = FakeRun([verify_reply(release), ok(f"{ANSWER}\n")])
        _test_hooks.run = runner
        endpoint = ran_queue("passed")
        _test_hooks.http_post = endpoint

        argv = placed_argv(DispatchLane.INSPECT, config_path, repo, paths)
        assert agent.main(argv) == 0

        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report", "dispatch_report"]
        assert endpoint.arguments[0]["claimedBy"] == f"fleet-inspect-{HUB_HOST}"
        # Every inspection, whatever machine it names: the machine is the target.
        assert endpoint.arguments[1]["lane"] == "inspect"
        assert "node" not in endpoint.arguments[1]
        # The release's own verifier, then its own CLI, and no hub pass after.
        assert runner.calls == [
            sealed_release.verify_argv(release, "fleet-audit"),
            expected_inspection(paths),
        ]
        assert runner.timeouts == [300, fleet_inspect.INSPECT_TIMEOUT_SECONDS]
        assert runner.set_env == [(), ()]
        # Nothing from the job but its machine and its probe reaches the CLI.
        assert runner.calls[1][4:] == (
            "--workspace",
            str(paths[agent_lanes.WORKSPACE_FLAG]),
            "--node",
            "pendragon",
            "--probe",
            "services",
        )
        started = endpoint.arguments[2]
        assert (started["action"], started["node"], started["runId"]) == (
            "start",
            HUB_HOST,
            tick_run_id(),
        )
        inspected = endpoint.arguments[3]
        assert inspected["action"] == "inspected"
        assert inspected["jobId"] == started["jobId"]
        assert inspected["inspection"] == load_json_str(ANSWER)
        assert "status" not in inspected

    def test_the_cli_s_refusal_closes_the_job_refused_in_its_own_words(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        refusal = "INSPECT_NODE_DISABLED: pendragon is disabled in the fleet registry"
        _test_hooks.run = FakeRun(
            [verify_reply(paths[agent_lanes.INSPECT_RELEASE_FLAG]), failed(2, f"{refusal}\n")]
        )
        endpoint = ran_queue("refused")
        _test_hooks.http_post = endpoint

        assert agent.main(placed_argv(DispatchLane.INSPECT, config_path, repo, paths)) == 0

        closed = endpoint.arguments[3]
        assert (closed["action"], closed["status"]) == ("close", "refused")
        assert "exitCode" not in closed
        assert closed["detail"] == (
            f"fleet-inspect services of pendragon from sealed release {RELEASE_ID} "
            f"(revision {REVISION}) refused: {refusal}"
        )

    def test_a_run_that_never_answered_closes_the_job_failed(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        _test_hooks.run = FakeRun(
            [
                verify_reply(paths[agent_lanes.INSPECT_RELEASE_FLAG]),
                failed(1, "INSPECT_RUN_UNANSWERED: pendragon: copy exited 255\n"),
            ]
        )
        endpoint = ran_queue("failed")
        _test_hooks.http_post = endpoint

        assert agent.main(placed_argv(DispatchLane.INSPECT, config_path, repo, paths)) == 0

        closed = endpoint.arguments[3]
        assert (closed["action"], closed["status"], closed["exitCode"]) == ("close", "failed", 1)

    def test_a_job_naming_no_machine_is_refused_before_the_release_is_read(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner
        endpoint = refused_queue(requestedNode=None)
        _test_hooks.http_post = endpoint

        assert agent.main(inspect_argv(config_path, repo, tmp_path)) == 0

        assert runner.calls == []
        refused = endpoint.arguments[2]
        assert refused["status"] == "refused"
        assert str(refused["detail"]).startswith(f"{fleet_inspect.JOB_MALFORMED_CODE}: ")

    def test_a_release_failing_its_seal_refuses_the_job_with_nothing_run(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([failed(1, "SEAL BROKEN\n")])
        _test_hooks.run = runner
        endpoint = refused_queue()
        _test_hooks.http_post = endpoint

        assert agent.main(inspect_argv(config_path, repo, tmp_path)) == 0

        assert len(runner.calls) == 1
        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report"]
        assert endpoint.arguments[2]["status"] == "refused"

    def test_a_hub_runner_that_claims_an_inspection_refuses_it_naming_the_flags(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """A job is run by its command, whichever lane gave it, so a runner
        started without a fleet-audit release refuses an inspection rather
        than guess one."""
        runner = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue()
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [*unrolled_calls(repo)]
        assert endpoint.arguments[2]["detail"] == (
            f"{fleet_inspect.NO_RELEASE_CODE}: this runner was started without "
            "--inspect-release, --nodejs, --workspace, so it has no fleet-audit release to "
            "inspect from; the control duty's fleet-inspect runner claims the inspect lane"
        )


class TestTheInspectCommandLine:
    @pytest.mark.parametrize(
        ("extra", "named", "why"),
        [
            (
                ["--node", "pendragon"],
                "--node",
                "it claims every inspection, and the machine an inspection names is its target",
            ),
            (["--served-root", "/srv/corvis-netboot"], "--served-root", "only a netboot tick arms"),
            (
                ["--mcps-root", "/home/corvis/PROJECTS/MCPs"],
                "--mcps-root",
                "only a hub tick rebuilds the bases",
            ),
        ],
    )
    def test_an_inspect_tick_refuses_another_lanes_flags(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        extra: list[str],
        named: str,
        why: str,
    ) -> None:
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint
        argv = inspect_argv(config_path, repo, tmp_path)

        with pytest.raises(ValueError) as refused:
            agent.main([*argv, *extra])

        assert str(refused.value) == f"FLEET_AGENT_USAGE: the inspect lane takes no {named}; {why}"
        assert endpoint.tools == []

    def test_a_netboot_tick_refuses_the_inspect_flags_it_does_not_read(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        argv = placed_argv(DispatchLane.NETBOOT, config_path, repo, {})

        with pytest.raises(ValueError) as refused:
            agent.main([*argv, "--workspace", str(tmp_path)])

        assert str(refused.value) == (
            "FLEET_AGENT_USAGE: the netboot lane takes no --workspace; "
            "only an inspect tick inspects"
        )

    def test_an_inspect_tick_without_its_release_is_refused_before_any_claim(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        del paths[agent_lanes.INSPECT_RELEASE_FLAG]
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(ValueError, match="--inspect-release"):
            agent.main(placed_argv(DispatchLane.INSPECT, config_path, repo, paths))

        assert endpoint.tools == []
