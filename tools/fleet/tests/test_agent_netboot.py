"""The netboot lane at the tick level (MCPs board task 83102d9d): a claimed
``netboot-arm`` job verifies the installed netboot release, runs its
``fleet-netboot arm`` for the machine the job names, and closes with what
the arm said; every netboot tick then runs the adopt pass (MCPs board task
49d366ac, ``test_agent_netboot_adopt.py``).

Split by role like ``test_agent_restart.py``: the queue speaks the real
wire shape, and the verifier and the arm go through the same command hook
every dispatch test uses. The release and the env file are real files under
the test's directory, read through the real hooks.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Final

import pytest
from board_watch import _test_hooks as board_watch_hooks
from board_watch import config as board_config
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, narrow_json_to_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL

from fleet.cli import agent
from fleet.core import _test_hooks, netboot_adopt, netboot_arm, queue, sealed_release
from tests._agent_cli import (
    HELD_THEN_CLAIMED,
    HUB_HOST,
    NOTHING_HELD,
    agent_argv,
    bind_hub_tick,
    netboot_argv,
    sessions_argv,
    tick_run_id,
)
from tests._hub_tick import UNROLLED_TICK, unrolled_calls
from tests._netboot_tick import (
    API_KEY,
    RELEASE_ID,
    REVISION,
    adopt_calls,
    adopt_replies,
    netboot_place,
    place,
    sealed,
    verify_reply,
)
from tests._queue_fakes import FakeEnv, FakeQueue, queue_job
from tests.conftest import FakeRun, failed, ok

#: What the arm prints once lavender is armed: the key's id, never a key.
ARMED: Final = (
    "NETBOOT ARMED host=lavender mac=1c:69:7a:fe:78:9a arm=ded2eb05 build=4be1 "
    f"commit={REVISION} key=kTEST5678CNTRL ephemeral=false expires=2026-10-19T08:00:00.000Z"
)

SUBMITTER: Final = "opus-mcps-1015-9d59"


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """The queue's variables, and the machine and unit the tick runs in."""
    bind_hub_tick()


def arm_row(**overrides: JSONValue) -> JSONObject:
    """A claimed netboot-arm row in the real wire shape.

    Args:
        **overrides: Fields to vary.

    Returns:
        The wire object: project MCPs, the machine to arm as the requested
        node, and no commit, target or tags (MCPs mig 807).
    """
    fields: dict[str, JSONValue] = {
        "status": "claimed",
        "command": "netboot-arm",
        "project": "MCPs",
        "requestedNode": "lavender",
        "submittedBy": SUBMITTER,
        "sha": None,
    }
    fields.update(overrides)
    return queue_job(**fields)


def ran_queue(closed: str) -> FakeQueue:
    """The queue's three answers for an arm that ran.

    Args:
        closed: The status it closes with.

    Returns:
        The claim, the start report and the close report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": arm_row()}),
            dump_json_str({"job": arm_row(status="running", node=HUB_HOST)}),
            dump_json_str({"job": arm_row(status=closed, node=HUB_HOST)}),
        ]
    )


def refused_queue(**claimed: JSONValue) -> FakeQueue:
    """The queue's two answers for an arm refused before it ran.

    Args:
        **claimed: Fields to vary on the claimed row.

    Returns:
        The claim and the refusal report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": arm_row(**claimed)}),
            dump_json_str({"job": arm_row(status="refused")}),
        ]
    )


def expected_arm(paths: dict[str, pathlib.Path], registry: pathlib.Path) -> tuple[str, ...]:
    """The arm the agent runs for lavender from the planted release.

    Args:
        paths: What :func:`tests._netboot_tick.place` laid out.
        registry: The registry directory the tick loaded.

    Returns:
        The argv :func:`fleet.core.netboot_arm.arm_argv` composes.
    """
    return netboot_arm.arm_argv(netboot_place(paths, registry), sealed(paths), "lavender")


class TestTheNetbootLane:
    def test_a_claimed_arm_verifies_the_release_runs_its_arm_and_closes_with_its_line(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        release = paths[agent.NETBOOT_RELEASE_FLAG]
        runner = FakeRun(
            [
                verify_reply(release),
                ok(f"NETBOOT BUILT digest=4be1\n{ARMED}\n"),
                *adopt_replies(paths),
            ]
        )
        _test_hooks.run = runner
        endpoint = ran_queue("passed")
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report", "dispatch_report"]
        # Every arm, whatever machine it names: the machine is the target.
        assert endpoint.arguments[0]["claimedBy"] == f"fleet-netboot-{HUB_HOST}"
        assert endpoint.arguments[1]["lane"] == "netboot"
        assert "node" not in endpoint.arguments[1]
        # The release's own verifier, then the release's own arm, then the
        # adopt pass, and no hub pass after it.
        assert runner.calls == [
            sealed_release.verify_argv(release, "netboot"),
            expected_arm(paths, config_path),
            *adopt_calls(paths, config_path),
        ]
        assert runner.timeouts == [
            300,
            netboot_arm.ARM_TIMEOUT_SECONDS,
            300,
            netboot_adopt.ADOPT_TIMEOUT_SECONDS,
        ]
        # The key reaches the arm's and adopt's environments and nothing else.
        key = (("TAILSCALE_API_KEY", API_KEY),)
        assert runner.set_env == [(), key, (), key]
        assert not any(API_KEY in token for call in runner.calls for token in call)
        started = endpoint.arguments[2]
        assert started["node"] == HUB_HOST
        assert started["runId"] == tick_run_id()
        closed = endpoint.arguments[3]
        assert closed["status"] == "passed"
        assert closed["exitCode"] == 0
        assert narrow_json_to_str(closed["detail"]) == (
            f"fleet-netboot arm of lavender from sealed release {RELEASE_ID} "
            f"(revision {REVISION}): {ARMED}"
        )
        assert "tskey-" not in dump_json_str(endpoint.arguments)

    def test_an_arm_the_cli_refuses_closes_refused_with_its_own_text(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        refusal = "NETBOOT_ARM_MACHINE_NOT_ARMABLE: lavender declares no bootMac"
        _test_hooks.run = FakeRun(
            [
                verify_reply(paths[agent.NETBOOT_RELEASE_FLAG]),
                failed(1, f"fleet-netboot: {refusal}\n"),
                *adopt_replies(paths),
            ]
        )
        endpoint = ran_queue("refused")
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        closed = endpoint.arguments[3]
        assert closed["status"] == "refused"
        assert "exitCode" not in closed
        assert narrow_json_to_str(closed["detail"]) == refusal

    def test_an_arm_that_fails_closes_failed_with_its_exit_and_output(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        _test_hooks.run = FakeRun(
            [
                verify_reply(paths[agent.NETBOOT_RELEASE_FLAG]),
                failed(1, "fleet-netboot: TAILNET_KEY_MINT_FAILED: Tailscale answered 403\n"),
                *adopt_replies(paths),
            ]
        )
        endpoint = ran_queue("failed")
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        closed = endpoint.arguments[3]
        assert closed["status"] == "failed"
        assert closed["exitCode"] == 1
        assert narrow_json_to_str(closed["detail"]).endswith(
            "exited 1: fleet-netboot: TAILNET_KEY_MINT_FAILED: Tailscale answered 403"
        )


class TestNetbootRefusals:
    """Nothing is minted when there is nothing verified to arm from, no key
    to mint with, or no machine to arm; the adopt pass that follows refuses
    alike when the release or the key is what is missing."""

    def test_a_release_that_fails_its_seal_refuses_the_arm_and_nothing_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        unsealed = failed(1, "RELEASE_DIGEST_MISMATCH: the payload changed")
        runner = FakeRun([unsealed, unsealed])
        _test_hooks.run = runner
        endpoint = refused_queue()
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        verify = sealed_release.verify_argv(paths[agent.NETBOOT_RELEASE_FLAG], "netboot")
        assert runner.calls == [verify, verify]
        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report"]
        detail = narrow_json_to_str(endpoint.arguments[2]["detail"])
        assert detail.startswith(f"{sealed_release.UNSEALED_CODE}: ")
        assert "RELEASE_DIGEST_MISMATCH: the payload changed" in detail

    def test_an_env_file_with_no_key_refuses_the_arm_before_it_starts(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        paths[agent.ENV_FILE_FLAG].write_text("VM_HOST=lavender-wsl\n", encoding="utf-8")
        release = paths[agent.NETBOOT_RELEASE_FLAG]
        runner = FakeRun([verify_reply(release), verify_reply(release)])
        _test_hooks.run = runner
        endpoint = refused_queue()
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        # The arm's verify and the adopt pass's, and neither command ran.
        assert len(runner.calls) == 2
        assert narrow_json_to_str(endpoint.arguments[2]["detail"]) == (
            f"{netboot_arm.NO_KEY_CODE}: {paths[agent.ENV_FILE_FLAG]} sets no TAILSCALE_API_KEY, "
            "or sets it empty, and the netboot duty's commands run with it; nothing ran"
        )

    def test_an_arm_naming_no_machine_is_refused_and_nothing_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        runner = FakeRun(adopt_replies(paths))
        _test_hooks.run = runner
        endpoint = refused_queue(requestedNode=None)
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        # Only the adopt pass ran.
        assert runner.calls == adopt_calls(paths, config_path)
        assert narrow_json_to_str(endpoint.arguments[2]["detail"]).startswith(
            f"{netboot_arm.NO_MACHINE_CODE}: the job names no machine to arm"
        )

    def test_a_hub_runner_that_claims_an_arm_refuses_it_naming_the_flags(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """A job is run by its command, whichever lane gave it, so a runner
        started without a release refuses an arm rather than guess one, and
        runs no adopt pass, which is the netboot duty's."""
        runner = FakeRun([*UNROLLED_TICK])
        _test_hooks.run = runner
        endpoint = refused_queue()
        _test_hooks.http_post = endpoint

        assert agent.main(agent_argv(config_path, repo)) == 0

        assert runner.calls == [*unrolled_calls(repo)]
        assert narrow_json_to_str(endpoint.arguments[2]["detail"]) == (
            f"{netboot_arm.NO_RELEASE_CODE}: this runner was started without "
            "--netboot-release, --nodejs, --served-root, --env-file, so it has no netboot "
            "release to arm from; the netboot duty's fleet-netboot runner claims arms"
        )


class TestTheNetbootCommandLine:
    @pytest.mark.parametrize(
        ("extra", "named", "why"),
        [
            (
                ["--node", "lavender"],
                "--node",
                "it claims every arm, and the machine an arm names is its target",
            ),
            (
                ["--mcps-root", "/home/corvis/PROJECTS/MCPs"],
                "--mcps-root",
                "only a hub tick rebuilds the bases",
            ),
        ],
    )
    def test_a_netboot_tick_refuses_another_lanes_flags(
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

        with pytest.raises(ValueError) as refused:
            agent.main([*netboot_argv(config_path, repo, place(tmp_path)), *extra])

        assert str(refused.value) == f"FLEET_AGENT_USAGE: the netboot lane takes no {named}; {why}"
        assert endpoint.tools == []

    @pytest.mark.parametrize("lane", ["hub", "sessions"])
    def test_every_other_lane_refuses_the_netboot_flags(
        self, config_path: pathlib.Path, repo: pathlib.Path, lane: str
    ) -> None:
        argv = (
            sessions_argv(config_path, repo, "ziggurat")
            if lane == "sessions"
            else agent_argv(config_path, repo)
        )

        with pytest.raises(ValueError) as refused:
            agent.main([*argv, agent.NODEJS_FLAG, "node", agent.ENV_FILE_FLAG, ".env"])

        assert str(refused.value) == (
            f"FLEET_AGENT_USAGE: the {lane} lane takes no --nodejs or --env-file; "
            "only a netboot tick arms"
        )

    def test_a_netboot_tick_without_its_release_is_refused_before_any_claim(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        del paths[agent.NETBOOT_RELEASE_FLAG]
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(ValueError, match="--netboot-release"):
            agent.main(netboot_argv(config_path, repo, paths))

        assert endpoint.tools == []

    def test_announce_registers_the_netboot_runner_and_needs_no_release(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        board_watch_hooks.env = FakeEnv(
            {
                board_config.API_KEY_VARIABLE: "board-key",
                board_config.TENANT_ID_VARIABLE: "tenant",
                board_config.URL_VARIABLE: DECLARED_TASKBOARD_URL,
            }
        )
        endpoint = FakeQueue(["checked in"])
        _test_hooks.http_post = endpoint
        argv = [*netboot_argv(config_path, repo, {}), queue.ANNOUNCE_FLAG]

        assert agent.main(argv) == 0

        checkin = endpoint.arguments[0]
        assert endpoint.tools == ["task_post"]
        assert checkin["machine"] == f"{sys.platform}:{HUB_HOST}"
        assert checkin["agent"] == f"fleet-netboot-{HUB_HOST}"
        assert checkin["body"] == (
            f"fleet-netboot-{HUB_HOST}: claims the queue's netboot lane (netboot-arm) and closes "
            "each job it claims, which posts the outcome to the job's task thread under this label"
        )
