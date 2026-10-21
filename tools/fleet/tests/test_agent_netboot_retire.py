"""The netboot lane's retirement at the tick level (MCPs board task 7b7570c5):
a claimed ``netboot-retire`` job verifies the installed netboot release, runs
its ``fleet-netboot retire`` for the machine the job names with the reason
the job carries, and closes with what the retirement said; the adopt pass
follows as on every netboot tick.

Split by role like ``test_agent_netboot.py``: the queue speaks the real wire
shape, and the verifier and the retirement go through the same command hook
every dispatch test uses. The release and the env file are real files under
the test's directory, read through the real hooks.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, narrow_json_to_str

from fleet.cli import agent
from fleet.core import _test_hooks, netboot_adopt, netboot_retire, sealed_release
from tests._agent_cli import HELD_THEN_CLAIMED, HUB_HOST, NOTHING_HELD, bind_hub_tick, netboot_argv
from tests._netboot_tick import (
    RELEASE_ID,
    REVISION,
    WRITER_ENV,
    adopt_calls,
    adopt_replies,
    netboot_place,
    place,
    sealed,
    verify_reply,
)
from tests._queue_fakes import FakeQueue, queue_job
from tests.conftest import FakeRun, failed, ok

#: Why pendragon's arm is retired, as the session gave it.
REASON: Final = "its ISO download was cut by a hub runner restart at 07:05:12Z"

#: What the retirement prints: the key's id, never a key.
RETIRED: Final = (
    "NETBOOT RETIRED host=pendragon mac=d8:5e:d3:84:53:d4 arm=8e6726b1 from=installing "
    f'key=kTEST5678CNTRL revoked=2026-10-21T07:26:30Z reason="{REASON}"'
)


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """The queue's variables, and the machine and unit the tick runs in."""
    bind_hub_tick()


def retire_row(**overrides: JSONValue) -> JSONObject:
    """A claimed netboot-retire row in the real wire shape.

    Args:
        **overrides: Fields to vary.

    Returns:
        The wire object: project MCPs, the machine as the requested node,
        the reason, and no commit, target or tags (MCPs mig 833).
    """
    fields: dict[str, JSONValue] = {
        "status": "claimed",
        "command": "netboot-retire",
        "project": "MCPs",
        "requestedNode": "pendragon",
        "sessionReason": REASON,
        "submittedBy": "opus-handoff-7b7570c5-0720-1021",
        "sha": None,
    }
    fields.update(overrides)
    return queue_job(**fields)


def ran_queue(closed: str) -> FakeQueue:
    """The queue's answers for a retirement that ran.

    Args:
        closed: The status it closes with.

    Returns:
        The held listing, the claim, the start report and the close report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": retire_row()}),
            dump_json_str({"job": retire_row(status="running", node=HUB_HOST)}),
            dump_json_str({"job": retire_row(status=closed, node=HUB_HOST)}),
        ]
    )


class TestARetirement:
    def test_a_claimed_retirement_runs_the_releases_retire_with_the_jobs_reason_and_closes_passed(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        release = paths[agent.NETBOOT_RELEASE_FLAG]
        runner = FakeRun([verify_reply(release), ok(f"{RETIRED}\n"), *adopt_replies(paths)])
        _test_hooks.run = runner
        endpoint = ran_queue("passed")
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report", "dispatch_report"]
        assert runner.calls == [
            sealed_release.verify_argv(release, "netboot"),
            netboot_retire.retire_argv(
                netboot_place(paths, config_path), sealed(paths), "pendragon", REASON
            ),
            *adopt_calls(paths, config_path),
        ]
        assert runner.timeouts == [
            300,
            netboot_retire.RETIRE_TIMEOUT_SECONDS,
            300,
            netboot_adopt.ADOPT_TIMEOUT_SECONDS,
        ]
        # The writer client reaches the retirement's environment and adopt's alone.
        assert runner.set_env == [(), WRITER_ENV, (), WRITER_ENV]
        closed = endpoint.arguments[3]
        assert (closed["status"], closed["exitCode"]) == ("passed", 0)
        assert narrow_json_to_str(closed["detail"]) == (
            f"fleet-netboot retire of pendragon's arm from sealed release {RELEASE_ID} "
            f"(revision {REVISION}): {RETIRED}"
        )
        assert "tskey-" not in dump_json_str(endpoint.arguments)

    def test_a_machine_with_no_live_arm_closes_refused_with_the_commands_text(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        refusal = (
            "NETBOOT_RETIRE_NO_LIVE_ARM: pendragon (d8:5e:d3:84:53:d4) has no waiting or "
            "installing arm to retire; its arms: 8e6726b1 expired"
        )
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

    def test_a_retirement_carrying_no_reason_is_refused_and_nothing_runs(
        self, config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        paths = place(tmp_path)
        runner = FakeRun(adopt_replies(paths))
        _test_hooks.run = runner
        endpoint = FakeQueue(
            [
                NOTHING_HELD,
                dump_json_str({"claimed": retire_row(sessionReason=None)}),
                dump_json_str({"job": retire_row(status="refused")}),
            ]
        )
        _test_hooks.http_post = endpoint

        assert agent.main(netboot_argv(config_path, repo, paths)) == 0

        # Only the adopt pass ran.
        assert runner.calls == adopt_calls(paths, config_path)
        assert endpoint.tools == [*HELD_THEN_CLAIMED, "dispatch_report"]
        assert narrow_json_to_str(endpoint.arguments[2]["detail"]) == (
            f"{netboot_retire.NO_REASON_CODE}: the retirement of pendragon's arm carries no "
            "reason, which dispatch_netboot_retire requires; nothing ran"
        )
