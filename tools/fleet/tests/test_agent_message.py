"""The message-session verb at the tick level (MCPs board tasks d3d514aa and c088658f).

A claimed ``message-session`` job rides the one session job path every verb
does: the sealed session-verbs release is verified, ``session-audit message``
runs with the row's target and message id and nothing the sender wrote, and
the job closes with the mode's own line, passed only on DELIVERED (MCPs board
task 394b7148). Split from ``test_agent_restart.py`` by role, with the same
fakes (``tests._session_jobs``).
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONValue, dump_json_str, narrow_json_to_str

from fleet.cli import agent
from fleet.core import _test_hooks, session_argv
from tests._agent_cli import HUB_HOST, NOTHING_HELD, agent_argv, bind_hub_tick
from tests._hub_tick import UNROLLED_TICK, unrolled_calls
from tests._queue_fakes import FakeQueue
from tests._session_jobs import TARGET, opening, refused_queue, restart_row, verb_reply
from tests._session_release_fixtures import plant_release, verify_call, verify_reply
from tests.conftest import FakeRun

#: The agent_session_messages row the claimed job delivers.
MESSAGE_ID = 5185


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the variables, machine and unit a hub lane job runs with."""
    bind_hub_tick()


def message_queue(closed: str) -> FakeQueue:
    """The queue's answers for a claimed message-session job that ran.

    Args:
        closed: The status it closes with.

    Returns:
        The claim, the start report and the close report.
    """
    row: dict[str, JSONValue] = {"command": "message-session", "messageId": MESSAGE_ID}
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": restart_row(**row)}),
            dump_json_str({"job": restart_row(**row, status="running", node=HUB_HOST)}),
            dump_json_str({"job": restart_row(**row, status=closed, node=HUB_HOST)}),
        ]
    )


def test_a_claimed_message_runs_session_audits_message_mode_with_its_row(
    config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    release = plant_release(tmp_path)
    line = f"MESSAGE DELIVERED: {TARGET} message {MESSAGE_ID}: {TARGET}.jsonl enqueued it\n"
    runner = FakeRun([verify_reply(tmp_path), verb_reply(0, line), *UNROLLED_TICK])
    _test_hooks.run = runner
    endpoint = message_queue("passed")
    _test_hooks.http_post = endpoint

    assert agent.main(agent_argv(config_path, repo)) == 0

    assert runner.calls == [
        verify_call(tmp_path),
        session_argv.message_argv(release["root"], HUB_HOST, TARGET, MESSAGE_ID),
        *unrolled_calls(repo),
    ]
    closed = endpoint.arguments[3]
    assert closed["status"] == "passed"
    assert narrow_json_to_str(closed["detail"]) == f"{opening('message', 0)} {line.strip()}"


def test_a_message_session_audit_did_not_deliver_closes_failed_with_its_outcome(
    config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    plant_release(tmp_path)
    held = f"MESSAGE HELD: {TARGET} message {MESSAGE_ID}: held (no-mode-asserted)\n"
    _test_hooks.run = FakeRun([verify_reply(tmp_path), verb_reply(1, held), *UNROLLED_TICK])
    endpoint = message_queue("failed")
    _test_hooks.http_post = endpoint

    assert agent.main(agent_argv(config_path, repo)) == 0

    closed = endpoint.arguments[3]
    assert closed["status"] == "failed"
    assert closed["exitCode"] == 1
    assert narrow_json_to_str(closed["detail"]) == f"{opening('message', 1)} {held.strip()}"


def test_a_message_session_naming_no_message_is_refused_before_the_release_is_read(
    config_path: pathlib.Path, repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    plant_release(tmp_path)
    runner = FakeRun([*UNROLLED_TICK])
    _test_hooks.run = runner
    endpoint = refused_queue(command="message-session", messageId=None)
    _test_hooks.http_post = endpoint

    assert agent.main(agent_argv(config_path, repo)) == 0

    assert runner.calls == [*unrolled_calls(repo)]
    assert "MESSAGE_ID_MISSING" in narrow_json_to_str(endpoint.arguments[2]["detail"])
