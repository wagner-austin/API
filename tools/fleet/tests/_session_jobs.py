"""A claimed session job on the wire and its queue answers, for the session-lane tests.

Lifted from ``test_agent_restart.py`` when the message-session cases (MCPs
board tasks d3d514aa and c088658f) moved to ``test_agent_message.py``: both
drive a session verb through the real runner path, so both build its row,
its queue answers, session-audit's reply and the closing detail's opening
here, once.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from fleet.core import _test_hooks
from tests._agent_cli import HUB_HOST, NOTHING_HELD
from tests._queue_fakes import FakeQueue, queue_job
from tests._session_release_fixtures import RELEASE_ID, REVISION

TARGET = "934d9975-0d65-4e68-83de-b74f8c4df0c4"
SUBMITTER = "fable-dm-versionsplit-0912"


def opening(mode: str, returncode: int) -> str:
    """The start of a session job's closing detail.

    Args:
        mode: The session-audit mode that ran.
        returncode: Its exit code.

    Returns:
        The mode, the planted release and revision, and the exit code.
    """
    return (
        f"session-audit {mode} from sealed release {RELEASE_ID} "
        f"(revision {REVISION}) exited {returncode}:"
    )


def restart_row(**overrides: JSONValue) -> JSONObject:
    """A claimed restart-session row in the real wire shape.

    Args:
        **overrides: Fields to vary.

    Returns:
        The wire object.
    """
    fields: dict[str, JSONValue] = {
        "status": "claimed",
        "command": "restart-session",
        "project": "MCPs",
        "requestedNode": HUB_HOST,
        "sessionTarget": TARGET,
        "submittedBy": SUBMITTER,
        # A session verb acts on the hub, not on a commit (MCPs mig 532).
        "sha": None,
    }
    fields.update(overrides)
    return queue_job(**fields)


def ran_queue(command: str, closed: str) -> FakeQueue:
    """The queue's three answers for a session job that ran.

    Args:
        command: The row's verb.
        closed: The status it closes with.

    Returns:
        The claim, the start report and the close report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": restart_row(command=command)}),
            dump_json_str({"job": restart_row(command=command, status="running", node=HUB_HOST)}),
            dump_json_str({"job": restart_row(command=command, status=closed, node=HUB_HOST)}),
        ]
    )


def refused_queue(**claimed: JSONValue) -> FakeQueue:
    """The queue's two answers for a session job refused before it ran.

    Args:
        **claimed: Fields to vary on the claimed row.

    Returns:
        The claim and the refusal report.
    """
    return FakeQueue(
        [
            NOTHING_HELD,
            dump_json_str({"claimed": restart_row(**claimed)}),
            dump_json_str({"job": restart_row(status="refused")}),
        ]
    )


def verb_reply(returncode: int, stdout: str) -> _test_hooks.CommandResult:
    """session-audit's answer.

    Args:
        returncode: Its exit code.
        stdout: What it printed.

    Returns:
        The result.
    """
    return _test_hooks.CommandResult(
        returncode=returncode, stdout=stdout, stderr="", timed_out=False
    )
