"""The message-session job on the wire (MCPs board tasks d3d514aa and c088658f).

The queue admits ``message-session`` only after every runner can read it, and
from MCPs migration 995 every job carries ``messageId``: the
agent_session_messages row a message-session delivers, ``null`` on every
other command. Split from ``test_dispatch_contracts.py`` by role, with its
fakes.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONValue, dump_json_str

from fleet.contracts.dispatch import DispatchCommand, decode_claim
from tests._queue_fakes import claimed_job, queue_job

TARGET = "58a72f7f-27c1-47dc-91cc-181b56cbac5c"


def test_a_message_session_decodes_with_its_target_and_message() -> None:
    job = claimed_job(
        dump_json_str(
            {
                "claimed": queue_job(
                    command="message-session",
                    project="MCPs",
                    requestedNode="ziggurat",
                    sessionTarget=TARGET,
                    sha=None,
                    messageId=5185,
                )
            }
        )
    )

    assert job["command"] is DispatchCommand.MESSAGE_SESSION
    assert job["session_target"] == TARGET
    assert job["message_id"] == 5185
    assert claimed_job(dump_json_str({"claimed": queue_job()}))["message_id"] is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "field 'messageId' is 0, not a whole number of at least 1"),
        ("5185", "field 'messageId' is '5185', not a whole number of at least 1"),
        (True, "field 'messageId' is True, not a whole number of at least 1"),
    ],
)
def test_a_message_id_the_queue_cannot_hold_is_refused(value: JSONValue, message: str) -> None:
    with pytest.raises(AppError) as raised:
        decode_claim(dump_json_str({"claimed": queue_job(messageId=value)}))

    assert raised.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED
    assert message in raised.value.message


def test_a_missing_message_id_is_a_changed_contract_not_an_absent_message() -> None:
    """A queue rolled before MCPs mig 995 sends none; the roll waits for it."""
    row = queue_job()
    del row["messageId"]

    with pytest.raises(AppError) as raised:
        decode_claim(dump_json_str({"claimed": row}))

    assert "field 'messageId' is missing" in raised.value.message
