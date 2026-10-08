"""The retires a settle owes, and the record they are kept in (MCPs board task 8776b828).

:func:`fleet.core.retire_owed.retire_or_owe` and
:func:`fleet.core.retire_owed.retire_owed` run against the real transport
(:mod:`fleet.core.remote`) with the node's ssh answered by
:class:`tests.conftest.FakeRun`, and the record written to a real file
under ``tmp_path``. A node that did not answer owes the retire; a later
pass retires it once; a node that answered and failed raises by name and
is never sent the retire again.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError

from fleet.contracts.retire_record import (
    RetireRecord,
    RetireState,
    decode_retire_record,
    encode_retire_record,
)
from fleet.core import _test_hooks, records, remote, retire_owed
from tests._toolchain_fixtures import node
from tests.conftest import DEMO_RUN_ID, FakeRun, failed, ok, pin_clock, retire_replies, timed_out

#: A second run on the same node.
OTHER_RUN_ID = "libs-demo-lavender-1757000100"

#: What ssh said when lavender-wsl missed its reads on 2026-10-07.
BANNER_TIMEOUT = "Connection timed out during banner exchange"

#: What a Windows retire says when a file it moves is held open.
HELD_OPEN = "Move-Item: The process cannot access the file because it is being used"


def _path(tmp_path: pathlib.Path) -> pathlib.Path:
    """The retire record file, beside a ledger under ``runs``.

    Args:
        tmp_path: The test's directory.

    Returns:
        The path; the file does not exist until something is recorded.
    """
    return tmp_path / "runs" / retire_owed.RETIRES_FILE


def _states(path: pathlib.Path) -> list[tuple[str, str, RetireState]]:
    """Every line of the record, as (run, node, state).

    Args:
        path: The record file.

    Returns:
        One tuple per line, in order.
    """
    return [
        (record["run_id"], record["node"], record["state"]) for record in records.read_retires(path)
    ]


def _owe(path: pathlib.Path, run_id: str, *, alias: str = "lavender") -> None:
    """Record a run as owed, as a settle whose node missed the retire does.

    Args:
        path: The record file.
        run_id: The run.
        alias: Its node's workspace name.
    """
    records.append_retire(
        path,
        RetireRecord(
            run_id=run_id, node=alias, state=RetireState.OWED, at_unix=1, detail=BANNER_TIMEOUT
        ),
    )


class TestASettleRetire:
    def test_one_the_node_answered_is_done_and_recorded_nowhere(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        runner = FakeRun(retire_replies()[:2])
        _test_hooks.run = runner

        assert retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert len(runner.calls) == 2
        assert not path.exists()

    def test_one_the_node_did_not_answer_is_owed(self, tmp_path: pathlib.Path) -> None:
        path = _path(tmp_path)
        pin_clock(1_757_000_200)
        _test_hooks.run = FakeRun([ok(""), timed_out(remote.SSH_TIMEOUT_SECONDS)])

        assert not retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        (owed,) = records.read_retires(path)
        assert owed["run_id"] == DEMO_RUN_ID
        assert owed["state"] is RetireState.OWED
        assert owed["at_unix"] == 1_757_000_200
        assert owed["detail"].startswith("ssh to lavender timed out while running")
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)

    def test_one_the_node_answered_and_failed_raises_by_name_and_is_never_sent_again(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        _test_hooks.run = FakeRun([ok(""), failed(1, HELD_OPEN)])

        with pytest.raises(AppError) as raised:
            retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert raised.value.code is FleetErrorCode.DISPATCH_FAILED
        assert HELD_OPEN in raised.value.message
        assert _states(path) == [(DEMO_RUN_ID, "lavender", RetireState.FAILED)]
        nothing = FakeRun([])
        _test_hooks.run = nothing
        retire_owed.retire_owed(path, node(), alias="lavender")
        assert nothing.calls == []


class TestTheOwedRetires:
    def test_a_node_with_none_owed_is_sent_nothing(self, tmp_path: pathlib.Path) -> None:
        path = _path(tmp_path)
        _owe(path, DEMO_RUN_ID, alias="sedona")
        nothing = FakeRun([])
        _test_hooks.run = nothing

        retire_owed.retire_owed(path, node(), alias="lavender")

        assert nothing.calls == []
        assert retire_owed.owed_on(path, alias="sedona") == (DEMO_RUN_ID,)

    def test_a_node_still_not_answering_keeps_every_run_owed_after_one_attempt(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        _owe(path, DEMO_RUN_ID)
        _owe(path, OTHER_RUN_ID)
        missed = FakeRun([failed(remote.SSH_FAILURE, BANNER_TIMEOUT)])
        _test_hooks.run = missed

        retire_owed.retire_owed(path, node(), alias="lavender")

        assert len(missed.calls) == 1
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID, OTHER_RUN_ID)
        assert len(records.read_retires(path)) == 2

    def test_an_answering_node_retires_each_once_and_sweeps_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        _owe(path, DEMO_RUN_ID)
        _owe(path, OTHER_RUN_ID)
        replies = retire_replies()
        answered = FakeRun([*replies[:2], *replies])
        _test_hooks.run = answered

        retire_owed.retire_owed(path, node(), alias="lavender")

        assert [
            any(f"retire-{run_id}.ps1" in " ".join(call) for call in answered.calls)
            for run_id in (DEMO_RUN_ID, OTHER_RUN_ID)
        ] == [True, True]
        assert sum("fleet-venv-sweep" in " ".join(call) for call in answered.calls) == 2
        assert _states(path)[2:] == [
            (DEMO_RUN_ID, "lavender", RetireState.RETIRED),
            (OTHER_RUN_ID, "lavender", RetireState.RETIRED),
        ]
        (retired, _) = records.read_retires(path)[2:]
        assert retired["detail"] == (
            f"its transcript is kept at C:/fleet/stage/logs/{DEMO_RUN_ID}.log"
        )
        assert retire_owed.owed_on(path, alias="lavender") == ()
        nothing = FakeRun([])
        _test_hooks.run = nothing
        retire_owed.retire_owed(path, node(), alias="lavender")
        assert nothing.calls == []

    def test_one_the_node_answered_and_failed_raises_by_name_and_is_never_sent_again(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        _owe(path, DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), failed(1, HELD_OPEN)])

        with pytest.raises(AppError) as raised:
            retire_owed.retire_owed(path, node(), alias="lavender")

        assert raised.value.code is FleetErrorCode.DISPATCH_FAILED
        assert HELD_OPEN in raised.value.message
        assert _states(path) == [
            (DEMO_RUN_ID, "lavender", RetireState.OWED),
            (DEMO_RUN_ID, "lavender", RetireState.FAILED),
        ]
        nothing = FakeRun([])
        _test_hooks.run = nothing
        retire_owed.retire_owed(path, node(), alias="lavender")
        assert nothing.calls == []


class TestTheRecord:
    def test_a_line_round_trips(self) -> None:
        record = RetireRecord(
            run_id=DEMO_RUN_ID,
            node="lavender",
            state=RetireState.RETIRED,
            at_unix=1_757_000_300,
            detail="kept",
        )

        assert decode_retire_record(encode_retire_record(record)) == record

    def test_a_line_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="retire record must be a JSON object, got list"):
            decode_retire_record([])

    def test_an_unknown_state_is_refused_rather_than_read_as_done(self) -> None:
        with pytest.raises(JSONTypeError, match="'swept'"):
            decode_retire_record(
                {
                    "run_id": DEMO_RUN_ID,
                    "node": "lavender",
                    "state": "swept",
                    "at_unix": 1,
                    "detail": "",
                }
            )
