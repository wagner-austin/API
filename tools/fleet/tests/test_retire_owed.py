"""Every run's retire, owed from its lease, and the record it is kept in.

MCPs board tasks 8776b828 and 3d9eec0d. :func:`fleet.core.retire_owed.owe`,
:func:`fleet.core.retire_owed.retire_or_owe`,
:func:`fleet.core.retire_owed.retire_or_raise` and
:func:`fleet.core.retire_owed.retire_owed` run against the real transport
(:mod:`fleet.core.remote`) with the node's ssh answered by
:class:`tests.conftest.FakeRun`, and the record, the leases and the ledger
written to real files under ``tmp_path``. A run is owed from its lease; a
retire that is done records it retired once; a node that did not answer
leaves it owed and writes nothing; a node that answered and failed raises by
name and is never sent the retire again; and a pass leaves alone every owed
run a lease or a running ledger row still holds.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError

from fleet.contracts.lease import Lease
from fleet.contracts.ledger import LedgerOutcome
from fleet.contracts.retire_record import (
    RetireRecord,
    RetireState,
    decode_retire_record,
    encode_retire_record,
)
from fleet.core import _test_hooks, dispatch, leases, records, remote, retire_owed
from tests._toolchain_fixtures import node
from tests.conftest import (
    DEMO_NOW,
    DEMO_RUN_ID,
    FakeRun,
    failed,
    ok,
    pin_clock,
    retire_replies,
    timed_out,
)

#: A second run on the same node.
OTHER_RUN_ID = "libs-demo-lavender-1757000100"

#: What ssh said when lavender-wsl missed its reads on 2026-10-07.
BANNER_TIMEOUT = "Connection timed out during banner exchange"

#: What a Windows retire says when a file it moves is held open.
HELD_OPEN = "Move-Item: The process cannot access the file because it is being used"

#: Where the demo node keeps the demo run's transcript once it is retired.
KEPT = f"C:/fleet/stage/logs/{DEMO_RUN_ID}.log"


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


def _lease(run_id: str, *, expires: int) -> Lease:
    """A lease one run holds on the demo node.

    Args:
        run_id: The run.
        expires: When it lapses.

    Returns:
        The lease.
    """
    return Lease(
        node="lavender",
        project="libs/demo",
        run_id=run_id,
        agent="opus-fleet-0904",
        session_id="acc774c0-3bc3-4cce-9dda-c7a12fb99519",
        acquired_unix=DEMO_NOW,
        expires_unix=expires,
        resources=(),
    )


def _pass(path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Send the demo node's owed retires, with the leases and ledger under ``tmp_path``.

    Args:
        path: The record file.
        tmp_path: The test's directory.
    """
    retire_owed.retire_owed(
        path,
        node(),
        alias="lavender",
        leases_path=tmp_path / "runs" / "leases.json",
        ledger_path=tmp_path / "runs" / "ledger.jsonl",
    )


class TestTheLease:
    def test_owes_the_run_s_retire(self, tmp_path: pathlib.Path) -> None:
        path = _path(tmp_path)
        pin_clock(1_757_000_100)

        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)

        (owed,) = records.read_retires(path)
        assert owed == RetireRecord(
            run_id=DEMO_RUN_ID,
            node="lavender",
            state=RetireState.OWED,
            at_unix=1_757_000_100,
            detail="its lease is held, so its directory is staged on the node",
        )
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)
        assert retire_owed.owed_on(path, alias="sedona") == ()


class TestASettleRetire:
    def test_one_the_node_answered_is_recorded_retired(self, tmp_path: pathlib.Path) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        runner = FakeRun(retire_replies()[:2])
        _test_hooks.run = runner

        assert retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert len(runner.calls) == 2
        retired = records.read_retires(path)[1]
        assert retired["state"] is RetireState.RETIRED
        assert retired["detail"] == f"its transcript is kept at {KEPT}"
        assert retire_owed.owed_on(path, alias="lavender") == ()

    def test_one_the_node_did_not_answer_stays_owed_and_writes_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), timed_out(remote.SSH_TIMEOUT_SECONDS)])

        assert not retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert _states(path) == [(DEMO_RUN_ID, "lavender", RetireState.OWED)]
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)

    def test_one_the_node_answered_and_failed_raises_by_name_and_is_never_sent_again(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), failed(1, HELD_OPEN)])

        with pytest.raises(AppError) as raised:
            retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert raised.value.code is FleetErrorCode.DISPATCH_FAILED
        assert HELD_OPEN in raised.value.message
        assert _states(path) == [
            (DEMO_RUN_ID, "lavender", RetireState.OWED),
            (DEMO_RUN_ID, "lavender", RetireState.FAILED),
        ]
        nothing = FakeRun([])
        _test_hooks.run = nothing
        _pass(path, tmp_path)
        assert nothing.calls == []


class TestARetireBeforeTheRowCloses:
    def test_one_the_node_answered_is_recorded_retired_and_names_the_transcript(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun(retire_replies()[:2])

        kept = retire_owed.retire_or_raise(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert kept == KEPT
        assert _states(path)[1:] == [(DEMO_RUN_ID, "lavender", RetireState.RETIRED)]

    @pytest.mark.parametrize(
        ("result", "code"),
        [
            (failed(remote.SSH_FAILURE, BANNER_TIMEOUT), FleetErrorCode.NODE_UNREACHABLE),
            (failed(1, HELD_OPEN), FleetErrorCode.DISPATCH_FAILED),
        ],
    )
    def test_one_not_done_raises_its_code_and_leaves_the_run_owed(
        self, tmp_path: pathlib.Path, result: _test_hooks.CommandResult, code: FleetErrorCode
    ) -> None:
        """The row is still live, so the next tick settles the run again."""
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), result])

        with pytest.raises(AppError) as raised:
            retire_owed.retire_or_raise(path, node(), alias="lavender", run_id=DEMO_RUN_ID)

        assert raised.value.code is code
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)


class TestTheOwedRetires:
    def test_a_node_with_none_owed_is_sent_nothing(self, tmp_path: pathlib.Path) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="sedona", run_id=DEMO_RUN_ID)
        nothing = FakeRun([])
        _test_hooks.run = nothing

        _pass(path, tmp_path)

        assert nothing.calls == []
        assert retire_owed.owed_on(path, alias="sedona") == (DEMO_RUN_ID,)

    def test_a_node_still_not_answering_keeps_every_run_owed_after_one_attempt(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        retire_owed.owe(path, alias="lavender", run_id=OTHER_RUN_ID)
        missed = FakeRun([failed(remote.SSH_FAILURE, BANNER_TIMEOUT)])
        _test_hooks.run = missed

        _pass(path, tmp_path)

        assert len(missed.calls) == 1
        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID, OTHER_RUN_ID)
        assert len(records.read_retires(path)) == 2

    def test_an_answering_node_retires_each_once_and_removes_generations_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        retire_owed.owe(path, alias="lavender", run_id=OTHER_RUN_ID)
        replies = retire_replies()
        answered = FakeRun([*replies[:2], *replies])
        _test_hooks.run = answered

        _pass(path, tmp_path)

        assert [
            any(f"retire-{run_id}.ps1" in " ".join(call) for call in answered.calls)
            for run_id in (DEMO_RUN_ID, OTHER_RUN_ID)
        ] == [True, True]
        assert sum("fleet-cache-generations" in " ".join(call) for call in answered.calls) == 2
        assert _states(path)[2:] == [
            (DEMO_RUN_ID, "lavender", RetireState.RETIRED),
            (OTHER_RUN_ID, "lavender", RetireState.RETIRED),
        ]
        assert retire_owed.owed_on(path, alias="lavender") == ()
        nothing = FakeRun([])
        _test_hooks.run = nothing
        _pass(path, tmp_path)
        assert nothing.calls == []

    def test_one_the_node_answered_and_failed_raises_by_name_and_is_never_sent_again(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _path(tmp_path)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        _test_hooks.run = FakeRun([ok(""), failed(1, HELD_OPEN)])

        with pytest.raises(AppError) as raised:
            _pass(path, tmp_path)

        assert raised.value.code is FleetErrorCode.DISPATCH_FAILED
        assert HELD_OPEN in raised.value.message
        assert _states(path) == [
            (DEMO_RUN_ID, "lavender", RetireState.OWED),
            (DEMO_RUN_ID, "lavender", RetireState.FAILED),
        ]
        nothing = FakeRun([])
        _test_hooks.run = nothing
        _pass(path, tmp_path)
        assert nothing.calls == []


class TestARunStillLive:
    def test_a_held_lease_keeps_its_run_owed_until_it_lapses(self, tmp_path: pathlib.Path) -> None:
        """A launch staging under its lease, or one whose runner ended and
        whose lease has not yet lapsed: its directory is not touched."""
        path = _path(tmp_path)
        clock = pin_clock(DEMO_NOW)
        leases.acquire(
            tmp_path / "runs" / "leases.json",
            _lease(DEMO_RUN_ID, expires=DEMO_NOW + 600),
            now_unix=DEMO_NOW,
        )
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        nothing = FakeRun([])
        _test_hooks.run = nothing

        _pass(path, tmp_path)

        assert nothing.calls == []
        clock.seconds = DEMO_NOW + 601
        _test_hooks.run = FakeRun(retire_replies())

        _pass(path, tmp_path)

        assert retire_owed.owed_on(path, alias="lavender") == ()

    def test_a_running_ledger_row_keeps_its_run_owed_until_it_closes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A build whose lease lapsed while it ran is still the row's settle's or stop's."""
        path = _path(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        running = dispatch.started_row(
            lease=_lease(DEMO_RUN_ID, expires=DEMO_NOW), host="node-1", workers=1, detail="run"
        )
        records.append_ledger(ledger, running)
        retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
        retire_owed.owe(path, alias="lavender", run_id=OTHER_RUN_ID)
        answered = FakeRun(retire_replies())
        _test_hooks.run = answered

        _pass(path, tmp_path)

        assert retire_owed.owed_on(path, alias="lavender") == (DEMO_RUN_ID,)
        assert not any(f"retire-{DEMO_RUN_ID}.ps1" in " ".join(c) for c in answered.calls)
        records.append_ledger(
            ledger,
            dispatch.closed_row(
                running,
                outcome=LedgerOutcome.PASSED,
                exit_code=0,
                ended_unix=DEMO_NOW + 60,
                detail="passed",
            ),
        )
        _test_hooks.run = FakeRun(retire_replies())

        _pass(path, tmp_path)

        assert retire_owed.owed_on(path, alias="lavender") == ()


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
