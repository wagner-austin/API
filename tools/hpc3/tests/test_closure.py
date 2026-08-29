"""Tests for the closure record and the false finding it prevents.

The bug: ``sacct`` retention is finite. A job that ran perfectly a month ago
eventually has no accounting row, which is character-for-character the same
observation as a job that never existed. Without a closure record the tool
reports it as ``unaccounted`` forever, the finding count climbs without bound,
and ``hpc3-triage`` exits non-zero permanently -- which is the same as having
no triage, because nobody reads a board that is always red.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.closure import Closure, decode_closure, encode_closure
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.status import JobStatus
from hpc3.core import ledger
from hpc3.core.triage import closures_for, open_entries, unaccounted_jobs
from tests.against_hpc3 import decode_job_status, decode_ledger_entry
from tests.conftest import ledger_row, write_file

_AT = "2026-08-22T16:00:00+00:00"


def _closed(job_id: str, state: str = "COMPLETED", closed_at: str = _AT) -> Closure:
    """Build a closure for a bookkeeping test.

    Args:
        job_id: Job id.
        state: Terminal state accounting reported.
        closed_at: When this tool noticed.

    Returns:
        A validated closure. ``elapsed_seconds`` is null throughout this file:
        these tests are about closure bookkeeping, not about runtimes, and a
        fabricated duration here would read as evidence somewhere it is not.
    """
    return decode_closure(
        {"job_id": job_id, "state": state, "closed_at": closed_at, "elapsed_seconds": None}
    )


def _entry(job_id: str) -> LedgerEntry:
    """Build a ledger entry.

    Args:
        job_id: Job id.

    Returns:
        A validated entry.
    """
    return decode_ledger_entry(
        ledger_row(job_id=job_id, name=f"abl.arm-{job_id}", submitted_at=_AT)
    )


def _status(job_id: str, state: str) -> JobStatus:
    """Build an accounting row in a given state.

    Args:
        job_id: Job id.
        state: State to report.

    Returns:
        A validated status.
    """
    base: dict[str, JSONValue] = {
        "job_id": job_id,
        "name": f"abl.arm-{job_id}",
        "partition": "free-gpu",
        "state": state,
        "elapsed_seconds": 60,
        "billing_tres": 8,
        "gpu_count": 1,
        "cpu_count": 8,
        "node_list": "n1",
    }
    return decode_job_status(base)


class TestTheRetentionFalsePositive:
    def test_a_forgotten_job_is_reported_when_it_has_no_closure(self) -> None:
        """The bug, reproduced: accounting forgot it, so it looks like it
        never existed."""
        findings = unaccounted_jobs([_entry("101")], [])
        assert [f.kind for f in findings] == ["unaccounted"]

    def test_a_closed_job_is_never_asked_about_again(self) -> None:
        """The fix: it is not in the set the question is asked of."""
        closed = {"101": _closed("101", "COMPLETED", _AT)}
        assert open_entries([_entry("101")], closed) == []

    def test_the_two_together_stop_the_permanent_finding(self) -> None:
        closed = {"101": _closed("101", "COMPLETED", _AT)}
        still_open = open_entries([_entry("101"), _entry("102")], closed)
        assert [f.job_id for f in unaccounted_jobs(still_open, [])] == ["102"]

    def test_a_job_that_never_finished_stays_reportable(self) -> None:
        """The finding must survive for the case it exists for: a job that
        vanished before this tool ever saw it end has no closure."""
        assert [e["job_id"] for e in open_entries([_entry("101")], {})] == ["101"]


class TestClosuresFor:
    def test_every_terminal_state_closes(self) -> None:
        terminal = ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL"]
        statuses = [_status(str(i), state) for i, state in enumerate(terminal)]
        assert len(closures_for(statuses, closed_at=_AT)) == len(terminal)

    def test_a_failure_closes_just_as_a_success_does(self) -> None:
        """Accounting forgets a failed job on the same schedule."""
        closures = closures_for([_status("101", "FAILED")], closed_at=_AT)
        assert closures[0]["state"] == "FAILED"

    def test_a_running_job_does_not_close(self) -> None:
        assert closures_for([_status("101", "RUNNING")], closed_at=_AT) == []

    def test_a_pending_job_does_not_close(self) -> None:
        assert closures_for([_status("101", "PENDING")], closed_at=_AT) == []

    def test_a_requeued_job_does_not_close(self) -> None:
        """Going back to the queue is protection working, not the run ending."""
        assert closures_for([_status("101", "REQUEUED")], closed_at=_AT) == []

    def test_the_timestamp_is_when_it_was_noticed(self) -> None:
        """Not when the job ended -- that is not something this can claim."""
        assert closures_for([_status("101", "COMPLETED")], closed_at=_AT)[0]["closed_at"] == _AT


class TestClosureStore:
    def test_closures_live_beside_their_ledger(self, tmp_path: pathlib.Path) -> None:
        """Derived, not configured: two files describing one set of jobs must
        not be separately addressable."""
        assert ledger.closure_path(tmp_path / "ledger.jsonl").name == "ledger.jsonl.closed"

    def test_a_written_closure_reads_back(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "l.jsonl.closed"
        written = Closure(job_id="101", state="COMPLETED", closed_at=_AT, elapsed_seconds=1618)
        ledger.append_closure(path, written)
        assert ledger.read_closures(path) == {"101": written}

    def test_an_absent_file_reads_as_nothing_closed(self, tmp_path: pathlib.Path) -> None:
        assert ledger.read_closures(tmp_path / "never.closed") == {}

    def test_closures_accumulate(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "l.closed"
        for job_id in ("101", "102", "103"):
            ledger.append_closure(path, _closed(job_id, "COMPLETED", _AT))
        assert sorted(ledger.read_closures(path)) == ["101", "102", "103"]

    def test_a_repeated_job_resolves_to_the_later_record(self, tmp_path: pathlib.Path) -> None:
        """Unlike the ledger a duplicate is harmless: both say it ended."""
        path = tmp_path / "l.closed"
        ledger.append_closure(path, _closed("101", "FAILED", _AT))
        ledger.append_closure(path, _closed("101", "COMPLETED", "later"))
        assert ledger.read_closures(path)["101"]["state"] == "COMPLETED"

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "l.closed"
        ledger.append_closure(path, _closed("101", "COMPLETED", _AT))
        write_file(path, path.read_bytes() + b"\n\n")
        assert len(ledger.read_closures(path)) == 1

    def test_a_malformed_line_fails_the_read(self, tmp_path: pathlib.Path) -> None:
        """Skipping it would resurrect a finished job as an unaccounted finding."""
        path = tmp_path / "l.closed"
        ledger.append_closure(path, _closed("101", "COMPLETED", _AT))
        write_file(path, path.read_bytes() + b'{"job_id": "102"}\n')
        with pytest.raises(JSONTypeError):
            ledger.read_closures(path)


class TestClosureContract:
    def test_a_valid_closure_round_trips(self) -> None:
        payload: dict[str, JSONValue] = {
            "job_id": "101",
            "state": "COMPLETED",
            "closed_at": _AT,
            "elapsed_seconds": 1618,
        }
        assert encode_closure(decode_closure(payload)) == payload

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_closure("101")

    def test_an_unrecognised_state_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_closure(
                {"job_id": "1", "state": "VANISHED", "closed_at": _AT, "elapsed_seconds": None}
            )

    def test_an_empty_job_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_closure(
                {"job_id": "", "state": "COMPLETED", "closed_at": _AT, "elapsed_seconds": None}
            )

    def test_an_empty_timestamp_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_closure(
                {"job_id": "1", "state": "COMPLETED", "closed_at": "", "elapsed_seconds": None}
            )
