"""Tests for triage: the three conditions that look like health.

The reason fixtures are the ones measured on HPC3 on 2026-08-22, where 261 of
621 pending GPU jobs sat on ``DependencyNeverSatisfied`` and 3 on
``Resources``. Those two look identical in ``squeue``'s state column.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.pending import PendingJob, decode_pending_job, encode_pending_job, is_blocked
from hpc3.contracts.status import JobStatus
from hpc3.core.squeue import parse_squeue_output, parse_squeue_row, squeue_command
from hpc3.core.triage import blocked_jobs, live_entries, silent_jobs, unaccounted_jobs
from tests.against_hpc3 import decode_job_status, decode_ledger_entry


def _entry(job_id: str, name: str = "arm") -> LedgerEntry:
    """Build a ledger entry.

    Args:
        job_id: Job id.
        name: Job name.

    Returns:
        A validated entry.
    """
    return decode_ledger_entry(
        {
            "job_id": job_id,
            "project": "abl",
            "name": name,
            "host": "hpc3",
            "partition": "free-gpu",
            "submitted_at": "2026-08-22T16:00:00+00:00",
            "log_dir": "/pub/logs",
            "deterministic": False,
            "experiment": {"arm": "B"},
        }
    )


def _status(job_id: str, state: str, **overrides: JSONValue) -> JobStatus:
    """Build an accounting row.

    Args:
        job_id: Job id.
        state: State to report.
        **overrides: Fields to replace.

    Returns:
        A validated status.
    """
    base: dict[str, JSONValue] = {
        "job_id": job_id,
        "name": "arm",
        "partition": "free-gpu",
        "state": state,
        "elapsed_seconds": 60,
        "billing_tres": 8,
        "gpu_count": 1,
        "node_list": "n1",
    }
    base.update(overrides)
    return decode_job_status(base)


def _pending(job_id: str, reason: str) -> PendingJob:
    """Build a pending row.

    Args:
        job_id: Job id.
        reason: The scheduler's reason.

    Returns:
        A validated pending job.
    """
    return decode_pending_job({"job_id": job_id, "name": "arm", "reason": reason})


class TestIsBlocked:
    def test_the_measured_killer_reason_is_blocked(self) -> None:
        """261 of 621 pending GPU jobs on HPC3 sat on exactly this."""
        assert is_blocked("DependencyNeverSatisfied") is True

    def test_waiting_on_resources_is_not_blocked(self) -> None:
        assert is_blocked("Resources") is False

    def test_our_own_queue_limits_are_transient(self) -> None:
        """Our finishing jobs release these; waiting is correct."""
        for reason in ("QOSMaxJobsPerUserLimit", "JobArrayTaskLimit", "Priority"):
            assert is_blocked(reason) is False

    def test_a_resolvable_dependency_is_transient(self) -> None:
        assert is_blocked("Dependency") is False

    def test_a_held_job_is_blocked(self) -> None:
        assert is_blocked("JobHeldUser") is True

    def test_an_empty_reason_is_transient(self) -> None:
        assert is_blocked("") is False
        assert is_blocked("None") is False

    def test_an_unrecognised_reason_is_treated_as_blocked(self) -> None:
        """A reason we have never seen is where patience costs a week."""
        assert is_blocked("SomeReasonInventedNextYear") is True

    def test_surrounding_whitespace_does_not_change_the_verdict(self) -> None:
        assert is_blocked("  Resources  ") is False


class TestBlockedJobs:
    def test_it_reports_only_the_blocked_ones(self) -> None:
        pending = [
            _pending("101", "Resources"),
            _pending("102", "DependencyNeverSatisfied"),
            _pending("103", "Priority"),
        ]
        findings = blocked_jobs(pending)
        assert [f.job_id for f in findings] == ["102"]
        assert findings[0].kind == "blocked"
        assert "DependencyNeverSatisfied" in findings[0].detail

    def test_a_healthy_queue_yields_nothing(self) -> None:
        assert blocked_jobs([_pending("101", "Resources")]) == []


class TestUnaccountedJobs:
    def test_a_recorded_job_the_cluster_never_heard_of_is_found(self) -> None:
        """The condition no cluster-side query can detect."""
        findings = unaccounted_jobs([_entry("101"), _entry("102")], [_status("101", "RUNNING")])
        assert [f.job_id for f in findings] == ["102"]
        assert findings[0].kind == "unaccounted"

    def test_everything_accounted_yields_nothing(self) -> None:
        assert unaccounted_jobs([_entry("101")], [_status("101", "COMPLETED")]) == []

    def test_the_detail_names_where_it_went(self) -> None:
        findings = unaccounted_jobs([_entry("101")], [])
        assert "hpc3" in findings[0].detail


class TestSilentJobs:
    def test_a_running_job_with_a_stale_log_is_found(self) -> None:
        findings = silent_jobs([_status("101", "RUNNING")], {"101": 4000}, quiet_seconds=1800)
        assert [f.job_id for f in findings] == ["101"]
        assert findings[0].kind == "silent"

    def test_a_recently_written_log_is_healthy(self) -> None:
        assert silent_jobs([_status("101", "RUNNING")], {"101": 30}, quiet_seconds=1800) == []

    def test_exactly_the_threshold_is_healthy(self) -> None:
        assert silent_jobs([_status("101", "RUNNING")], {"101": 1800}, quiet_seconds=1800) == []

    def test_a_job_with_no_reading_is_skipped_not_assumed_silent(self) -> None:
        """No reading is not the same as a bad reading."""
        assert silent_jobs([_status("101", "RUNNING")], {}, quiet_seconds=1800) == []

    def test_a_finished_job_is_never_silent(self) -> None:
        stale = {"101": 99999}
        assert silent_jobs([_status("101", "COMPLETED")], stale, quiet_seconds=1800) == []

    def test_a_pending_job_is_never_silent(self) -> None:
        stale = {"101": 99999}
        assert silent_jobs([_status("101", "PENDING")], stale, quiet_seconds=1800) == []

    def test_the_detail_names_the_gpus_it_is_holding(self) -> None:
        findings = silent_jobs(
            [_status("101", "RUNNING", gpu_count=4)], {"101": 4000}, quiet_seconds=1800
        )
        assert "4 GPU(s)" in findings[0].detail


class TestLiveEntries:
    def test_finished_jobs_drop_out(self) -> None:
        entries = [_entry("101"), _entry("102")]
        statuses = [_status("101", "COMPLETED"), _status("102", "RUNNING")]
        assert [e["job_id"] for e in live_entries(entries, statuses)] == ["102"]

    def test_an_unaccounted_job_counts_as_live(self) -> None:
        """It might be running; the cluster simply has not said."""
        assert [e["job_id"] for e in live_entries([_entry("101")], [])] == ["101"]

    def test_a_requeued_job_counts_as_live(self) -> None:
        statuses = [_status("101", "REQUEUED")]
        assert len(live_entries([_entry("101")], statuses)) == 1


class TestSqueueParsing:
    def test_it_parses_a_row(self) -> None:
        job = parse_squeue_row("101|arm-b-42|DependencyNeverSatisfied")
        assert job == {"job_id": "101", "name": "arm-b-42", "reason": "DependencyNeverSatisfied"}

    def test_it_parses_several_rows(self) -> None:
        output = "101|a|Resources\n102|b|Priority\n"
        assert [j["job_id"] for j in parse_squeue_output(output)] == ["101", "102"]

    def test_an_empty_queue_yields_nothing(self) -> None:
        """The normal healthy case: none of our jobs is pending."""
        assert parse_squeue_output("") == []

    def test_a_malformed_row_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_squeue_row("101|arm")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE

    def test_one_bad_row_fails_the_parse(self) -> None:
        with pytest.raises(AppError):
            parse_squeue_output("101|a|Resources\nbroken\n")

    def test_an_empty_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            parse_squeue_row("|arm|Resources")

    def test_the_command_restricts_to_pending_and_to_our_ids(self) -> None:
        command = squeue_command(["101", "102"])
        assert "-j 101,102" in command
        assert "-t PD" in command

    def test_no_ids_is_refused(self) -> None:
        """An id-less query reports 1,400 rows of other people's work."""
        with pytest.raises(ValueError, match="at least one job id"):
            squeue_command([])


class TestPendingContract:
    def test_a_valid_row_round_trips(self) -> None:
        payload: dict[str, JSONValue] = {"job_id": "1", "name": "a", "reason": "Resources"}
        assert encode_pending_job(decode_pending_job(payload)) == payload

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_pending_job("Resources")

    def test_an_empty_reason_is_allowed(self) -> None:
        """squeue reports blank for a job the scheduler has not looked at yet."""
        assert decode_pending_job({"job_id": "1", "name": "a", "reason": ""})["reason"] == ""
