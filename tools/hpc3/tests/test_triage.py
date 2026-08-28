"""Tests for triage: the three conditions that look like health.

The reason fixtures are the ones measured on HPC3 on 2026-08-22, where 261 of
621 pending GPU jobs sat on ``DependencyNeverSatisfied`` and 3 on
``Resources``. Those two look identical in ``squeue``'s state column.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.account import AccountJob, decode_account_job
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.pending import (
    PendingJob,
    decode_pending_job,
    encode_pending_job,
    is_blocked,
    reason_code,
)
from hpc3.contracts.status import JobStatus
from hpc3.core.squeue import parse_squeue_output, parse_squeue_row, squeue_command
from hpc3.core.triage import (
    blocked_jobs,
    live_entries,
    silent_jobs,
    unaccounted_jobs,
    unclaimed_jobs,
)
from tests.against_hpc3 import decode_job_status, decode_ledger_entry
from tests.conftest import ledger_row


def _entry(job_id: str, name: str = "arm") -> LedgerEntry:
    """Build a ledger entry.

    Args:
        job_id: Job id.
        name: Job name.

    Returns:
        A validated entry.
    """
    return decode_ledger_entry(ledger_row(job_id=job_id, name=name))


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
        "cpu_count": 8,
        "node_list": "n1",
    }
    base.update(overrides)
    return decode_job_status(base)


def _account(job_id: str, name: str = "arm", state: str = "RUNNING") -> AccountJob:
    """Build a row from the account enumeration.

    Args:
        job_id: Job id.
        name: Job name, verbatim from the cluster.
        state: Slurm state.

    Returns:
        A validated account job.
    """
    return decode_account_job({"job_id": job_id, "name": name, "state": state})


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

    def test_the_decorated_reservation_reason_is_transient(self) -> None:
        """The exact string HPC3 returned for six pending jobs on 2026-08-28.

        `ReqNodeNotAvail` was in the allowlist and could never match it, so
        every job waiting on a reservation was reported blocked -- an entry
        that read as a decision while doing nothing.
        """
        assert is_blocked("ReqNodeNotAvail, May be reserved for other job") is False

    def test_the_other_decorated_form_is_transient_too(self) -> None:
        assert is_blocked("ReqNodeNotAvail, UnavailableNodes:hpc3-gpu-l54-05") is False

    def test_the_killer_reason_carries_no_comma_and_is_untouched(self) -> None:
        """The split must not reach DependencyNeverSatisfied, which is the
        261-job failure this check was built for."""
        assert reason_code("DependencyNeverSatisfied") == "DependencyNeverSatisfied"
        assert is_blocked("DependencyNeverSatisfied") is True

    def test_a_decorated_unknown_reason_is_still_blocked(self) -> None:
        """Splitting recognises decoration; it does not admit new reasons."""
        assert is_blocked("SomeReasonInventedNextYear, with detail") is True

    def test_the_code_is_the_text_before_the_first_comma(self) -> None:
        assert reason_code("ReqNodeNotAvail, May be reserved for other job") == "ReqNodeNotAvail"
        assert reason_code("  Resources  ") == "Resources"
        assert reason_code("") == ""


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


class TestUnclaimedJobs:
    """The mirror of unaccounted, and the direction that went unbuilt.

    Caught a real one the first time it was run against HPC3:
    ``55645549 img.abl-sif-v22``, an image build started from a login node by
    the raw ``ssh <host> sbatch`` this package's own README prescribes.
    """

    def test_a_job_on_the_cluster_with_no_ledger_row_is_found(self) -> None:
        findings = unclaimed_jobs([_entry("101")], [_account("101"), _account("999")])
        assert [f.job_id for f in findings] == ["999"]
        assert findings[0].kind == "unclaimed"

    def test_the_measured_build_job_is_found(self) -> None:
        """The exact row the cluster returned on 2026-08-28."""
        build = _account("55645549", name="img.abl-sif-v22")
        findings = unclaimed_jobs([_entry("55645374")], [build])
        assert [(f.job_id, f.name) for f in findings] == [("55645549", "img.abl-sif-v22")]

    def test_an_unprefixed_name_is_still_found(self) -> None:
        """Matching only `<project>.<name>` would be the natural narrowing and
        would defeat the check: a job that bypassed this package is under no
        obligation to be named the way this package names things."""
        findings = unclaimed_jobs([], [_account("999", name="build.sbatch")])
        assert [f.job_id for f in findings] == ["999"]

    def test_an_empty_ledger_reports_every_job_on_the_cluster(self) -> None:
        """The strongest form of the condition, not a reason to skip it."""
        findings = unclaimed_jobs([], [_account("101"), _account("102")])
        assert [f.job_id for f in findings] == ["101", "102"]

    def test_an_empty_cluster_yields_nothing(self) -> None:
        assert unclaimed_jobs([_entry("101")], []) == []

    def test_a_fully_claimed_cluster_yields_nothing(self) -> None:
        assert unclaimed_jobs([_entry("101"), _entry("102")], [_account("101")]) == []

    def test_a_closed_job_still_claims_its_row(self) -> None:
        """This takes the WHOLE ledger, not the open subset. squeue holds a
        job for minutes after it ends, so filtering by closure would report
        every just-finished job as though nobody had submitted it."""
        assert unclaimed_jobs([_entry("101")], [_account("101", state="COMPLETING")]) == []

    def test_the_detail_carries_the_state_and_says_what_happened(self) -> None:
        """The difference between something to stop and something to cancel
        before it starts."""
        findings = unclaimed_jobs([], [_account("999", state="PENDING")])
        assert "PENDING" in findings[0].detail
        assert "no ledger row claims it" in findings[0].detail

    def test_findings_keep_the_order_the_cluster_reported(self) -> None:
        account = [_account("103"), _account("101"), _account("102")]
        assert [f.job_id for f in unclaimed_jobs([], account)] == ["103", "101", "102"]


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

    def test_the_detail_names_cores_for_a_job_holding_no_gpu(self) -> None:
        """ "0 GPU(s)" would read as a broken allocation on a CPU job."""
        findings = silent_jobs(
            [_status("101", "RUNNING", gpu_count=0, cpu_count=16)],
            {"101": 4000},
            quiet_seconds=1800,
        )
        assert "16 core(s)" in findings[0].detail
        assert "GPU" not in findings[0].detail


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
