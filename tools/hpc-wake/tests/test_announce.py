"""Grouping newly terminal jobs into posts, and the marker contract."""

from __future__ import annotations

import pytest
from hpc3.contracts.closure import Closure
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.status import JobState
from platform_core.error_codes import HpcWakeErrorCode
from platform_core.errors import AppError

from hpc_wake.announce import LINE_CAP, MARKER, announcements


def _entry(
    job_id: str, *, project: str = "mi", submitter: str | None = "label-a-0906"
) -> LedgerEntry:
    """Build a ledger entry with only the fields announcements read varying.

    Args:
        job_id: The job's id.
        project: The project it belongs to.
        submitter: The recorded board label, ``""``, or None for a
            pre-field row.

    Returns:
        The entry.
    """
    return LedgerEntry(
        job_id=job_id,
        project=project,
        name=f"{project}.job-{job_id}",
        host="hpc3",
        partition="free-gpu",
        submitted_at="2026-09-06T05:00:00+00:00",
        log_dir="/pub/w/logs",
        deterministic=False,
        experiment={"arm": "x"},
        image_digest="",
        submitter=submitter,
        artifact=None,
    )


def _closure(job_id: str, *, state: JobState = "COMPLETED", elapsed: int | None = 4688) -> Closure:
    """Build a closure.

    Args:
        job_id: The job that ended.
        state: How it ended.
        elapsed: Its recorded runtime, or None for unrecorded.

    Returns:
        The closure.
    """
    return Closure(
        job_id=job_id,
        state=state,
        closed_at="2026-09-06T07:00:00+00:00",
        elapsed_seconds=elapsed,
    )


class TestOneAnnouncement:
    def test_marker_leads_mention_trails_and_the_job_line_is_complete(self) -> None:
        result = announcements([_closure("101")], {"101": _entry("101")})
        assert len(result) == 1
        body = result[0]["body"]
        lines = body.splitlines()
        assert lines[0] == f"{MARKER} mi: 1 job(s) ended (COMPLETED x1)"
        assert lines[1] == "101 mi.job-101 COMPLETED 4688s"
        assert lines[-1] == "@label-a-0906 your job(s) reached terminal state"
        assert result[0]["submitter"] == "label-a-0906"
        assert result[0]["project"] == "mi"

    def test_an_unrecorded_elapsed_says_so_rather_than_inventing_zero(self) -> None:
        result = announcements([_closure("101", elapsed=None)], {"101": _entry("101")})
        assert "101 mi.job-101 COMPLETED elapsed unrecorded" in result[0]["body"]

    def test_a_blank_submitter_posts_without_a_mention(self) -> None:
        result = announcements([_closure("101")], {"101": _entry("101", submitter="")})
        assert result[0]["submitter"] == ""
        assert "@" not in result[0]["body"]

    def test_the_tally_counts_every_state(self) -> None:
        closures = [
            _closure("101"),
            _closure("102", state="FAILED", elapsed=12),
            _closure("103"),
        ]
        entries = {job_id: _entry(job_id) for job_id in ("101", "102", "103")}
        body = announcements(closures, entries)[0]["body"]
        assert "3 job(s) ended (COMPLETED x2, FAILED x1)" in body


class TestGrouping:
    def test_one_post_per_project_and_submitter_sorted(self) -> None:
        closures = [_closure("1"), _closure("2"), _closure("3")]
        entries = {
            "1": _entry("1", project="rusted", submitter="label-b-0906"),
            "2": _entry("2", project="mi", submitter="label-a-0906"),
            "3": _entry("3", project="mi", submitter="label-b-0906"),
        }
        result = announcements(closures, entries)
        assert [(a["project"], a["submitter"]) for a in result] == [
            ("mi", "label-a-0906"),
            ("mi", "label-b-0906"),
            ("rusted", "label-b-0906"),
        ]

    def test_a_pre_field_null_groups_with_declared_none(self) -> None:
        """Either way there is nobody to tag; two groups would be a
        distinction the post cannot act on."""
        closures = [_closure("1"), _closure("2")]
        entries = {
            "1": _entry("1", submitter=None),
            "2": _entry("2", submitter=""),
        }
        result = announcements(closures, entries)
        assert len(result) == 1
        assert result[0]["submitter"] == ""
        assert "2 job(s) ended" in result[0]["body"]

    def test_a_sweep_sized_group_is_capped_with_an_honest_remainder(self) -> None:
        count = LINE_CAP + 3
        ids = [str(i) for i in range(count)]
        closures = [_closure(job_id) for job_id in ids]
        entries = {job_id: _entry(job_id) for job_id in ids}
        body = announcements(closures, entries)[0]["body"]
        lines = body.splitlines()
        # header + LINE_CAP job lines + remainder + mention
        assert len(lines) == LINE_CAP + 3
        assert lines[-2] == "+3 more, all in the ledger's closure record"
        assert f"{count} job(s) ended" in lines[0]


class TestLedgerIntegrity:
    def test_a_terminal_job_the_ledger_never_recorded_is_a_defect(self) -> None:
        with pytest.raises(AppError) as caught:
            announcements([_closure("999")], {"101": _entry("101")})
        assert caught.value.code is HpcWakeErrorCode.JOB_UNKNOWN_TO_LEDGER
        assert "999" in caught.value.message
