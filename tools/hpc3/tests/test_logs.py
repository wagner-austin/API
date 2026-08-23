"""Tests for log-staleness measurement.

The clock comes from the cluster, not this machine. A few minutes of skew
between them would either invent staleness or hide it, and both failures
look like a correct reading.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.ledger import LedgerEntry
from hpc3.core.logs import age_command, log_ages, log_path, parse_ages
from tests.against_hpc3 import decode_ledger_entry
from tests.conftest import FakeRun


def _entry(job_id: str, name: str = "abl.arm-b-42") -> LedgerEntry:
    """Build a ledger entry.

    Args:
        job_id: Job id.
        name: The qualified job name, as the ledger stores it.

    Returns:
        A validated entry.
    """
    base: dict[str, JSONValue] = {
        "job_id": job_id,
        "project": "abl",
        "name": name,
        "host": "hpc3",
        "partition": "free-gpu",
        "submitted_at": "2026-08-22T16:00:00+00:00",
        "log_dir": "/pub/logs",
        "experiment": {"arm": "B"},
    }
    return decode_ledger_entry(base)


class TestLogPath:
    def test_it_reconstructs_what_sbatch_was_told_to_write(self) -> None:
        assert log_path(_entry("101")) == "/pub/logs/abl.arm-b-42-101.out"

    def test_two_projects_writing_the_same_name_do_not_collide(self) -> None:
        """The ledger stores the qualified name, so the path is qualified too."""
        mine = log_path(_entry("101", name="abl.train"))
        theirs = log_path(_entry("102", name="sirius.train"))
        assert mine != theirs


class TestAgeCommand:
    def test_it_reads_the_cluster_clock(self) -> None:
        """Comparing against this machine's clock would invent staleness."""
        assert "date +%s" in age_command([_entry("101")])

    def test_it_stats_each_log(self) -> None:
        command = age_command([_entry("101"), _entry("102")])
        assert "/pub/logs/abl.arm-b-42-101.out" in command
        assert "/pub/logs/abl.arm-b-42-102.out" in command

    def test_a_missing_log_emits_nothing_rather_than_a_zero(self) -> None:
        """A job whose output has not appeared has not been quiet."""
        assert "test -f" in age_command([_entry("101")])

    def test_no_entries_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            age_command([])


class TestParseAges:
    def test_it_subtracts_mtime_from_the_cluster_clock(self) -> None:
        assert parse_ages("now 1000\n101 400\n") == {"101": 600}

    def test_several_jobs_are_read(self) -> None:
        assert parse_ages("now 1000\n101 400\n102 900\n") == {"101": 600, "102": 100}

    def test_a_job_with_no_log_is_absent_not_zero(self) -> None:
        """Absent means 'not writing yet'; zero would mean 'written just now'."""
        ages = parse_ages("now 1000\n101 400\n")
        assert "102" not in ages

    def test_no_logs_at_all_yields_an_empty_mapping(self) -> None:
        assert parse_ages("now 1000\n") == {}

    def test_a_missing_cluster_clock_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_ages("101 400\n")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE

    def test_a_non_numeric_timestamp_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_ages("now 1000\n101 yesterday\n")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE

    def test_unexpected_lines_are_skipped(self) -> None:
        assert parse_ages("stat: cannot stat\nnow 1000\n101 400\n") == {"101": 600}


class TestLogAges:
    def test_it_queries_and_parses(self, fake_run: FakeRun) -> None:
        fake_run.add("date +%s", stdout="now 1000\n101 400\n")
        assert log_ages("hpc3", [_entry("101")]) == {"101": 600}

    def test_no_entries_makes_no_remote_call(self, fake_run: FakeRun) -> None:
        assert log_ages("hpc3", []) == {}
        assert fake_run.calls == []
