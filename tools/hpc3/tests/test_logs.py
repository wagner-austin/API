"""Tests for log-staleness measurement.

The clock comes from the cluster, not this machine. A few minutes of skew
between them would either invent staleness or hide it, and both failures
look like a correct reading.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.ledger import LedgerEntry
from hpc3.core.logs import CLOCK_PROBE, age_commands, log_ages, log_path, parse_ages
from hpc3.core.remote import MAX_COMMAND_CHARS
from tests.against_hpc3 import decode_ledger_entry
from tests.conftest import FakeRun, ledger_row


def _entry(job_id: str, name: str = "abl.arm-b-42") -> LedgerEntry:
    """Build a ledger entry.

    Args:
        job_id: Job id.
        name: The qualified job name, as the ledger stores it.

    Returns:
        A validated entry.
    """
    return decode_ledger_entry(ledger_row(job_id=job_id, name=name))


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
        assert "date +%s" in age_commands([_entry("101")])[0]

    def test_it_stats_each_log(self) -> None:
        commands = age_commands([_entry("101"), _entry("102")])
        assert len(commands) == 1
        assert "/pub/logs/abl.arm-b-42-101.out" in commands[0]
        assert "/pub/logs/abl.arm-b-42-102.out" in commands[0]

    def test_a_missing_log_emits_nothing_rather_than_a_zero(self) -> None:
        """A job whose output has not appeared has not been quiet."""
        assert "if [ -f " in age_commands([_entry("101")])[0]

    def test_a_wide_probe_splits_and_every_batch_reads_its_own_clock(self) -> None:
        """Measured on the real cluster 2026-09-05, and it is NOT the local
        argv limit: the ~29 KB single command was accepted by CreateProcess,
        sent, and arrived at bash TRUNCATED mid-quote --
        `bash: -c: line 1: unexpected EOF while looking for matching "'"`.

        An age is `now - mtime`. A batch that inherited another batch's clock
        would be subtracting across a different instant, so each carries its
        own."""
        entries = [_entry(str(index), name=f"abl.arm-{index}") for index in range(400)]
        commands = age_commands(entries)
        assert len(commands) > 1
        assert all(len(command) <= MAX_COMMAND_CHARS for command in commands)
        assert all(command.startswith(CLOCK_PROBE) for command in commands)
        assert sum(command.count("if [ -f ") for command in commands) == 400

    def test_a_missing_log_does_not_fail_the_whole_query(self) -> None:
        """Measured on the real cluster, not deduced.

        The first form was ``test -f PATH && echo ...``. It emits the right
        thing, and when the LAST probe's file is missing the failed test
        becomes the command's exit status, so the whole age query is reported
        as a failed remote command. A job that was submitted and never ran has
        no log -- which is exactly the job this reconciliation exists to find,
        so the failure landed on the case that matters most.

        An ``if`` block exits zero whether or not the file is there.
        """
        command = age_commands([_entry("101"), _entry("102")])[0]
        assert "&&" not in command
        assert command.count("if [ -f ") == 2
        assert command.count("; fi") == 2

    def test_no_entries_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            age_commands([])


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

    def test_every_batch_is_asked_and_a_late_batch_still_reports(self, fake_run: FakeRun) -> None:
        """A split that asked only the first batch would report every job
        after it as 'not writing yet' -- a wedged job read as health, which
        is the exact condition this measurement exists to catch."""
        entries = [_entry(str(index), name=f"abl.arm-{index}") for index in range(400)]
        commands = age_commands(entries)
        assert len(commands) > 1
        late = [entry for entry in entries if log_path(entry) in commands[-1]][-1]
        fake_run.add(log_path(late), stdout=f"now 1000\n{late['job_id']} 900\n", once=True)
        fake_run.add("date +%s", stdout="now 1000\n0 400\n")
        assert log_ages("hpc3", entries) == {"0": 600, late["job_id"]: 100}
        assert len(fake_run.calls) == len(commands)
