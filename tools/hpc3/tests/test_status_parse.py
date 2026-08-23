"""Tests for parsing Slurm accounting output.

The fixtures are real rows measured from HPC3 on 2026-08-22, including the
two shapes that break naive parsing: a state carrying a cancelling uid, and
an empty ``AllocTRES`` on a job that never allocated.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.status import (
    SACCT_FIELDS,
    parse_elapsed_seconds,
    parse_state,
    parse_tres_int,
    sacct_command,
)
from tests.against_hpc3 import parse_sacct_output, parse_sacct_row

_REAL_ROW = (
    "55519937|abl-verify|free-gpu32|COMPLETED|48|"
    "billing=11,cpu=11,gres/gpu:rtx6000=1,gres/gpu=1,mem=64G,node=1|hpc3-gpu-n54-00"
)


class TestSacctCommand:
    def test_it_requests_every_parsed_field(self) -> None:
        command = sacct_command("55519937")
        for field in SACCT_FIELDS:
            assert field in command

    def test_it_restricts_to_the_job_row(self) -> None:
        """Without -X, batch and extern steps triple every result."""
        assert " -X" in sacct_command("55519937")

    def test_it_asks_for_parseable_output(self) -> None:
        command = sacct_command("55519937")
        assert " -n " in command
        assert " -P " in command


class TestParseTresInt:
    def test_it_reads_the_measured_row(self) -> None:
        assert parse_tres_int("billing=11,cpu=11,mem=64G,node=1", "billing") == 11

    def test_a_pending_job_with_no_allocation_bills_nothing(self) -> None:
        assert parse_tres_int("", "billing") == 0

    def test_an_allocation_without_billing_reports_zero(self) -> None:
        assert parse_tres_int("cpu=4,mem=16G,node=1", "billing") == 0

    def test_billing_last_in_the_list_is_still_found(self) -> None:
        assert parse_tres_int("cpu=4,mem=16G,billing=4", "billing") == 4

    def test_a_non_numeric_billing_value_is_refused(self) -> None:
        """Defaulting to zero here would report a billed job as free."""
        with pytest.raises(AppError) as excinfo:
            parse_tres_int("billing=many,cpu=4", "billing")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE

    def test_a_negative_billing_value_is_refused(self) -> None:
        with pytest.raises(AppError):
            parse_tres_int("billing=-4,cpu=4", "billing")

    def test_a_bare_token_without_a_value_is_skipped(self) -> None:
        assert parse_tres_int("billing,cpu=4,billing=7", "billing") == 7


class TestParseElapsedSeconds:
    def test_it_reads_a_count(self) -> None:
        assert parse_elapsed_seconds("48") == 48

    def test_it_tolerates_surrounding_whitespace(self) -> None:
        assert parse_elapsed_seconds("  48 ") == 48

    def test_zero_is_valid_for_a_pending_job(self) -> None:
        assert parse_elapsed_seconds("0") == 0

    def test_a_clock_format_is_refused(self) -> None:
        """ElapsedRaw is seconds; Elapsed is HH:MM:SS and is not what we asked for."""
        with pytest.raises(AppError) as excinfo:
            parse_elapsed_seconds("00:00:48")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE


class TestParseState:
    def test_a_plain_state_passes_through(self) -> None:
        assert parse_state("COMPLETED") == "COMPLETED"

    def test_a_cancelling_uid_suffix_is_dropped(self) -> None:
        assert parse_state("CANCELLED by 1880454") == "CANCELLED"

    def test_it_uppercases(self) -> None:
        assert parse_state("completed") == "COMPLETED"

    def test_an_empty_field_yields_an_empty_state(self) -> None:
        assert parse_state("   ") == ""


class TestParseSacctRow:
    def test_it_parses_the_measured_row(self) -> None:
        status = parse_sacct_row(_REAL_ROW)
        assert status["job_id"] == "55519937"
        assert status["name"] == "abl-verify"
        assert status["partition"] == "free-gpu32"
        assert status["state"] == "COMPLETED"
        assert status["elapsed_seconds"] == 48
        assert status["billing_tres"] == 11
        assert status["node_list"] == "hpc3-gpu-n54-00"

    def test_a_pending_row_parses(self) -> None:
        status = parse_sacct_row("55517763|abl-smoke|free-gpu|PENDING|0||")
        assert status["state"] == "PENDING"
        assert status["billing_tres"] == 0
        assert status["node_list"] == ""

    def test_a_cancelled_row_keeps_only_the_state(self) -> None:
        row = "55517700|x|free-gpu|CANCELLED by 1880454|12|billing=4|hpc3-gpu-16-00"
        assert parse_sacct_row(row)["state"] == "CANCELLED"

    def test_too_few_columns_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_sacct_row("55519937|abl-verify|free-gpu")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE

    def test_too_many_columns_is_refused(self) -> None:
        with pytest.raises(AppError):
            parse_sacct_row(_REAL_ROW + "|extra")


class TestParseSacctOutput:
    def test_it_parses_several_rows(self) -> None:
        output = _REAL_ROW + "\n" + "55517763|abl-smoke|free-gpu|PENDING|0||" + "\n"
        rows = parse_sacct_output(output)
        assert [row["job_id"] for row in rows] == ["55519937", "55517763"]

    def test_blank_lines_are_skipped(self) -> None:
        assert len(parse_sacct_output(f"\n{_REAL_ROW}\n\n")) == 1

    def test_empty_output_yields_no_rows(self) -> None:
        assert parse_sacct_output("\n  \n") == []

    def test_one_bad_row_fails_the_whole_parse(self) -> None:
        """A partial list reads as 'these are the jobs' and hides the missing one."""
        with pytest.raises(AppError):
            parse_sacct_output(f"{_REAL_ROW}\nbroken|row\n")


class TestMultiJobQuery:
    def test_several_ids_become_one_comma_separated_query(self) -> None:
        """Six separate calls would observe six different moments."""
        command = sacct_command(["101", "102", "103"])
        assert "sacct -j 101,102,103 " in command

    def test_a_single_id_still_works(self) -> None:
        assert "sacct -j 55519937 " in sacct_command(["55519937"])

    def test_no_ids_is_refused(self) -> None:
        """An id-less query would return every job the user ever ran."""
        with pytest.raises(ValueError, match="at least one job id"):
            sacct_command([])


class TestGpuTresParsing:
    def test_the_gpu_count_is_read_from_alloc_tres(self) -> None:
        assert parse_tres_int("billing=11,cpu=11,gres/gpu=2,mem=64G", "gres/gpu") == 2

    def test_a_typed_gres_entry_is_not_read_as_the_untyped_one(self) -> None:
        """Both appear in the same list; reading one as the other double-counts."""
        alloc = "billing=11,cpu=11,gres/gpu:rtx6000=1,gres/gpu=1,mem=64G"
        assert parse_tres_int(alloc, "gres/gpu") == 1

    def test_a_cpu_only_job_holds_no_gpus(self) -> None:
        assert parse_tres_int("billing=43,cpu=43,mem=256G,node=1", "gres/gpu") == 0

    def test_a_row_carries_both_billing_and_gpu_count(self) -> None:
        status = parse_sacct_row(_REAL_ROW)
        assert status["billing_tres"] == 11
        assert status["gpu_count"] == 1
