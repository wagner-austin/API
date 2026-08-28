"""Tests for the account enumeration: what the cluster holds, whoever put it there.

The row fixtures are the shape ``squeue --me -h -o '%i|%j|%T'`` really returned
from HPC3 on 2026-08-28, including the one that mattered:
``55645549|img.abl-sif-v22|RUNNING`` -- an image build started by a raw
``ssh <host> sbatch`` from a login node, holding eight cores, with no ledger
row anywhere.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.account import AccountJob, decode_account_job, encode_account_job
from hpc3.core.squeue import (
    ACCOUNT_FORMAT,
    account_command,
    parse_account_output,
    parse_account_row,
)

_REAL_OUTPUT = """55645549|img.abl-sif-v22|RUNNING
55645430|turkic-lstm.bases-fi|PENDING
55645374|turkic-lstm.bases-tr|RUNNING
"""


class TestTheQuery:
    def test_it_names_no_job_ids(self) -> None:
        """An id-restricted query cannot return a job we do not know about,
        which is the only thing this query is for."""
        assert account_command() == "squeue --me -h -o '%i|%j|%T'"

    def test_it_asks_for_state_rather_than_reason(self) -> None:
        """A running job's reason is 'None', which would make it look like a
        job the scheduler has not looked at yet."""
        assert ACCOUNT_FORMAT == "%i|%j|%T"

    def test_it_scopes_to_the_authenticated_account(self) -> None:
        """`--me` rather than `-u <name>`: the workspace declares an SSH
        destination and no username, and a username here would be a second
        place for the account's identity to be wrong."""
        assert "--me" in account_command()
        assert "-u " not in account_command()


class TestParsingWhatTheClusterReturned:
    def test_the_real_three_row_output_parses(self) -> None:
        jobs = parse_account_output(_REAL_OUTPUT)
        assert [j["job_id"] for j in jobs] == ["55645549", "55645430", "55645374"]
        assert [j["state"] for j in jobs] == ["RUNNING", "PENDING", "RUNNING"]

    def test_the_build_job_keeps_its_unprefixed_name(self) -> None:
        """Nothing here requires `<project>.<name>`; demanding it is what
        would make a bypassed job invisible."""
        assert parse_account_output(_REAL_OUTPUT)[0]["name"] == "img.abl-sif-v22"

    def test_an_empty_account_parses_to_nothing(self) -> None:
        """The normal state between runs, and not an error."""
        assert parse_account_output("") == []

    def test_blank_lines_are_skipped(self) -> None:
        assert len(parse_account_output("1|a|RUNNING\n\n2|b|PENDING\n")) == 2

    def test_surrounding_whitespace_is_stripped(self) -> None:
        job = parse_account_row("  55645549 | img.abl-sif-v22 | RUNNING  ")
        assert job == AccountJob(job_id="55645549", name="img.abl-sif-v22", state="RUNNING")

    def test_a_row_with_the_wrong_column_count_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_account_row("55645549|img.abl-sif-v22")
        assert excinfo.value.code is Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE

    def test_the_refusal_names_the_format_that_produced_the_row(self) -> None:
        """Two queries share this parser; a reader needs to know which one
        returned something unreadable."""
        with pytest.raises(AppError) as excinfo:
            parse_account_row("55645549|img.abl-sif-v22")
        assert ACCOUNT_FORMAT in str(excinfo.value)

    def test_one_bad_row_fails_the_whole_enumeration(self) -> None:
        """A partial enumeration reads as a COMPLETE list of what the account
        holds, and the row that failed is the one that would have been
        reported as unrecorded."""
        with pytest.raises(AppError):
            parse_account_output("1|a|RUNNING\nbroken\n")


class TestTheContract:
    def test_it_round_trips(self) -> None:
        job = AccountJob(job_id="55645549", name="img.abl-sif-v22", state="RUNNING")
        assert decode_account_job(encode_account_job(job)) == job

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_account_job(["55645549"])

    def test_every_field_is_required(self) -> None:
        for missing in ("job_id", "name", "state"):
            record: dict[str, JSONValue] = {"job_id": "1", "name": "a", "state": "RUNNING"}
            del record[missing]
            with pytest.raises(JSONTypeError):
                decode_account_job(record)

    def test_no_field_may_be_empty(self) -> None:
        """Unlike a pending reason: this row exists because the cluster
        volunteered it, so a blank field means the parse is wrong rather than
        that the scheduler has not looked yet."""
        for blank in ("job_id", "name", "state"):
            record: dict[str, JSONValue] = {"job_id": "1", "name": "a", "state": "RUNNING"}
            record[blank] = ""
            with pytest.raises(JSONTypeError, match="must not be empty"):
                decode_account_job(record)

    def test_a_mistyped_field_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_account_job({"job_id": 55645549, "name": "a", "state": "RUNNING"})
