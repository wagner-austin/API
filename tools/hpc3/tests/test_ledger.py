"""Tests for the local ledger.

The failure this file guards against is a job that runs for ten hours on a
shared machine with nobody able to find it, because the only thing that knew
its id was a process that exited.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.ledger import LedgerEntry, encode_ledger_entry
from hpc3.core import ledger
from tests.against_hpc3 import decode_ledger_entry, read_ledger
from tests.conftest import write_file


def _entry(**overrides: str) -> LedgerEntry:
    """Build a ledger entry.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated entry.
    """
    base: dict[str, JSONValue] = {
        "job_id": "101",
        "project": "abl",
        "name": "arm-b-42",
        "host": "hpc3",
        "partition": "free-gpu",
        "submitted_at": "2026-08-22T16:00:00+00:00",
        "log_dir": "/pub/wagnera3/logs",
        "deterministic": False,
        "experiment": {"arm": "B", "seed": "42"},
    }
    base.update(overrides)
    return decode_ledger_entry(base)


class TestAppendAndRead:
    def test_a_written_entry_reads_back(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger.append(path, _entry())
        assert read_ledger(path) == [_entry()]

    def test_entries_accumulate_in_order(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger.append(path, _entry(job_id="101"))
        ledger.append(path, _entry(job_id="102"))
        ledger.append(path, _entry(job_id="103"))
        assert [e["job_id"] for e in read_ledger(path)] == ["101", "102", "103"]

    def test_it_creates_parent_directories(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "deep" / "nested" / "ledger.jsonl"
        ledger.append(path, _entry())
        assert len(read_ledger(path)) == 1

    def test_an_absent_ledger_reads_as_empty(self, tmp_path: pathlib.Path) -> None:
        """Nothing submitted yet is a real state, not an error."""
        assert read_ledger(tmp_path / "never-written.jsonl") == []

    def test_one_record_per_line(self, tmp_path: pathlib.Path) -> None:
        """A crash mid-sweep truncates at a line boundary, losing at most one."""
        path = tmp_path / "ledger.jsonl"
        ledger.append(path, _entry(job_id="101"))
        ledger.append(path, _entry(job_id="102"))
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger.append(path, _entry())
        write_file(path, path.read_bytes() + b"\n\n")
        assert len(read_ledger(path)) == 1

    def test_a_malformed_line_fails_the_read(self, tmp_path: pathlib.Path) -> None:
        """Skipping it would hide a job, which is what this file prevents."""
        path = tmp_path / "ledger.jsonl"
        ledger.append(path, _entry())
        write_file(path, path.read_bytes() + b'{"job_id": ""}\n')
        with pytest.raises(JSONTypeError):
            read_ledger(path)


class TestUnfinished:
    def test_it_excludes_jobs_reported_finished(self) -> None:
        entries = [_entry(job_id="101"), _entry(job_id="102"), _entry(job_id="103")]
        assert [e["job_id"] for e in ledger.unfinished(entries, ["102"])] == ["101", "103"]

    def test_everything_finished_leaves_nothing(self) -> None:
        entries = [_entry(job_id="101")]
        assert ledger.unfinished(entries, ["101"]) == []

    def test_nothing_finished_leaves_everything(self) -> None:
        entries = [_entry(job_id="101"), _entry(job_id="102")]
        assert len(ledger.unfinished(entries, [])) == 2


class TestLedgerEntryContract:
    def test_a_valid_entry_round_trips(self) -> None:
        payload: dict[str, JSONValue] = {
            "job_id": "101",
            "project": "abl",
            "name": "abl.arm-b-42",
            "host": "hpc3",
            "partition": "free-gpu",
            "submitted_at": "2026-08-22T16:00:00+00:00",
            "log_dir": "/pub/wagnera3/logs",
            "deterministic": False,
            "experiment": {"arm": "B", "seed": "42"},
            "image_digest": "sha256:" + "ab" * 32,
            "artifact": "/pub/wagnera3/floor/results/run.json",
        }
        assert encode_ledger_entry(decode_ledger_entry(payload)) == payload

    @staticmethod
    def _row() -> dict[str, JSONValue]:
        """A ledger row carrying only the fields that predate the index.

        Returns:
            The row, ready to extend with whatever a case is testing.
        """
        return {
            "job_id": "101",
            "project": "abl",
            "name": "abl.arm-b-42",
            "host": "hpc3",
            "partition": "free-gpu",
            "submitted_at": "2026-08-22T16:00:00+00:00",
            "log_dir": "/pub/wagnera3/logs",
            "deterministic": False,
            "experiment": {"arm": "B", "seed": "42"},
        }

    def test_a_row_written_before_the_index_fields_existed_still_reads(self) -> None:
        """The ledger is APPEND-ONLY and the live one predates both fields.

        Requiring them would make `hpc3-triage` and `hpc3-trace` fail to read
        the history they exist to read. Reading history is not comparing --
        the same reasoning that lets a training manifest decode without a
        fingerprint.
        """
        old_row: dict[str, JSONValue] = {
            "job_id": "101",
            "project": "abl",
            "name": "abl.arm-b-42",
            "host": "hpc3",
            "partition": "free-gpu",
            "submitted_at": "2026-08-22T16:00:00+00:00",
            "log_dir": "/pub/wagnera3/logs",
            "deterministic": False,
            "experiment": {"arm": "B", "seed": "42"},
        }

        entry = decode_ledger_entry(old_row)

        assert entry["image_digest"] == ""
        assert entry["artifact"] is None

    def test_a_mistyped_image_digest_is_a_corrupted_row_not_an_old_one(self) -> None:
        """Absent is history; present-and-wrong is corruption, and they differ."""
        with pytest.raises(JSONTypeError, match="must be a string or null"):
            decode_ledger_entry({**self._row(), "image_digest": 7})

    def test_a_mistyped_artifact_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string or null"):
            decode_ledger_entry({**self._row(), "artifact": 7})

    def test_an_artifact_naming_nowhere_is_refused(self) -> None:
        """Null says "declares none"; empty says "declares one, names nowhere"."""
        with pytest.raises(JSONTypeError, match="not an empty string"):
            decode_ledger_entry(
                {
                    "job_id": "101",
                    "project": "abl",
                    "name": "abl.arm-b-42",
                    "host": "hpc3",
                    "partition": "free-gpu",
                    "submitted_at": "2026-08-22T16:00:00+00:00",
                    "log_dir": "/pub/wagnera3/logs",
                    "deterministic": False,
                    "experiment": {"arm": "B", "seed": "42"},
                    "artifact": "",
                }
            )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_ledger_entry("101")

    def test_a_partition_this_cluster_lacks_is_refused(self) -> None:
        """A ledger written for another machine must not read as ours."""
        with pytest.raises(AppError) as excinfo:
            decode_ledger_entry(
                {
                    "job_id": "1",
                    "project": "abl",
                    "name": "abl.n",
                    "host": "h",
                    "partition": "turbo",
                    "submitted_at": "t",
                    "log_dir": "/l",
                    "deterministic": False,
                    "experiment": {"arm": "B"},
                }
            )
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_a_malformed_project_is_refused(self) -> None:
        """The project is what makes the row legible among 102 users' jobs."""
        with pytest.raises(JSONTypeError):
            decode_ledger_entry(
                {
                    "job_id": "1",
                    "project": "Abl/v2",
                    "name": "abl.n",
                    "host": "h",
                    "partition": "free-gpu",
                    "submitted_at": "t",
                    "log_dir": "/l",
                    "deterministic": False,
                    "experiment": {"arm": "B"},
                }
            )

    def test_every_field_must_be_present_and_non_empty(self) -> None:
        full: dict[str, JSONValue] = {
            "job_id": "1",
            "project": "abl",
            "name": "abl.n",
            "host": "h",
            "partition": "free-gpu",
            "submitted_at": "t",
            "log_dir": "/l",
            "deterministic": False,
            "experiment": {"arm": "B"},
        }
        for key in ("job_id", "project", "name", "host", "submitted_at", "log_dir"):
            broken = dict(full)
            broken[key] = ""
            with pytest.raises(JSONTypeError):
                decode_ledger_entry(broken)
