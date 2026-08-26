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
        "image_digest": None,
        "artifact": None,
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
        """A valid ledger row, index fields explicitly unrecorded.

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
            "image_digest": None,
            "artifact": None,
        }

    def test_a_row_omitting_an_index_field_is_refused(self) -> None:
        """Tolerating absence was the shim; the 122 historical rows were
        backfilled to explicit null instead.

        A reader that accepts absence accepts it forever, including from a
        writer that forgets the field tomorrow -- and those rows would decode
        as "unknown" silently. The backfill happened once and is in git; the
        tolerance would have been permanent.
        """
        for key in ("image_digest", "artifact"):
            broken = dict(self._row())
            broken.update({"image_digest": None, "artifact": None})
            del broken[key]
            with pytest.raises(JSONTypeError, match=f"Field '{key}' is required"):
                decode_ledger_entry(broken)

    def test_the_message_says_what_to_write_instead(self) -> None:
        """A required field nobody knows the value of needs an escape that is
        not "leave it out"."""
        broken = dict(self._row())
        del broken["artifact"]
        with pytest.raises(JSONTypeError, match="write null"):
            decode_ledger_entry(broken)

    def test_an_unrecorded_image_is_not_the_same_as_no_image(self) -> None:
        """Three states, because rows named `...-v4` demonstrably ran inside
        an image and record nothing. Collapsing null into "" would make every
        one of them assert it ran bare."""
        unrecorded = decode_ledger_entry({**self._row(), "image_digest": None, "artifact": None})
        directory = decode_ledger_entry({**self._row(), "image_digest": "", "artifact": None})

        assert unrecorded["image_digest"] is None
        assert directory["image_digest"] == ""

    def test_a_mistyped_image_digest_is_a_corrupted_row_not_an_old_one(self) -> None:
        """Absent is history; present-and-wrong is corruption, and they differ."""
        with pytest.raises(JSONTypeError, match="must be a string or null"):
            decode_ledger_entry({**self._row(), "image_digest": 7})

    def test_a_mistyped_artifact_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a string or null"):
            decode_ledger_entry({**self._row(), "artifact": 7})

    def test_an_artifact_naming_nowhere_is_refused(self) -> None:
        """Null says "names none"; empty says "names one, and it is nowhere"."""
        with pytest.raises(JSONTypeError, match="not an empty string"):
            decode_ledger_entry({**self._row(), "artifact": ""})

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_ledger_entry("101")

    def test_a_partition_this_cluster_lacks_is_refused(self) -> None:
        """A ledger written for another machine must not read as ours."""
        with pytest.raises(AppError) as excinfo:
            decode_ledger_entry({**self._row(), "partition": "turbo"})
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_a_malformed_project_is_refused(self) -> None:
        """The project is what makes the row legible among 102 users' jobs."""
        with pytest.raises(JSONTypeError):
            decode_ledger_entry({**self._row(), "project": "Abl/v2"})

    def test_every_field_must_be_present_and_non_empty(self) -> None:
        for key in ("job_id", "project", "name", "host", "submitted_at", "log_dir"):
            broken = dict(self._row())
            broken[key] = ""
            with pytest.raises(JSONTypeError):
                decode_ledger_entry(broken)
