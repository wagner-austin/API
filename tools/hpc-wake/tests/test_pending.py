"""The pending store, against real files under ``tmp_path``.

The production ``hpc3.core._test_hooks`` defaults do the I/O here, exactly
as ``hpc3``'s own ledger suite exercises them. Nothing is faked: a store
tested against an in-memory stand-in would not have caught that
``write_text`` requires its parent to exist, which is the one property this
module depends on and does not control.
"""

from __future__ import annotations

import pathlib

import pytest
from hpc3.contracts.closure import Closure
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from hpc_wake.pending import (
    PendingClosure,
    decode_pending,
    encode_pending,
    pending_path,
    read_pending,
    write_pending,
)


def a_record(job_id: str = "55802014_1", observed_epoch: int = 1757200000) -> PendingClosure:
    """Build one pending record.

    Args:
        job_id: The job's id.
        observed_epoch: When this bridge first saw it terminal.

    Returns:
        The record.
    """
    return PendingClosure(
        closure=Closure(
            job_id=job_id,
            state="COMPLETED",
            closed_at="2026-09-07T00:00:00+00:00",
            elapsed_seconds=4175,
        ),
        observed_epoch=observed_epoch,
    )


class TestRoundTrip:
    """Encode and decode, against the real contract."""

    def test_a_record_survives_encoding_and_decoding_unchanged(self) -> None:
        original = a_record()
        assert decode_pending(encode_pending(original)) == original

    def test_a_closure_with_an_unrecorded_runtime_round_trips(self) -> None:
        # ``elapsed_seconds`` is a three-state field in hpc3's contract and
        # None is a real thing to have been, not a zero.
        original = a_record()
        original["closure"]["elapsed_seconds"] = None
        assert decode_pending(encode_pending(original))["closure"]["elapsed_seconds"] is None


class TestDecodeRefusals:
    """Every way a record can be malformed, refused rather than softened."""

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_pending([1, 2])

    def test_a_missing_closure_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_pending({"observed_epoch": 1})

    def test_a_missing_observation_time_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_pending({"closure": encode_pending(a_record())["closure"]})

    def test_a_negative_observation_time_is_refused(self) -> None:
        """A negative epoch makes every group answer 'aged' immediately.

        That would announce each group the instant it was created -- the
        exact defect this package was changed to remove, wearing the
        costume of a fix. It is refused at the edge rather than left to
        produce a plausible-looking cycle.
        """
        encoded = encode_pending(a_record())
        encoded["observed_epoch"] = -1
        with pytest.raises(JSONTypeError, match="must not be negative"):
            decode_pending(encoded)

    def test_a_malformed_closure_inside_a_well_formed_record_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_pending({"closure": {"job_id": ""}, "observed_epoch": 1})


class TestFile:
    """Reading and writing the record on disk."""

    def test_the_path_is_derived_from_the_ledger(self) -> None:
        assert pending_path(pathlib.Path("/runs/hpc3.jsonl")).name == "hpc3.jsonl.pending"

    def test_an_absent_file_reads_as_nothing_waiting(self, tmp_path: pathlib.Path) -> None:
        assert read_pending(tmp_path / "never-written.pending") == []

    def test_records_survive_a_write_and_a_read(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "hpc3.jsonl.pending"
        records = [a_record("1", 100), a_record("2", 200)]
        write_pending(path, records)
        assert read_pending(path) == records

    def test_a_write_replaces_rather_than_appends(self, tmp_path: pathlib.Path) -> None:
        """The property that keeps this out of the append-only ledger.

        Records leave the pending file when their group is announced. An
        appending writer would leave the announced ones behind and
        re-announce them on the next cycle forever.
        """
        path = tmp_path / "hpc3.jsonl.pending"
        write_pending(path, [a_record("1", 100), a_record("2", 200)])
        write_pending(path, [a_record("2", 200)])
        assert [r["closure"]["job_id"] for r in read_pending(path)] == ["2"]

    def test_writing_nothing_leaves_an_empty_file_rather_than_deleting_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "hpc3.jsonl.pending"
        write_pending(path, [a_record()])
        write_pending(path, [])
        assert path.is_file()
        assert read_pending(path) == []

    def test_blank_lines_are_skipped_but_content_is_not(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "hpc3.jsonl.pending"
        write_pending(path, [a_record("1", 100)])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(read_pending(path)) == 1

    def test_a_malformed_line_fails_the_read_rather_than_being_skipped(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A skipped line is an ending that will never be announced.

        Silence is the failure this bridge exists to remove, so an
        unreadable record stops the cycle instead of quietly shrinking the
        waiting set.
        """
        path = tmp_path / "hpc3.jsonl.pending"
        write_pending(path, [a_record("1", 100)])
        path.write_text(
            path.read_text(encoding="utf-8") + '{"closure": {}, "observed_epoch": 1}\n',
            encoding="utf-8",
        )
        with pytest.raises(JSONTypeError):
            read_pending(path)

    def test_a_line_that_is_not_json_fails_the_read(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "hpc3.jsonl.pending"
        path.write_text("not json at all\n", encoding="utf-8")
        with pytest.raises(InvalidJsonError):
            read_pending(path)
