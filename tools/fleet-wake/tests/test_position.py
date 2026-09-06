"""The position record: what stops a dispatch being announced twice.

Every test here writes and reads through the PRODUCTION hooks against a real
file under ``tmp_path``, because the thing being checked is durability across
cycles and an in-memory double would prove nothing about it.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    load_json_str,
)

from fleet_wake import _test_hooks
from fleet_wake.position import (
    AnnouncedRun,
    append_announced,
    decode_announced_run,
    encode_announced_run,
    position_path,
    read_announced,
)

RECORD = AnnouncedRun(run_id="tools-fleet-1788633781", outcome="passed", announced_unix=1788700000)


class TestTheRecordSurvivesEncoding:
    def test_a_record_round_trips_through_json(self) -> None:
        assert (
            decode_announced_run(load_json_str(dump_json_str(encode_announced_run(RECORD))))
            == RECORD
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_announced_run(["tools-fleet-1788633781"])

    def test_a_missing_field_is_refused(self) -> None:
        encoded = encode_announced_run(RECORD)
        del encoded["outcome"]

        with pytest.raises(JSONTypeError):
            decode_announced_run(encoded)

    def test_a_mistyped_timestamp_is_refused(self) -> None:
        """Recorded as an int because the fleet ledger records its own times
        that way; a string here would compare wrongly against them."""
        with pytest.raises(JSONTypeError):
            decode_announced_run({**encode_announced_run(RECORD), "announced_unix": "now"})


class TestThePositionFile:
    def test_it_lives_beside_the_ledger(self, tmp_path: pathlib.Path) -> None:
        """So a workspace carries its bridge's memory with it, and moving one
        moves both."""
        assert position_path(tmp_path / "runs" / "ledger.jsonl") == (
            tmp_path / "runs" / "announced.jsonl"
        )

    def test_an_absent_file_reads_as_nothing_announced(self, tmp_path: pathlib.Path) -> None:
        """A workspace whose bridge has never run has announced nothing.
        Refusing the first cycle for having no history would make the bridge
        impossible to start using."""
        assert read_announced(tmp_path / "announced.jsonl") == frozenset()

    def test_appended_ids_are_read_back(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "announced.jsonl"
        append_announced(path, RECORD)
        append_announced(path, {**RECORD, "run_id": "libs-platform_core-1788669961"})

        assert read_announced(path) == frozenset(
            {"tools-fleet-1788633781", "libs-platform_core-1788669961"}
        )

    def test_a_trailing_newline_is_not_a_record(self, tmp_path: pathlib.Path) -> None:
        """The writer ends every line with one, so every file it has written
        ends this way.

        ``splitlines`` already drops the final empty piece, so this does NOT
        exercise the blank-line guard -- the test below does that with a real
        blank line. Kept separate because believing this one covered both is
        how the guard ends up untested while looking tested.
        """
        path = tmp_path / "announced.jsonl"
        append_announced(path, RECORD)

        assert path.read_text(encoding="utf-8").endswith("\n")
        assert read_announced(path) == frozenset({"tools-fleet-1788633781"})

    def test_a_blank_line_between_records_is_skipped(self, tmp_path: pathlib.Path) -> None:
        """A partially-written line followed by a fresh append leaves one,
        and it carries no record -- unlike a malformed line, which does and
        must be fatal."""
        path = tmp_path / "announced.jsonl"
        append_announced(path, RECORD)
        _test_hooks.append_text(path, "")
        append_announced(path, {**RECORD, "run_id": "second"})

        assert read_announced(path) == frozenset({"tools-fleet-1788633781", "second"})

    def test_the_parent_directory_is_created_on_first_write(self, tmp_path: pathlib.Path) -> None:
        """The position file is created by its first write, and a workspace
        pointing at a fresh runs/ directory is the ordinary first-run case."""
        path = tmp_path / "fresh" / "announced.jsonl"

        append_announced(path, RECORD)

        assert read_announced(path) == frozenset({"tools-fleet-1788633781"})


class TestAMalformedLineIsFatal:
    def test_a_line_that_is_not_an_object_names_the_line(self, tmp_path: pathlib.Path) -> None:
        """NOT SKIPPED. A line read as absent means the dispatch it names is
        announced a second time, and the reader of that post cannot tell it
        from a genuinely new ending."""
        path = tmp_path / "announced.jsonl"
        append_announced(path, RECORD)
        _test_hooks.append_text(path, '"just-a-string"')

        with pytest.raises(JSONTypeError, match=r"line 2 is a str, not an object"):
            read_announced(path)

    def test_a_line_that_is_not_json_raises(self, tmp_path: pathlib.Path) -> None:
        """``InvalidJsonError``, not ``JSONTypeError``: the two are distinct
        and the parser raises before any narrowing happens. Pinned to the
        actual type rather than to ``Exception``, so a change in which one
        surfaces is visible here."""
        path = tmp_path / "announced.jsonl"
        _test_hooks.append_text(path, "{not json")

        with pytest.raises(InvalidJsonError):
            read_announced(path)
