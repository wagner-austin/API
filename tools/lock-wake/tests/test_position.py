"""The position file: one integer, read strictly, derived beside the journal."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError

from lock_wake.position import position_path, read_offset, write_offset


class TestPositionPath:
    def test_derives_beside_the_journal(self) -> None:
        journal = pathlib.Path("C:/repo/.fleet-events.jsonl")
        assert position_path(journal) == pathlib.Path(
            "C:/repo/.fleet-events.jsonl.lock-wake-offset.json"
        )


class TestReadOffset:
    def test_an_absent_file_reads_as_zero(self, tmp_path: pathlib.Path) -> None:
        assert read_offset(tmp_path / "never-written.json") == 0

    def test_round_trips_what_write_offset_wrote(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        write_offset(path, 16952)
        assert read_offset(path) == 16952

    def test_a_non_object_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        path.write_text("42\n", encoding="utf-8")
        with pytest.raises(JSONTypeError, match="not an object"):
            read_offset(path)

    def test_a_negative_offset_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "offset.json"
        path.write_text('{"offset": -3}\n', encoding="utf-8")
        with pytest.raises(JSONTypeError, match="negative"):
            read_offset(path)
