"""The production hook implementations, exercised directly.

These are the bindings the pump's run actually uses, so they are tested
against the real filesystem and real stdout rather than described. A seam
whose production side is only ever replaced by a fake is a seam whose
production side is untested.
"""

from __future__ import annotations

import pathlib

import pytest

from lock_wake import _test_hooks
from lock_wake._test_hooks import (
    _default_emit,
    _default_file_exists,
    _default_read_bytes,
    _default_write_text,
)


class TestDefaults:
    def test_read_bytes_reads_raw_bytes(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "journal.jsonl"
        path.write_bytes(b'{"k":1}\n{"torn')
        assert _default_read_bytes(path) == b'{"k":1}\n{"torn'

    def test_write_text_replaces_and_creates_parents(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "deep" / "offset.json"
        _default_write_text(path, '{"offset": 1}\n')
        _default_write_text(path, '{"offset": 2}\n')
        assert path.read_text(encoding="utf-8") == '{"offset": 2}\n'

    def test_file_exists_distinguishes_files_from_absence(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "present.json"
        assert _default_file_exists(path) is False
        path.write_text("{}", encoding="utf-8")
        assert _default_file_exists(path) is True

    def test_emit_writes_one_flushed_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        _default_emit("one line")
        assert capsys.readouterr().out == "one line\n"

    def test_the_production_bindings_are_the_defaults(self) -> None:
        _test_hooks.reset_hooks()
        assert _test_hooks.read_bytes is _default_read_bytes
        assert _test_hooks.write_text is _default_write_text
        assert _test_hooks.file_exists is _default_file_exists
        assert _test_hooks.emit is _default_emit
