"""The production hook implementations, exercised directly."""

from __future__ import annotations

import datetime

import pytest

from hpc_wake import _test_hooks


class TestDefaults:
    def test_emit_writes_one_flushed_line_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _test_hooks.emit("a line")
        assert capsys.readouterr().out == "a line\n"

    def test_now_iso_is_utc_with_an_explicit_offset(self) -> None:
        stamp = _test_hooks.now_iso()
        parsed = datetime.datetime.fromisoformat(stamp)
        assert parsed.utcoffset() == datetime.timedelta(0)

    def test_reset_restores_every_default(self) -> None:
        held: list[str] = []

        def _capture(line: str) -> None:
            held.append(line)

        def _frozen() -> str:
            return "2026-09-06T07:00:00+00:00"

        _test_hooks.emit = _capture
        _test_hooks.now_iso = _frozen
        _test_hooks.http_post = _test_hooks.http_post

        _test_hooks.reset_hooks()

        assert _test_hooks.emit is not _capture
        assert _test_hooks.now_iso is not _frozen
        assert _test_hooks.now_iso() != "2026-09-06T07:00:00+00:00"
