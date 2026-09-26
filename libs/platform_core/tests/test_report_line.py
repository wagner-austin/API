"""The report-stream writer, against real stdout."""

from __future__ import annotations

import pytest

from platform_core.report_line import emit_line


def test_emit_line_writes_one_newline_terminated_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_line("one line")
    emit_line("two")
    assert capsys.readouterr().out == "one line\ntwo\n"
