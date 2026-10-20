"""The report-stream writer, against real stdout."""

from __future__ import annotations

import pytest

from platform_core.report_line import emit_error_line, emit_line


def test_emit_line_writes_one_newline_terminated_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_line("one line")
    emit_line("two")
    assert capsys.readouterr() == ("one line\ntwo\n", "")


def test_emit_error_line_writes_to_the_error_stream_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_error_line("refused")
    emit_error_line("again")
    assert capsys.readouterr() == ("", "refused\nagain\n")
