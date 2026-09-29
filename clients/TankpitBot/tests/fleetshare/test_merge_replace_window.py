"""Tests for reading a sibling report through the writer's replace window.

Every form the window takes is absorbed the same way: retried within the
budget, then the sibling is skipped for this exchange with a
``fleet_report_read_denied`` diagnostic naming each window seen. Any other
read failure, and a non-empty report that fails to decode, still raise.
"""

from __future__ import annotations

import errno
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from platform_core.json_utils import InvalidJsonError

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import ReadTextProtocol
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.fleetshare.merge import read_team_reports
from tankpit_bot.fleetshare.replace_window import ReplaceWindow
from tankpit_bot.fleetshare.report import FLEET_REPORT_FILENAME
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.runtime_records import RuntimeEventRecordDict
from tests.conftest import FakeFileSystem
from tests.fleetshare.test_merge import _NOW, _report, _write


@pytest.fixture(autouse=True)
def _restore_read_text() -> Generator[None, None, None]:
    """Put back the real reader a test replaced.

    Yields:
        None, with the original reader restored after.
    """
    real_read = _test_hooks.read_text
    yield
    _test_hooks.read_text = real_read


def _window_then_real(windows: list[Callable[[], str]]) -> ReadTextProtocol:
    """A reader that plays ``windows`` in order, then reads the real file.

    Args:
        windows: One callable per attempt, each raising or returning what
            the replace window showed that read.

    Returns:
        The reader to install as ``_test_hooks.read_text``.
    """
    real_read = _test_hooks.read_text
    pending = list(windows)

    def read(path: Path) -> str:
        if pending:
            return pending.pop(0)()
        return real_read(path)

    return read


def _denied() -> str:
    raise PermissionError(errno.EACCES, "Permission denied")


def _missing() -> str:
    raise FileNotFoundError(errno.ENOENT, "No such file or directory")


def _no_data() -> str:
    raise OSError(errno.ENODATA, "No data available")


def _empty() -> str:
    return ""


def _denied_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``fleet_report_read_denied`` record in the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "fleet_report_read_denied"
    ]


@pytest.mark.parametrize("window", [_denied, _missing, _no_data, _empty])
def test_one_window_then_the_retry_lands(
    fake_fs: FakeFileSystem, window: Callable[[], str]
) -> None:
    """Any single replace-window read is followed by a read of the fresh report.

    The PermissionError form is the arterial tick-264 crash (2026-08-26
    03:01:06) on a Windows host; the other three are the same window
    seen through the container fleet's Docker Desktop bind mount
    (board task b651224a).
    """
    _write(fake_fs, _report("artax"))
    _test_hooks.read_text = _window_then_real([window])

    reports = read_team_reports("arterial", 2, "6", _NOW)

    assert [report["instance"] for report in reports] == ["artax"]


def test_a_sibling_hidden_for_the_whole_budget_is_skipped_and_named(
    fake_fs: FakeFileSystem,
) -> None:
    """Three windows in a row skip the sibling, and the diagnostic names each.

    Live on sedona 2026-09-29 05:56:43, one ENODATA read killed bot p2 at
    tick 24 over one beat of advisory data that rewrites every ~2 s.
    """
    artifacts = configure_bot_runtime_logging("20260929-055600")
    _write(fake_fs, _report("artax"))
    _test_hooks.read_text = _window_then_real([_no_data, _missing, _empty])

    assert read_team_reports("arterial", 2, "6", _NOW) == []

    records = _denied_records(artifacts["latest_events_path"])
    assert len(records) == 1
    fields = records[0]["fields"]
    assert fields["attempts"] == 3
    assert fields["windows"] == ",".join(
        window.value
        for window in (ReplaceWindow.NO_DATA, ReplaceWindow.MISSING, ReplaceWindow.EMPTY)
    )
    assert fields["report_path"] == str(bot_run_dir("artax") / FLEET_REPORT_FILENAME)


def test_permission_denied_through_the_budget_is_named_too(fake_fs: FakeFileSystem) -> None:
    """The Windows-host form keeps its skip, now with its window named.

    Falsified premise (arterial tick 316, 2026-08-28 21:10): three
    immediate retries all landed inside one Windows replace swap.
    """
    artifacts = configure_bot_runtime_logging("20260929-055601")
    _write(fake_fs, _report("artax"))
    _test_hooks.read_text = _window_then_real([_denied, _denied, _denied])

    assert read_team_reports("arterial", 2, "6", _NOW) == []
    assert _denied_records(artifacts["latest_events_path"])[0]["fields"]["windows"] == (
        "permission_denied,permission_denied,permission_denied"
    )


def test_any_other_read_failure_still_raises(fake_fs: FakeFileSystem) -> None:
    """Only the replace window's forms are absorbed; an I/O error is not one."""
    _write(fake_fs, _report("artax"))

    def _io_error() -> str:
        raise OSError(errno.EIO, "Input/output error")

    _test_hooks.read_text = _window_then_real([_io_error])

    with pytest.raises(OSError, match="Input/output error"):
        read_team_reports("arterial", 2, "6", _NOW)


def test_a_non_empty_malformed_report_still_raises(fake_fs: FakeFileSystem) -> None:
    """The window only ever shows an EMPTY file; anything else malformed is a bug."""
    _write(fake_fs, _report("artax"))
    _test_hooks.read_text = _window_then_real([lambda: '{"instance": "art'])

    with pytest.raises(InvalidJsonError):
        read_team_reports("arterial", 2, "6", _NOW)
