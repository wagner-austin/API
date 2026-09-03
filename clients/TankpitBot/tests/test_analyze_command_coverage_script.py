"""Tests for scripts.analyze_command_coverage.

The archive is injected through the analysis IO seam rather than read
off disk: the script's default is the real ``runs/`` tree, so letting
it resolve would sweep 343 sessions per assertion and the result would
drift with the corpus.

The exit code is the point of these tests. The audit is not a report —
an unmapped command byte is a crash waiting for the first real client,
so the script has to FAIL, not merely mention it.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.analyze_command_coverage import DEFAULT_DIRECTORIES, main

from scripts import _test_hooks as script_hooks
from tankpit_bot.analysis import _test_hooks as analysis_hooks
from tankpit_bot.analysis.command_coverage import (
    analyze_command_coverage,
    format_command_coverage,
)
from tankpit_bot.protocol.command_builders import build_query_command
from tankpit_bot.protocol.commands import CMD_RADAR
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _command,
    _payload,
    _received,
    _sent,
    _session_json,
    _tank_info,
)

ARCHIVE = Path("injected-archive")

#: The script writes one after the report; the assertions compare the
#: whole stream, so they have to account for it.
_TRAILING_NEWLINE = chr(10)

#: A byte no constant names and the decoder cannot map.
_UNKNOWN_BYTE = 0xFE

_HANDLED_SESSION = _session_json(
    messages=[
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_query_command(CMD_RADAR))), timestamp_ms=1100),
    ]
)
_CRASHING_SESSION = _session_json(
    messages=[
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_query_command(_UNKNOWN_BYTE))), timestamp_ms=1100),
    ]
)


def _install(session_json: str) -> None:
    """Point the analysis seam at one injected session.

    Args:
        session_json: The session every capture path serves.
    """

    def _list(directory: Path) -> list[Path]:
        """Enumerate one capture per directory.

        Args:
            directory: Directory being scanned.

        Returns:
            A single capture path inside it.
        """
        return [directory / "injected.capture_session.json"]

    def _read(path: Path) -> str:
        """Serve the injected session.

        Args:
            path: Ignored.

        Returns:
            The session JSON.
        """
        del path
        return session_json

    analysis_hooks.set_analysis_hooks(read_text_fn=_read, list_session_paths_fn=_list)


@pytest.fixture(autouse=True)
def _seams() -> Generator[None, None, None]:
    """Inject the logging and path seams; restore them after.

    Save-and-restore on the hook attributes, never a monkeypatch — the
    guard bans the latter and the seams exist for exactly this.
    """
    real_logging = script_hooks.setup_rich_logging
    real_exists = script_hooks.path_exists

    def _silent(level: script_hooks.LogLevel) -> None:
        """Absorb the logging call without configuring handlers.

        Args:
            level: Ignored.
        """

    def _exists(path: Path) -> bool:
        """Report the injected archive and the defaults as present.

        Args:
            path: Directory the script checked.

        Returns:
            True for the injected directory and the real defaults.
        """
        return path == ARCHIVE or path in DEFAULT_DIRECTORIES

    script_hooks.setup_rich_logging = _silent
    script_hooks.path_exists = _exists
    try:
        yield
    finally:
        script_hooks.setup_rich_logging = real_logging
        script_hooks.path_exists = real_exists
        analysis_hooks.reset_analysis_hooks()


@pytest.fixture()
def _argv() -> Generator[None, None, None]:
    """Restore ``sys.argv`` after a test rewrites it."""
    old = sys.argv
    try:
        yield
    finally:
        sys.argv = old


def test_the_default_archive_is_the_real_one_including_the_sniffs() -> None:
    """The sniffs are what make this audit work at all.

    They are REAL CLIENT traffic, and the commands our bot never sends
    are exactly the ones that crash the sim. Auditing only ``runs/bot``
    would ask our own bot whether it surprises us.
    """
    assert (Path("runs") / "bot", Path("runs") / "sniff") == DEFAULT_DIRECTORIES


def test_a_clean_archive_reports_and_exits_zero(
    capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """Nothing unmapped: say so plainly and do not fail the build."""
    _install(_HANDLED_SESSION)
    sys.argv = ["analyze_command_coverage", str(ARCHIVE)]

    main()

    expected = format_command_coverage(analyze_command_coverage([ARCHIVE])) + _TRAILING_NEWLINE
    assert capsys.readouterr().out == expected


def test_an_unmapped_byte_exits_nonzero(capsys: pytest.CaptureFixture[str], _argv: None) -> None:
    """THE POINT. An unmapped byte fails, it does not merely report.

    A hosted server dies on the first client that sends one, so a
    green audit that mentioned it in passing would be worse than no
    audit at all.
    """
    _install(_CRASHING_SESSION)
    sys.argv = ["analyze_command_coverage", str(ARCHIVE)]

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    expected = format_command_coverage(analyze_command_coverage([ARCHIVE])) + _TRAILING_NEWLINE
    assert capsys.readouterr().out == expected


def test_a_missing_directory_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """Auditing on regardless would report coverage over nothing.

    An empty archive has no unmapped bytes, so the run would pass —
    the most dangerous possible answer.
    """
    absent = tmp_path / "nope"
    sys.argv = ["analyze_command_coverage", str(absent)]

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    assert capsys.readouterr().out == f"No such directory: {absent}\n"


def test_module_runs_as_a_script(capsys: pytest.CaptureFixture[str], _argv: None) -> None:
    """The ``__main__`` entry point drives the same audit."""
    _install(_HANDLED_SESSION)
    sys.argv = ["analyze_command_coverage", str(ARCHIVE)]
    sys.modules.pop("scripts.analyze_command_coverage", None)

    runpy.run_module("scripts.analyze_command_coverage", run_name="__main__")

    expected = format_command_coverage(analyze_command_coverage([ARCHIVE])) + _TRAILING_NEWLINE
    assert capsys.readouterr().out == expected


def test_with_no_argument_the_audit_reads_the_real_archive(
    capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """No argument means the whole live corpus, bot AND sniff.

    Asserted through the injected seam so the test does not depend on
    the developer's own ``runs/`` tree: both defaults resolve, both are
    read, and the report covers them together.
    """
    _install(_HANDLED_SESSION)
    sys.argv = ["analyze_command_coverage"]

    main()

    expected = format_command_coverage(analyze_command_coverage(list(DEFAULT_DIRECTORIES)))
    assert capsys.readouterr().out == expected + _TRAILING_NEWLINE
