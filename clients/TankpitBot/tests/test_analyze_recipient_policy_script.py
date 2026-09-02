"""Tests for scripts.analyze_recipient_policy.

Sessions are real JSON on disk read through the real production hooks;
only the logging seam is injected, because setting up rich logging is a
process-wide side effect and not the thing under test.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.analyze_recipient_policy import DEFAULT_DIRECTORIES, main

from scripts import _test_hooks as script_hooks
from tankpit_bot.analysis.recipient_policy_types import VERDICT_BROADCAST
from tests.analysis._capture_fixtures import (
    FOREIGN_TANK,
    OWN_TANK,
    _build_pickup,
    _payload,
    _received,
    _session_json,
    _tank_info,
    _write,
)


@pytest.fixture(autouse=True)
def _quiet_logging() -> Generator[None, None, None]:
    """Swap the logging seam for a no-op and restore it after.

    Save-and-restore on the hook attribute, never a monkeypatch — the
    guard bans the latter and the seam exists for exactly this.
    """
    real = script_hooks.setup_rich_logging

    def _silent(level: script_hooks.LogLevel) -> None:
        """Absorb the logging call without configuring handlers.

        Args:
            level: Ignored.
        """

    script_hooks.setup_rich_logging = _silent
    try:
        yield
    finally:
        script_hooks.setup_rich_logging = real


@pytest.fixture()
def _argv() -> Generator[None, None, None]:
    """Restore ``sys.argv`` after a test rewrites it."""
    old = sys.argv
    try:
        yield
    finally:
        sys.argv = old


def _archive(tmp_path: Path) -> Path:
    """Write one decodable capture carrying a foreign 0x42.

    Args:
        tmp_path: Directory to write into.

    Returns:
        The directory holding the capture.
    """
    _write(
        tmp_path,
        "a.capture_session.json",
        _session_json(
            messages=[
                _received(_payload(_tank_info(OWN_TANK))),
                _received(_payload(_build_pickup(FOREIGN_TANK))),
            ]
        ),
    )
    return tmp_path


def test_main_reports_every_family_and_its_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """A named directory is swept and each family's verdict printed."""
    sys.argv = ["analyze_recipient_policy", str(_archive(tmp_path))]
    main()
    output = capsys.readouterr().out
    assert "sessions_examined=1" in output
    assert "sessions_decoded=1" in output
    assert f"0x42 BuildPickup -> {VERDICT_BROADCAST}" in output
    assert "foreign_actor_hits=1" in output


def test_main_sweeps_both_archive_directories_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """With no arguments the two default archive directories are used.

    Asserted through the ``No such directory`` refusal so the test does
    not depend on the developer's own ``runs/`` tree existing.
    """
    assert (Path("runs") / "bot", Path("runs") / "sniff") == DEFAULT_DIRECTORIES
    sys.argv = ["analyze_recipient_policy"]
    real_exists = script_hooks.path_exists
    missing: list[Path] = []

    def _never_exists(path: Path) -> bool:
        """Report every path absent, recording what was asked for.

        Args:
            path: Directory the script checked.

        Returns:
            Always False.
        """
        missing.append(path)
        return False

    script_hooks.path_exists = _never_exists
    try:
        with pytest.raises(SystemExit) as excinfo:
            main()
    finally:
        script_hooks.path_exists = real_exists

    assert excinfo.value.code == 1
    assert missing == [Path("runs") / "bot"]
    assert "No such directory" in capsys.readouterr().out


def test_main_refuses_a_directory_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """A missing directory names itself and exits non-zero.

    Sweeping on regardless would report a verdict over an archive the
    run never read.
    """
    absent = tmp_path / "nope"
    sys.argv = ["analyze_recipient_policy", str(absent)]
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert f"No such directory: {absent}" in capsys.readouterr().out


def test_module_runs_as_a_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """The ``__main__`` entry point drives the same sweep."""
    sys.argv = ["analyze_recipient_policy", str(_archive(tmp_path))]
    sys.modules.pop("scripts.analyze_recipient_policy", None)
    runpy.run_module("scripts.analyze_recipient_policy", run_name="__main__")
    assert "sessions_decoded=1" in capsys.readouterr().out
