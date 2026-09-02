"""Tests for scripts.analyze_response_shapes.

Both archives are injected through the analysis IO seam rather than
read off disk. That is not only for speed: the script's live side is
deliberately fixed to the real ``runs/`` archive, so a test that let it
resolve would sweep 341 real sessions per assertion and its result
would drift with the corpus. Injecting both sides makes the diff
deterministic and the assertions exact.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.analyze_response_shapes import (
    DEFAULT_SIM_DIRECTORIES,
    LIVE_DIRECTORIES,
    ROW_LIMIT,
    main,
)

from scripts import _test_hooks as script_hooks
from tankpit_bot.analysis import _test_hooks as analysis_hooks
from tankpit_bot.analysis.response_shapes import (
    analyze_response_shapes,
    format_response_shape_diff,
)
from tankpit_bot.protocol.commands import build_teleport_command
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _command,
    _payload,
    _radar_result,
    _received,
    _sent,
    _session_json,
    _tank_info,
)

SIM_DIR = Path("injected-sim")

#: The live session's teleport draws a 0x46; the sim's draws nothing.
#: One shape per side, so the diff carries exactly one row of each
#: verdict and the assertions can be exact rather than substring.
_LIVE_SESSION = _session_json(
    messages=[
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
        _received(_payload(_radar_result()), timestamp_ms=1200),
    ]
)
_SIM_SESSION = _session_json(
    messages=[
        _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
        _sent(_payload(_command(build_teleport_command(9, 9))), timestamp_ms=1100),
    ]
)


@pytest.fixture(autouse=True)
def _injected_world() -> Generator[None, None, None]:
    """Inject the logging, path and archive seams; restore them after.

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
        """Report every archive directory this test names as present.

        Args:
            path: Directory the script checked.

        Returns:
            True for the live directories and the injected sim one.
        """
        return path in LIVE_DIRECTORIES or path == SIM_DIR

    def _list(directory: Path) -> list[Path]:
        """Enumerate one capture per archive directory.

        Args:
            directory: Directory being scanned.

        Returns:
            A single capture path inside it.
        """
        return [directory / "injected.capture_session.json"]

    def _read(path: Path) -> str:
        """Serve the live or sim session by which archive asked.

        Args:
            path: Capture path being read.

        Returns:
            The session JSON for that side.
        """
        return _SIM_SESSION if path.parent == SIM_DIR else _LIVE_SESSION

    script_hooks.setup_rich_logging = _silent
    script_hooks.path_exists = _exists
    analysis_hooks.set_analysis_hooks(read_text_fn=_read, list_session_paths_fn=_list)
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


def test_the_live_side_is_the_real_archive() -> None:
    """The live directories are fixed; only the sim side is an argument.

    The useful question is "how faithful is THIS sim", so the argument
    names the side that varies — and a mixed ``runs/sim`` answers it
    about several past sims at once.
    """
    assert (Path("runs") / "bot", Path("runs") / "sniff") == LIVE_DIRECTORIES
    assert (Path("runs") / "sim",) == DEFAULT_SIM_DIRECTORIES


def test_main_names_both_verdicts_for_a_named_sim_archive(
    capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """The live-only shape is a missing law, the sim-only one invented."""
    sys.argv = ["analyze_response_shapes", str(SIM_DIR)]
    main()
    output = capsys.readouterr().out

    # Two live sessions (runs/bot + runs/sniff) against one sim session.
    assert output.startswith("live_sessions=2 live_windows=2\nsim_sessions=1 sim_windows=1\n")
    assert "MISSING LAWS (real server does it, sim never does): 1" in output
    assert "INVENTED LAWS (sim does it, archive never shows it): 1" in output
    assert "teleport           46" in output
    assert "teleport           (silent)" in output


def test_with_no_argument_the_sim_side_defaults_to_the_run_archive(
    capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """No argument means ``runs/sim`` — whatever it happens to hold.

    Asserted through the refusal so the test does not depend on the
    developer's own ``runs/`` tree: the injected world reports only the
    live directories and the named sim one as present, so the default
    resolves, is checked, and is refused by name.
    """
    sys.argv = ["analyze_response_shapes"]
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert capsys.readouterr().out == f"No such directory: {Path('runs') / 'sim'}\n"


def test_main_refuses_a_sim_directory_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _argv: None
) -> None:
    """A missing directory names itself and exits non-zero.

    Sweeping on would report a fidelity verdict over an archive the run
    never read — and an empty sim side makes every live shape read as a
    missing law, which looks like a result.
    """
    absent = tmp_path / "nope"
    sys.argv = ["analyze_response_shapes", str(absent)]
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert capsys.readouterr().out == f"No such directory: {absent}\n"


def test_module_runs_as_a_script(capsys: pytest.CaptureFixture[str], _argv: None) -> None:
    """The ``__main__`` entry point drives the same diff."""
    sys.argv = ["analyze_response_shapes", str(SIM_DIR)]
    sys.modules.pop("scripts.analyze_response_shapes", None)
    runpy.run_module("scripts.analyze_response_shapes", run_name="__main__")
    diff = analyze_response_shapes(list(LIVE_DIRECTORIES), [SIM_DIR])
    expected = format_response_shape_diff(diff, ROW_LIMIT) + "\n"
    assert capsys.readouterr().out == expected
