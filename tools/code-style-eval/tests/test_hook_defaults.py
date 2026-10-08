"""The production checker runner, run for real against children that fail.

Every scoring test replaces ``Hooks.run_checker`` with a recorder, so none of
them shows the real runner returning a failing checker's status and output
rather than raising, or decoding a byte the locale codec cannot. These call
the implementation itself, ``_default_run_checker`` (``effect-seam-twin``,
board task cc7222ca).
"""

from __future__ import annotations

import pathlib
import sys

from platform_core.config import config_test_hooks

from code_style_eval.core import _test_hooks as core_hooks


def test_a_checker_reporting_findings_returns_its_status_and_output(
    tmp_path: pathlib.Path,
) -> None:
    """A nonzero exit is the measurement, so it comes back, never raised.

    The child writes byte 0x81, which cp1252 cannot map: the case a real
    sweep hit, where locale decoding lost the whole stream.
    """
    finished = core_hooks._default_run_checker(
        (
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'\\x81 finding'); sys.exit(1)",
        ),
        tmp_path,
        config_test_hooks.get_environment(),
    )
    assert finished.returncode == 1
    assert finished.stdout == "� finding"
    assert finished.stderr == ""


def test_the_checker_runs_in_its_directory_with_the_environment_given(
    tmp_path: pathlib.Path,
) -> None:
    environment = dict(config_test_hooks.get_environment())
    environment["MYPYPATH"] = "roots-for-the-sandbox"
    finished = core_hooks._default_run_checker(
        (
            sys.executable,
            "-c",
            "import os, pathlib; print(os.environ['MYPYPATH'], pathlib.Path.cwd().name)",
        ),
        tmp_path,
        environment,
    )
    assert finished.returncode == 0
    assert finished.stdout.split() == ["roots-for-the-sandbox", tmp_path.name]
