"""The pump's production publisher runner, run for real through failures.

Every run_cycle test replaces ``run_process`` with a recording fake, so none
of them shows the real runner returning a publisher's nonzero status and
words, or killing one that never returns. These call the implementation
itself, ``_default_run_process`` (``effect-seam-twin``, board task cc7222ca).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
from platform_core.config import config_test_hooks
from scripts import _test_hooks


def test_a_red_publisher_returns_its_status_and_both_streams(tmp_path: pathlib.Path) -> None:
    """A red half is logged and the tick goes on, so it must come back."""
    environment = dict(config_test_hooks.get_environment())
    environment["PUMP_PROBE"] = "from-env"
    completed = _test_hooks._default_run_process(
        [
            sys.executable,
            "-c",
            "import os, pathlib, sys; print(os.environ['PUMP_PROBE'], pathlib.Path.cwd().name);"
            " sys.stderr.write('board refused'); sys.exit(3)",
        ],
        cwd=tmp_path,
        env=environment,
        timeout=60,
    )
    assert completed.returncode == 3
    assert completed.stdout.split() == ["from-env", tmp_path.name]
    assert completed.stderr == "board refused"


def test_a_publisher_past_its_wall_is_killed_and_ends_the_tick(tmp_path: pathlib.Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        _test_hooks._default_run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=config_test_hooks.get_environment(),
            timeout=1,
        )
