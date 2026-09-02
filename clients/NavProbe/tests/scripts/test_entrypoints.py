"""Tests for each measurement script's ``if __name__ == "__main__"`` block.

Run through :mod:`runpy` rather than a subprocess so the executed lines are the
ones under measurement. A subprocess would run the same code in a process the
tracer is not attached to, leaving the entry point unexercised while appearing
to test it -- and would also miss the installed hooks, so the scripts would
reach for a real GPU.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.apply_tactile_alias_patch import _LAUNCH_OLD, _MAX_OLD, _SIG_OLD

from tests.scripts.conftest import Harness, WitnessHarness

#: The directory the scripts live in.
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(name: str, argv: list[str]) -> int | str | None:
    """Execute one script as ``__main__`` under a synthetic command line.

    Args:
        name: The script's file name.
        argv: Arguments to place after the program name.

    Returns:
        The exit code the module raised.
    """
    original = sys.argv
    sys.argv = [name, *argv]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_path(str(SCRIPTS / name), run_name="__main__")
    finally:
        sys.argv = original
    return caught.value.code


class TestSweepEntryPoints:
    """The three Warp-driven scripts, against the installed fakes."""

    def test_gpu_deterministic_sweep_exits_zero(self, harness: Harness) -> None:
        """A completed sweep exits clean through the entry point."""
        assert _run("gpu_deterministic_sweep.py", ["RUN_TO_RUN", "cache"]) == 0

    def test_world_scaling_sweep_exits_zero(self, harness: Harness) -> None:
        """A completed ladder exits clean through the entry point."""
        assert _run("world_scaling_sweep.py", ["RUN_TO_RUN", "cache", "64", "256", "2"]) == 0

    def test_det_compile_test_exits_zero(self, harness: Harness) -> None:
        """A mode that compiles exits clean through the entry point."""
        assert _run("det_compile_test.py", ["RUN_TO_RUN", "cache"]) == 0

    def test_collision_pair_probe_exits_zero(self, witness_harness: WitnessHarness) -> None:
        """A completed pair sweep exits clean through the entry point.

        Driven by the witness harness rather than the shared one: this script
        loads its factory through ``load_witness_factory``, which the plain
        harness does not install.
        """
        assert _run("collision_pair_probe.py", ["RUN_TO_RUN", "cache", "4096"]) == 0


class TestPatchEntryPoint:
    """The patch script, against a real file."""

    def test_apply_exits_zero(self, tmp_path: Path) -> None:
        """A completed patch exits clean through the entry point."""
        target = tmp_path / "sensor.py"
        target.write_text(
            "\n".join(["prelude", _SIG_OLD, "middle", _MAX_OLD, "tail", _LAUNCH_OLD, "end"]) + "\n",
            encoding="utf-8",
        )
        assert _run("apply_tactile_alias_patch.py", ["apply", str(target)]) == 0
