"""The search driver against the in-memory queue: rounds really halve.

The rig completes each submitted round the moment the driver sleeps on
it -- every queued job flips to done and files a scripted card -- so the
test drives the whole schedule: submit, wait, score, halve, graduate.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from scripts.search import EXIT_BAD_USAGE, EXIT_OK, SCHEDULE, SPACE, main, run_search

from rw_bot.service import _test_hooks
from rw_bot.service._test_hooks import Connection
from tests.service_fakes import FakeConnection


class _Rig:
    """Serves one shared fake connection and plays rounds during sleeps."""

    def __init__(self, sweeps_root: Path) -> None:
        self.conn = FakeConnection()
        self.sweeps_root = sweeps_root
        self.slept: list[float] = []

    def connect(self, dsn: str) -> Connection:
        self.conn.closed = False
        return self.conn

    def sleep(self, seconds: float) -> None:
        """Complete every queued job and file its scripted card."""
        self.slept.append(seconds)
        for row in self.conn.store.jobs:
            if row.state != "queued":
                continue
            row.state = "done"
            if row.label == "decoys2":
                verdict, samples = "won", 500
            elif row.label == "control":
                verdict, samples = "survived", 1000
            else:
                verdict, samples = "wiped", 1000
            batch_dir = self.sweeps_root / str(row.batch)
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / f"{row.label}-s{row.seed}.txt").write_text(
                f"### {row.label}-s{row.seed}\n"
                f"verdict        {verdict} ({verdict})\n"
                f"samples seen   {samples}\n",
                encoding="utf-8",
            )


def test_the_search_halves_toward_the_scripted_winner(tmp_path: Path) -> None:
    """decoys2 wins fast in every round; everything else is wiped. The
    driver must rank it first for graduation, and each round must halve
    the field."""
    rig = _Rig(tmp_path / "sweeps")
    saved = (_test_hooks.connect, _test_hooks.sleep)
    _test_hooks.connect = rig.connect
    _test_hooks.sleep = rig.sleep
    try:
        lines = run_search(
            "dsn://demo",
            "probe",
            rng_seed=3,
            sweeps_root=tmp_path / "sweeps",
            variant_dir=tmp_path / "variants",
        )
    finally:
        (_test_hooks.connect, _test_hooks.sleep) = saved
    singles = sum(len(values) for values in SPACE.values())
    assert lines[0] == f"# search probe (rng 3): {singles + 6} candidates"
    # Round zero fields every candidate; round one fields half of them.
    assert f"# round 0: {singles + 6} arms, {SCHEDULE[0]} pairs" in lines[1]
    survivors_r1 = (singles + 6) // 2
    round1_header = next(line for line in lines if line.startswith("# round 1:"))
    assert f"{survivors_r1} arms, {SCHEDULE[1]} pairs" in round1_header
    # The scripted winner: won at half the longest match, control survived,
    # so the paired delta is (2 + 0.5) - 1 = +1.5 in both rounds.
    decoys_lines = [line for line in lines if " decoys2 " in line]
    assert len(decoys_lines) == 2
    assert all("margin delta +1.500" in line for line in decoys_lines)
    graduation = lines.index("# graduation order (full win-bar panel next, laws six and nine):")
    assert lines[graduation + 1] == "#   decoys2"
    # The variant files really exist and carry the move.
    written = (tmp_path / "variants" / "decoys2.doctrine").read_text(encoding="utf-8")
    assert "decoys 2" in written
    assert "name decoys2" in written


def test_main_rejects_bad_usage() -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert main(["a", "b", "c", "d"]) == EXIT_BAD_USAGE


def test_main_runs_a_search_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rig = _Rig(tmp_path / "sweeps")
    saved = (_test_hooks.connect, _test_hooks.sleep)
    _test_hooks.connect = rig.connect
    _test_hooks.sleep = rig.sleep
    try:
        code = main(
            ["dsn://demo", "probe", "3"],
            sweeps_root=tmp_path / "sweeps",
            variant_dir=tmp_path / "variants",
        )
    finally:
        (_test_hooks.connect, _test_hooks.sleep) = saved
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("# search probe (rng 3):")
    assert "#   decoys2\n" in out


def test_the_module_guard_runs_main() -> None:
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("scripts.search", run_name="__main__")
    assert caught.value.code == EXIT_BAD_USAGE
