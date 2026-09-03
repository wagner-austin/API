"""The bespoke-batch command: one sweep file, the whole canonical chain.

The scripted-conversation approach of the cluster-round tests, driven
through the CLI: the fake serves the freeze-stage-converge-pull exchange
and the test asserts the command entered the canonical chain with the
file's own lines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from scripts.batch import EXIT_BAD_USAGE, EXIT_OK, main

from rw_bot.harness import _test_hooks as host_hooks


def test_main_rejects_bad_usage_paths() -> None:
    assert main([]) == EXIT_BAD_USAGE
    # A queue DSN is not a batch destination.
    assert main(["dsn://demo", "b", "sweeps/x.txt", "2"]) == EXIT_BAD_USAGE


def test_a_missing_sweep_file_is_refused_before_any_command(tmp_path: Path) -> None:
    assert (
        main(["hpc3:runs/hpc3-rusted.json", "ghost", str(tmp_path / "ghost.txt"), "2"])
        == EXIT_BAD_USAGE
    )


def test_a_batch_name_must_match_the_files_stem(tmp_path: Path) -> None:
    sweep = tmp_path / "real.txt"
    sweep.write_text("a|1|d.doctrine|100\n", encoding="utf-8")
    assert main(["hpc3:runs/hpc3-rusted.json", "other", str(sweep), "2"]) == EXIT_BAD_USAGE


def test_the_cluster_route_plays_the_files_own_lines(tmp_path: Path) -> None:
    sweep = tmp_path / "probe.txt"
    sweep.write_text(
        "# a bespoke panel\nalpha|11|doctrines/a.doctrine|100\nbeta|11|doctrines/b.doctrine|100\n",
        encoding="utf-8",
    )
    (tmp_path / "runs").mkdir()

    argvs: list[tuple[str, ...]] = []

    def serve(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        argvs.append(tuple(argv))
        joined = " ".join(argv)
        if joined.endswith("rev-parse HEAD"):
            return 0, ("abc123",)
        if "hpc3.cli.campaign" in joined:
            return 0, ("0 done, 0 in flight, 2 submitted, 2 remaining",)
        if 'grep -c "\\.txt$"' in joined:
            return 0, ("2",)
        return 0, ()

    saved = host_hooks.run_capture
    host_hooks.run_capture = serve
    try:
        code = main(
            ["hpc3:runs/hpc3-rusted.json", "probe", str(sweep), "3"],
            sweeps_root=tmp_path / "runs",
        )
    finally:
        host_hooks.run_capture = saved
    assert code == EXIT_OK
    joined = [" ".join(argv) for argv in argvs]
    # The canonical chain ran, at the difficulty asked for.
    assert any("scripts.stage_payload" in line for line in joined)
    doc = next(line for line in joined if "scripts.campaign_doc" in line)
    assert "--difficulty 3" in doc
    assert "--payload payload-probe" in doc
    # The job file the members read is the file's own content, rewritten
    # in place (stem == batch, so no clone appears).
    assert "beta|11|doctrines/b.doctrine|100" in sweep.read_text(encoding="utf-8")
    assert sorted(p.name for p in tmp_path.glob("*.txt")) == ["probe.txt"]


def test_the_module_guard_runs_main() -> None:
    import runpy

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("scripts.batch", run_name="__main__")
    assert caught.value.code == EXIT_BAD_USAGE
