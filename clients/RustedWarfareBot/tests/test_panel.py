"""The unattended panel driver: layout, transport, judgement, in order.

A scripted runner stands in for the cluster exactly as the cluster-round
tests script the ssh conversation: it receives the job lines the driver
laid out and files the scorecards a real drain would, so the test walks
the whole flow -- seed allocation off the real sweep files, interleaved
job lines, judgement off the filed cards -- without a mock anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from scripts.panel import (
    EXIT_BAD_USAGE,
    FIRST_HIGH_SEED,
    FIRST_SEED,
    SEARCH_SEED_FLOOR,
    SEED_BLOCK,
    PanelError,
    main,
    panel_job_lines,
    run_panel,
    seed_block,
    used_seeds,
)

from rw_bot.harness import _test_hooks as host_hooks


def _write_sweep(path: Path, seeds: Sequence[int]) -> None:
    lines = ["# a prior experiment"]
    for seed in seeds:
        lines.append(f"old|{seed}|doctrines/flame-nocover.doctrine|4000")
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


class _Cluster:
    """Plays a panel by filing scripted scorecards for every job line."""

    def __init__(self, sweeps_root: Path, arm_verdicts: dict[int, str]) -> None:
        self.sweeps_root = sweeps_root
        self.arm_verdicts = arm_verdicts
        self.batches: list[str] = []
        self.job_lines: list[str] = []

    def run(self, batch: str, job_lines: Sequence[str]) -> None:
        self.batches.append(batch)
        self.job_lines.extend(job_lines)
        batch_dir = self.sweeps_root / batch
        batch_dir.mkdir(parents=True, exist_ok=True)
        for line in job_lines:
            if line.startswith("#") or line.strip() == "":
                continue
            label, seed_text, _doctrine, _samples = line.split("|")
            seed = int(seed_text)
            if label == "control":
                verdict, samples = "survived", 1000
            else:
                verdict = self.arm_verdicts[seed]
                samples = 500 if verdict == "won" else 1000
            (batch_dir / f"{label}-s{seed}.txt").write_text(
                f"### {label}-s{seed}\n"
                f"verdict        {verdict} ({verdict})\n"
                f"samples seen   {samples}\n",
                encoding="utf-8",
            )


def test_a_panel_lays_out_plays_and_judges_in_one_run(tmp_path: Path) -> None:
    """Fresh seeds from the next block, interleaved lines, and the same
    pairs + margin judgement the manual flow printed."""
    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    _write_sweep(jobs_dir / "old.txt", [10001, 10003])
    sweeps_root = tmp_path / "runs"

    # The block above the used seeds starts at 11001 and holds 2,500 odd
    # values; a 2-pair draw strides the pool, picking 11001 and 13501.
    arm_verdicts = {11001: "won", 13501: "wiped"}
    cluster = _Cluster(sweeps_root, arm_verdicts)

    written: list[str] = []

    def record(text: str) -> None:
        written.append(text)

    saved = host_hooks.write_line
    host_hooks.write_line = record
    try:
        lines = run_panel(
            cluster,
            "probe",
            "doctrines/flame-close6.doctrine",
            "cand",
            "doctrines/close0-flame4.doctrine",
            2,
            sweeps_root=sweeps_root,
            jobs_dir=jobs_dir,
        )
    finally:
        host_hooks.write_line = saved

    assert cluster.batches == ["probe"]
    jobs = [line for line in cluster.job_lines if not line.startswith("#")]
    assert jobs == [
        "control|11001|doctrines/flame-close6.doctrine|10000",
        "cand|11001|doctrines/close0-flame4.doctrine|10000",
        "control|13501|doctrines/flame-close6.doctrine|10000",
        "cand|13501|doctrines/close0-flame4.doctrine|10000",
    ]
    assert lines[0] == "paired 2 seed(s): probe:control (left) vs probe:cand (right)"
    assert lines[1] == "wins   0 -> 1"
    assert lines[2] == "flips  1 to-W against 0 from-W (net +1 for probe:cand)"
    # The margin report follows, reading the same directory.
    assert lines[-1].startswith("paired control - cand:")
    # Everything returned was also streamed the moment it happened.
    assert list(lines) == written[1:]
    assert written[0].startswith("# panel probe: 2 pairs, seeds 11001-13501")


def test_the_seed_block_advances_past_panels_and_ignores_the_search_floor() -> None:
    assert seed_block(set()) == (FIRST_SEED, FIRST_SEED + SEED_BLOCK)
    assert seed_block({12345}) == (13001, 18001)
    # Seeds at or above the floor belong to searches or historical grads
    # and must not drag the panel namespace up to them.
    assert seed_block({12345, 214001}) == (13001, 18001)


def test_a_full_low_region_continues_in_the_high_region() -> None:
    """The measured 2026-09-04 shape: the block after impden48's would
    have crossed the search floor and RW-PANEL-001 refused the layout.
    Allocation now continues above PANEL_HIGH_FLOOR -- over every search
    and evolution seed by construction -- instead of refusing."""
    assert seed_block({SEARCH_SEED_FLOOR - 500}) == (
        FIRST_HIGH_SEED,
        FIRST_HIGH_SEED + SEED_BLOCK,
    )
    # Search and evolution seeds between the regions never drag the high
    # block up; only prior HIGH-region panels advance it.
    assert seed_block({SEARCH_SEED_FLOOR - 500, 214001, 500_011}) == (
        FIRST_HIGH_SEED,
        FIRST_HIGH_SEED + SEED_BLOCK,
    )
    assert seed_block({SEARCH_SEED_FLOOR - 500, 1_004_321}) == (1_005_001, 1_010_001)


def test_a_panel_of_zero_pairs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PanelError) as caught:
        run_panel(
            _Cluster(tmp_path / "runs", {}),
            "probe",
            "doctrines/a.doctrine",
            "cand",
            "doctrines/b.doctrine",
            0,
            sweeps_root=tmp_path / "runs",
            jobs_dir=tmp_path / "sweeps",
        )
    assert caught.value.code == "RW-PANEL-002"


def test_a_relaunch_reuses_the_batchs_own_seeds(tmp_path: Path) -> None:
    """The kill-resume property: a swept driver relaunched with the same
    request must replay the SAME seeds so the cluster's converge dedupes,
    never abandon in-flight matches for a fresh block."""
    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    sweeps_root = tmp_path / "runs"
    arm_verdicts = {10001: "won", 12501: "wiped"}

    first = _Cluster(sweeps_root, arm_verdicts)
    run_panel(
        first,
        "probe",
        "doctrines/a.doctrine",
        "cand",
        "doctrines/b.doctrine",
        2,
        sweeps_root=sweeps_root,
        jobs_dir=jobs_dir,
    )
    # The runner wrote nothing to jobs_dir (a scripted stand-in), so
    # simulate what the real ClusterRound leaves behind: the job file.
    (jobs_dir / "probe.txt").write_text(
        "".join(f"{line}\n" for line in first.job_lines), encoding="utf-8"
    )

    second = _Cluster(sweeps_root, arm_verdicts)
    run_panel(
        second,
        "probe",
        "doctrines/a.doctrine",
        "cand",
        "doctrines/b.doctrine",
        2,
        sweeps_root=sweeps_root,
        jobs_dir=jobs_dir,
    )
    jobs_first = [line for line in first.job_lines if not line.startswith("#")]
    jobs_second = [line for line in second.job_lines if not line.startswith("#")]
    assert jobs_second == jobs_first


def test_a_relaunch_with_a_different_request_is_refused(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    (jobs_dir / "probe.txt").write_text(
        "control|10001|doctrines/a.doctrine|10000\ncand|10001|doctrines/b.doctrine|10000\n",
        encoding="utf-8",
    )
    with pytest.raises(PanelError) as caught:
        run_panel(
            _Cluster(tmp_path / "runs", {}),
            "probe",
            "doctrines/a.doctrine",
            "cand",
            "doctrines/b.doctrine",
            48,
            sweeps_root=tmp_path / "runs",
            jobs_dir=jobs_dir,
        )
    assert caught.value.code == "RW-PANEL-003"
    assert "1 'cand' seed(s) but 48 pairs" in caught.value.message


def test_used_seeds_count_generated_search_rounds_too(tmp_path: Path) -> None:
    """sweeps/search is gitignored but its seeds are just as consumed."""
    jobs_dir = tmp_path / "sweeps"
    (jobs_dir / "search").mkdir(parents=True)
    _write_sweep(jobs_dir / "panel.txt", [10001])
    _write_sweep(jobs_dir / "search" / "round.txt", [200001])
    assert used_seeds(jobs_dir) == {10001, 200001}


def test_used_seeds_read_the_pre_doctrine_era_too(tmp_path: Path) -> None:
    """The committed historical sweeps carry seven-field job lines the
    strict parser refuses as jobs; their seeds are consumed all the same,
    and the first live panel launch died on exactly this file shape."""
    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    (jobs_dir / "aggression.txt").write_text(
        "# the 2026-07 era\nattack|12345|extractorT1,c_tank,c_tank|99|4000|25|-1\n",
        encoding="utf-8",
    )
    _write_sweep(jobs_dir / "modern.txt", [10001])
    assert used_seeds(jobs_dir) == {12345, 10001}


def test_a_seedless_job_line_is_refused_not_skipped(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    (jobs_dir / "broken.txt").write_text("label|not-a-seed|x|y\n", encoding="utf-8")
    with pytest.raises(PanelError) as caught:
        used_seeds(jobs_dir)
    assert caught.value.code == "RW-PANEL-004"
    assert "broken.txt" in caught.value.message


def test_job_lines_interleave_and_document_the_layout() -> None:
    lines = panel_job_lines(
        "probe", "doctrines/a.doctrine", "cand", "doctrines/b.doctrine", (11, 13)
    )
    assert lines[-4:] == (
        "control|11|doctrines/a.doctrine|10000",
        "cand|11|doctrines/b.doctrine|10000",
        "control|13|doctrines/a.doctrine|10000",
        "cand|13|doctrines/b.doctrine|10000",
    )
    header = "\n".join(line for line in lines if line.startswith("#"))
    assert "11-13" in header
    assert "cand (doctrines/b.doctrine)" in header


def test_main_rejects_bad_usage_and_non_cluster_routes(tmp_path: Path) -> None:
    assert main([]) == EXIT_BAD_USAGE
    # Seven arguments but a queue DSN: the panel is a cluster tool, and a
    # half-supported second transport would be a fallback.
    assert main(["dsn://demo", "b", "c.doctrine", "arm", "d.doctrine", "2", "2"]) == EXIT_BAD_USAGE


def test_main_builds_the_cluster_runner_for_the_cluster_route(tmp_path: Path) -> None:
    """The first real act must be the runner asking git for the commit --
    proof the canonical freeze chain was entered, not a parallel path."""
    from rw_bot.harness.cluster_round import ClusterRoundError

    seen: list[tuple[str, ...]] = []

    def refuse(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        seen.append(tuple(argv))
        return 9, ("stopped by the test before anything real",)

    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    saved = host_hooks.run_capture
    host_hooks.run_capture = refuse
    try:
        with pytest.raises(ClusterRoundError):
            main(
                [
                    "hpc3:runs/hpc3-rusted.json",
                    "probe",
                    "doctrines/a.doctrine",
                    "cand",
                    "doctrines/b.doctrine",
                    "2",
                    "2",
                ],
                sweeps_root=tmp_path / "runs",
                jobs_dir=jobs_dir,
            )
    finally:
        host_hooks.run_capture = saved
    assert seen[0] == ("git", "rev-parse", "HEAD")


def test_main_succeeds_when_the_scripted_cluster_files_everything(tmp_path: Path) -> None:
    """The whole command path: main builds the runner, the scripted
    conversation drains the batch, and the judgement reads the cards."""
    jobs_dir = tmp_path / "sweeps"
    jobs_dir.mkdir()
    sweeps_root = tmp_path / "runs"
    batch_dir = sweeps_root / "probe"
    batch_dir.mkdir(parents=True)
    # The cards the "pull" would have landed: block 10001's 2-pair draw
    # strides 2,500 odd values, picking 10001 and 12501.
    for label, seed, verdict in (
        ("control", 10001, "survived"),
        ("cand", 10001, "won"),
        ("control", 12501, "survived"),
        ("cand", 12501, "wiped"),
    ):
        (batch_dir / f"{label}-s{seed}.txt").write_text(
            f"### {label}-s{seed}\nverdict        {verdict} ({verdict})\nsamples seen   1000\n",
            encoding="utf-8",
        )

    def serve(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        joined = " ".join(argv)
        if joined.endswith("rev-parse HEAD"):
            return 0, ("abc123",)
        if "hpc3.cli.campaign" in joined:
            return 0, ("0 done, 0 in flight, 4 submitted, 4 remaining",)
        if 'grep -c "\\.txt$"' in joined:
            return 0, ("4",)
        return 0, ()

    saved = host_hooks.run_capture
    host_hooks.run_capture = serve
    try:
        code = main(
            [
                "hpc3:runs/hpc3-rusted.json",
                "probe",
                "doctrines/a.doctrine",
                "cand",
                "doctrines/b.doctrine",
                "2",
                "2",
            ],
            sweeps_root=sweeps_root,
            jobs_dir=jobs_dir,
        )
    finally:
        host_hooks.run_capture = saved
    assert code == 0
    # The panel's job file was really written where the freeze reads.
    written = (jobs_dir / "probe.txt").read_text(encoding="utf-8")
    assert "control|10001|doctrines/a.doctrine|10000" in written
    assert "cand|12501|doctrines/b.doctrine|10000" in written


def test_the_module_guard_runs_main() -> None:
    import runpy

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("scripts.panel", run_name="__main__")
    assert caught.value.code == EXIT_BAD_USAGE
