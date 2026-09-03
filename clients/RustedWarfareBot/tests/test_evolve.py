"""The population loop: sample, play, select, refit -- deterministic.

A scripted runner files the generation's scorecards the way the
cluster-round and panel tests do, with one member per generation scripted
to win fast; the loop must rank it first, refit toward its genome, and
name the overall best at the end. Determinism is the load-bearing
property -- a relaunch must replay identical genomes -- so one test runs
the whole evolution twice and compares every job line.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from scripts.evolve import (
    ELITE,
    EXIT_BAD_USAGE,
    GENERATIONS,
    PAIRS,
    POPULATION,
    SIGMA_FLOOR,
    generation_seeds,
    main,
    member_label,
    refit,
    run_evolution,
    softmax,
)

from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.policy.doctrine_file import parse_doctrine_lines


class _Cluster:
    """Files scripted scorecards: member 0 wins fast, the rest wipe."""

    def __init__(self, sweeps_root: Path) -> None:
        self.sweeps_root = sweeps_root
        self.job_lines: list[str] = []

    def run(self, batch: str, job_lines: Sequence[str]) -> None:
        self.job_lines.extend(job_lines)
        batch_dir = self.sweeps_root / batch
        batch_dir.mkdir(parents=True, exist_ok=True)
        for line in job_lines:
            if line.startswith("#") or line.strip() == "":
                continue
            label, seed_text, _doctrine, _samples = line.split("|")
            if label == "control":
                verdict, samples = "survived", 1000
            elif label.endswith("m0"):
                verdict, samples = "won", 500
            else:
                verdict, samples = "wiped", 1000
            (batch_dir / f"{label}-s{seed_text}.txt").write_text(
                f"### {label}-s{seed_text}\n"
                f"verdict        {verdict} ({verdict})\n"
                f"samples seen   {samples}\n",
                encoding="utf-8",
            )


def _run(tmp_path: Path, tag: str) -> tuple[tuple[str, ...], list[str]]:
    cluster = _Cluster(tmp_path / f"sweeps-{tag}")
    written: list[str] = []

    def record(text: str) -> None:
        written.append(text)

    saved = host_hooks.write_line
    host_hooks.write_line = record
    try:
        lines = run_evolution(
            cluster,
            "probe",
            rng_seed=5,
            sweeps_root=tmp_path / f"sweeps-{tag}",
            variant_dir=tmp_path / f"variants-{tag}",
        )
    finally:
        host_hooks.write_line = saved
    return lines, cluster.job_lines


def test_the_evolution_runs_ranks_and_names_the_best(tmp_path: Path) -> None:
    lines, job_lines = _run(tmp_path, "a")
    header = lines[0]
    assert header.startswith(
        f"# evolve probe (rng 5): population {POPULATION}, elite {ELITE}, {GENERATIONS} generations"
    )
    generations = [line for line in lines if line.startswith("# generation ") and "members" in line]
    assert len(generations) == GENERATIONS
    # The scripted winner won at half the longest match while control
    # survived: paired delta (2 + 0.5) - 1 = +1.5, best of every
    # generation, so the overall best is a generation-0 m0.
    assert lines[-1].startswith("# best member: g0m0 (margin delta +1.500)")
    # Every generation fielded the full population plus control per seed.
    assert len(job_lines) == GENERATIONS * PAIRS * (POPULATION + 1)


def test_the_evolution_is_deterministic_end_to_end(tmp_path: Path) -> None:
    """A relaunch replays identical genomes -- the property the cluster's
    converge dedupe turns into free fast-forward after a harness sweep.
    Compared on what determinism claims: labels, seeds, and the compiled
    doctrine bytes -- the variant DIRECTORY differs per run by design."""

    def shape(job_lines: list[str]) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for line in job_lines:
            label, seed, _doctrine, samples = line.split("|")
            rows.append((label, seed, samples))
        return rows

    _, first = _run(tmp_path, "one")
    _, second = _run(tmp_path, "two")
    assert shape(first) == shape(second)
    sample = "g2m7.doctrine"
    assert (tmp_path / "variants-one" / sample).read_text(encoding="utf-8") == (
        tmp_path / "variants-two" / sample
    ).read_text(encoding="utf-8")


def test_compiled_candidates_are_valid_doctrines_on_disk(tmp_path: Path) -> None:
    _run(tmp_path, "files")
    variant_dir = tmp_path / "variants-files"
    sample = variant_dir / "g0m0.doctrine"
    doctrine = parse_doctrine_lines(sample.read_text(encoding="utf-8").splitlines())
    assert doctrine["name"] == "g0m0"
    # The scaffold survives and the army tail fills the champion's slots.
    assert doctrine["goals"][:3] == ("extractorT1", "extractorT1", "extractorT1")
    assert len(doctrine["goals"]) == 8


def test_generation_seeds_live_in_their_own_namespace() -> None:
    seeds = generation_seeds(5, 0)
    assert len(seeds) == PAIRS
    assert all(seed % 2 == 1 for seed in seeds)
    assert min(seeds) >= 500_000
    assert set(generation_seeds(5, 0)) & set(generation_seeds(5, 1)) == set()


def test_softmax_lands_on_the_simplex() -> None:
    weights = softmax((0.0, 1.0, -1.0, 0.5))
    assert abs(sum(weights) - 1.0) < 1e-12
    assert all(weight > 0.0 for weight in weights)
    assert weights[1] == max(weights)


def test_refit_floors_the_deviation() -> None:
    mean, sigma = refit([(1.0, 2.0), (1.0, 2.0)])
    assert mean == (1.0, 2.0)
    assert sigma == (SIGMA_FLOOR, SIGMA_FLOOR)


def test_member_labels_are_stable() -> None:
    assert member_label(3, 11) == "g3m11"


def test_main_rejects_bad_usage_and_non_cluster_routes() -> None:
    assert main([]) == EXIT_BAD_USAGE
    assert main(["dsn://demo", "probe"]) == EXIT_BAD_USAGE


def test_main_builds_the_cluster_runner_for_the_cluster_route(tmp_path: Path) -> None:
    from rw_bot.harness.cluster_round import ClusterRoundError

    seen: list[tuple[str, ...]] = []

    def refuse(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        seen.append(tuple(argv))
        return 9, ("stopped by the test before anything real",)

    saved = host_hooks.run_capture
    host_hooks.run_capture = refuse
    try:
        with pytest.raises(ClusterRoundError):
            main(
                ["hpc3:runs/hpc3-rusted.json", "probe", "5"],
                sweeps_root=tmp_path / "sweeps",
                variant_dir=tmp_path / "variants",
            )
    finally:
        host_hooks.run_capture = saved
    assert seen[0] == ("git", "rev-parse", "HEAD")


def test_the_module_guard_runs_main() -> None:
    import runpy

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("scripts.evolve", run_name="__main__")
    assert caught.value.code == EXIT_BAD_USAGE
