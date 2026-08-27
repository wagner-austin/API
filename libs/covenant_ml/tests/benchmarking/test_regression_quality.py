"""Tests for the registry-driven regression benchmark.

Runs the real measurement path with all three real learners on a small
grouped corpus written into a tmp external directory under the
``rw_value`` registry entry's folder, so the module is exercised end to
end — including the grouped-split law.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.comparability import cpu_run_fingerprint, decode_run_fingerprint
from platform_core.determinism_cpu import apply_cpu_determinism
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_bool,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_list,
    narrow_json_to_str,
)
from platform_core.testing import SAMPLE_HOST, SAMPLE_PACKAGES

from covenant_ml.benchmarking.regression_quality import (
    RegressionBenchConfig,
    encode_regression_manifest,
    run_regression_benchmark,
    split_rows,
)


def write_rw_value_fixture(
    external_dir: Path, n_matches: int = 12, rows_per_match: int = 30
) -> None:
    """Write a small rw_value-shaped corpus under ``external_dir``.

    A deterministic decaying-signal corpus: each match counts down from a
    per-match end frame, and the state columns correlate with the
    remaining time, so a real model can learn something.

    Args:
        external_dir: Root the registry loader will read from.
        n_matches: Number of match groups.
        rows_per_match: Samples per match.
    """
    folder = external_dir / "rw_value"
    folder.mkdir(parents=True, exist_ok=True)
    header = (
        "match,frame,army,credits,enemies,extractors,lost,lost_cum,killed_cum,"
        "producers,idle,orders,refused,worth,rival,income,rival_income,frames_remaining"
    )
    lines = [header]
    for match_idx in range(n_matches):
        end_frame = 300 + 40 * match_idx
        for row_idx in range(rows_per_match):
            frame = (end_frame * row_idx) // rows_per_match
            remaining = end_frame - frame
            army = 5 + (match_idx * 3 + row_idx) % 11
            worth = 3500 + 10 * frame + 17 * match_idx
            rival = 3500 + 9 * remaining
            lines.append(
                f"m{match_idx},{frame},{army},{4000 - frame % 900},2,{row_idx % 4},"
                f"{row_idx % 3},{row_idx},{row_idx // 2},{1 + row_idx % 5},0,"
                f"{row_idx % 7},0,{worth},{rival},{18 + match_idx},{20 + row_idx % 6},"
                f"{remaining}"
            )
    (folder / "data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


NOT_LOADED: tuple[str, ...] = ()
"""A module table with no native numeric library in it.

This test module has numpy loaded -- it is a numpy test suite -- so the pin
would rightly refuse against the real `sys.modules`. Stating the precondition
keeps the fixture describing a production run rather than this process.
"""


def _discard_env(name: str, value: str) -> None:
    """Accept a pinned variable without writing it.

    Args:
        name: The variable the pin would set.
        value: The value it would be set to.
    """


_FINGERPRINT = cpu_run_fingerprint(
    apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
    {"IMAGE_DIGEST": "sha256:" + "ef" * 32}.get,
    SAMPLE_HOST,
    SAMPLE_PACKAGES,
)
"""The configuration these benchmarks claim to run under.

Built from the real `apply_cpu_determinism` with a no-op writer, so the
record says exactly what a production run's would say without this test
process reaching into the environment of every other test sharing the
worker. The image digest is supplied rather than read for the same reason.
"""


def _small_config() -> RegressionBenchConfig:
    """Return hyperparameters small enough for a fast test run."""
    return RegressionBenchConfig(
        dataset="rw_value",
        n_estimators=25,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.2,
        max_bins=16,
        min_samples_leaf=5,
        early_stopping_rounds=10,
    )


def _group_codes(groups: NDArray[np.int64], indices: NDArray[np.intp]) -> set[int]:
    """Distinct group codes covered by an index array, as plain ints."""
    out: set[int] = set()
    for i in range(len(indices)):
        idx: np.intp = indices[i]
        value: np.int64 = groups[idx]
        out.add(int(value))
    return out


class TestSplitRows:
    """The split honours grouping when groups exist."""

    def test_grouped_split_never_straddles_a_group(self) -> None:
        """Every group's rows land in exactly one partition."""
        groups = np.repeat(np.arange(10, dtype=np.int64), 20)
        indices = split_rows(len(groups), groups, seed=42)
        train_groups = _group_codes(groups, indices["train"])
        val_groups = _group_codes(groups, indices["val"])
        test_groups = _group_codes(groups, indices["test"])
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)
        assert len(train_groups | val_groups | test_groups) == 10

    def test_grouped_split_partitions_by_group_ratio(self) -> None:
        """The 0.6/0.2/0.2 ratios apply to GROUPS, not rows."""
        groups = np.repeat(np.arange(10, dtype=np.int64), 7)
        indices = split_rows(len(groups), groups, seed=7)
        assert len(_group_codes(groups, indices["train"])) == 6
        assert len(_group_codes(groups, indices["val"])) == 2
        assert len(_group_codes(groups, indices["test"])) == 2

    def test_row_split_covers_every_row_once(self) -> None:
        """Without groups, the three partitions tile the row range."""
        indices = split_rows(100, None, seed=42)
        combined_rows: list[int] = []
        for part in (indices["train"], indices["val"], indices["test"]):
            for i in range(len(part)):
                idx: np.intp = part[i]
                combined_rows.append(int(idx))
        assert sorted(combined_rows) == list(range(100))
        assert len(indices["train"]) == 60
        assert len(indices["val"]) == 20

    def test_split_is_deterministic_per_seed(self) -> None:
        """Same seed reproduces the same partition."""
        groups = np.repeat(np.arange(8, dtype=np.int64), 5)
        first = split_rows(len(groups), groups, seed=3)
        second = split_rows(len(groups), groups, seed=3)
        assert np.array_equal(first["train"], second["train"])
        assert np.array_equal(first["test"], second["test"])


class TestRunRegressionBenchmark:
    """All four arms measure, learn, time, and encode to a manifest."""

    def test_all_four_arms_report_on_a_grouped_corpus(self, tmp_path: Path) -> None:
        """One record per arm per seed, all finite, with fit times."""
        write_rw_value_fixture(tmp_path)
        manifest = run_regression_benchmark(_small_config(), [42], tmp_path, _FINGERPRINT)
        assert manifest["grouped"] is True
        arms = [r["model"] for r in manifest["results"]]
        assert arms == ["cleargbm", "cleargbm@leaf_wise", "lightgbm", "xgboost"]
        for result in manifest["results"]:
            assert math.isfinite(result["quality"]["rmse"])
            assert math.isfinite(result["quality"]["mae"])
            assert math.isfinite(result["quality"]["r_squared"])
            assert result["fit_seconds"] > 0.0

    def test_the_signal_is_learnable(self, tmp_path: Path) -> None:
        """The decaying-signal fixture rewards real learning: R2 > 0."""
        write_rw_value_fixture(tmp_path)
        manifest = run_regression_benchmark(_small_config(), [42], tmp_path, _FINGERPRINT)
        by_arm = {r["model"]: r["quality"] for r in manifest["results"]}
        assert by_arm["cleargbm"]["r_squared"] > 0.0

    def test_the_manifest_says_what_configuration_produced_the_numbers(
        self, tmp_path: Path
    ) -> None:
        """Until 2026-08-25 it said nothing, while the p6 sweeps ran on HPC3.

        An RMSE measured on a 24-core cluster node and one measured on an
        8-core workstation were indistinguishable in the file, and could be
        subtracted with nothing anywhere to say that was unsound. The BLAS
        thread count alone changed 865,498 of 16,777,216 matmul elements
        between 1, 8 and 24 threads.
        """
        write_rw_value_fixture(tmp_path)
        manifest = run_regression_benchmark(_small_config(), [42], tmp_path, _FINGERPRINT)

        assert manifest["fingerprint"] == _FINGERPRINT

    def test_the_configuration_survives_to_the_file_on_disk(self, tmp_path: Path) -> None:
        """A fingerprint the encoder drops is a fingerprint nobody ever reads."""
        write_rw_value_fixture(tmp_path)
        manifest = run_regression_benchmark(_small_config(), [42], tmp_path, _FINGERPRINT)

        encoded = narrow_json_to_dict(
            load_json_str(dump_json_str(encode_regression_manifest(manifest)))
        )

        assert decode_run_fingerprint(encoded["fingerprint"]) == _FINGERPRINT

    def test_the_thread_count_is_in_the_recorded_posture(self, tmp_path: Path) -> None:
        """The axis that decides these numbers, named in the record itself.

        These arms use no GPU, so the card is not the interesting axis here.
        The reduction order is, and it is set by a thread count that nothing
        recorded before.
        """
        write_rw_value_fixture(tmp_path)
        manifest = run_regression_benchmark(_small_config(), [42], tmp_path, _FINGERPRINT)

        settings = dict(manifest["fingerprint"]["determinism"]["settings"])
        assert settings["OMP_NUM_THREADS"] == SINGLE_THREAD
        assert manifest["fingerprint"]["gpu_model"] == ""

    def test_manifest_encodes_to_json(self, tmp_path: Path) -> None:
        """The encoded manifest round-trips through the JSON codec."""
        write_rw_value_fixture(tmp_path)
        manifest = run_regression_benchmark(_small_config(), [42], tmp_path, _FINGERPRINT)
        encoded = encode_regression_manifest(manifest)
        decoded = narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
        assert narrow_json_to_bool(decoded["grouped"]) is True
        config = narrow_json_to_dict(decoded["config"])
        assert narrow_json_to_str(config["dataset"]) == "rw_value"
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 4
        for entry in results:
            record = narrow_json_to_dict(entry)
            assert narrow_json_to_float(record["fit_seconds"]) > 0.0
