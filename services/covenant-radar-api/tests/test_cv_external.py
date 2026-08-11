"""Grouped k-fold CV runner: whole groups per fold, spread reported.

The single-split AUC swung 0.79 -> 0.63 between two exports for no modelling
reason; the runner exists to report the spread instead of a lucky draw. These
drive it with a fake loader and backend through the worker hooks, the house
save-and-restore DI.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from covenant_ml.datasets.registry import DatasetRegistry
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
    TargetColumnSpec,
)
from covenant_ml.types import (
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
)
from scripts.cv_external import EXIT_BAD_USAGE, EXIT_OK, main

from covenant_radar_api.worker import _test_hooks as hooks

if TYPE_CHECKING:
    from covenant_ml.backends.protocol import ProgressCallback
    from numpy.typing import NDArray


def _dataset_config(name: str, grouped: bool) -> DatasetConfig:
    config = DatasetConfig(
        name=name,
        display_name=f"Test {name}",
        folder=name,
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="won",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=120,
        n_features_expected=3,
        positive_class_ratio_expected=0.5,
    )
    if grouped:
        config["group_column"] = "match"
    return config


def _fake_dataset(grouped: bool) -> LoadedDataset:
    """Twelve groups of ten rows; the first feature perfectly ranks the label."""
    n_groups, rows_per = 12, 10
    n = n_groups * rows_per
    y = np.zeros(n, dtype=np.int64)
    groups = np.zeros(n, dtype=np.int64)
    x = np.zeros((n, 3), dtype=np.float64)
    for g in range(n_groups):
        start = g * rows_per
        groups[start : start + rows_per] = g
        label = 1 if g % 2 == 0 else 0
        y[start : start + rows_per] = label
        x[start : start + rows_per, 0] = float(label) + 0.1 * g / n_groups
    meta = DatasetMeta(
        name="grouped_fake",
        n_samples=n,
        n_features=3,
        n_positive=int(np.sum(y)),
        n_negative=n - int(np.sum(y)),
        positive_ratio=float(np.sum(y)) / n,
        feature_names=("a", "b", "c"),
        categorical_encodings=(),
    )
    return LoadedDataset(meta=meta, x=x, y=y, groups=groups if grouped else None)


class _FakePrepared:
    """Scores rows by the first feature, which perfectly ranks the label."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        positive: NDArray[np.float64] = np.clip(x[:, 0], 0.0, 1.0)
        negative: NDArray[np.float64] = 1.0 - positive
        columns: list[NDArray[np.float64]] = [negative, positive]
        stacked: NDArray[np.float64] = np.column_stack(columns)
        return stacked


class _FakeCVBackend:
    """Records the grouped inner split and files a model like the real ones."""

    def __init__(self) -> None:
        self.inner_groups_seen: list[bool] = []

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
        groups: NDArray[np.int64] | None = None,
    ) -> TrainOutcome:
        self.inner_groups_seen.append(groups is not None)
        model_path = output_dir / "model.json"
        model_path.write_text("fake", encoding="utf-8")
        metrics: EvalMetrics = {
            "loss": 0.3,
            "ppl": 1.3,
            "auc": 0.9,
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1_score": 0.9,
        }
        importances: list[FeatureImportance] = [{"name": "a", "importance": 1.0, "rank": 1}]
        n = int(x_features.shape[0])
        return {
            "model_path": str(model_path),
            "model_id": "fake",
            "samples_total": n,
            "samples_train": n,
            "samples_val": 0,
            "samples_test": 0,
            "train_metrics": metrics,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "best_val_auc": 0.9,
            "best_round": 1,
            "total_rounds": 1,
            "early_stopped": False,
            "config": config,
            "feature_importances": importances,
            "scale_pos_weight_computed": 1.0,
        }

    def load(self, *, path: str) -> _FakePrepared:
        assert Path(path).exists()
        return _FakePrepared()


class _FakeClassifierRegistry:
    def __init__(self, backend: _FakeCVBackend) -> None:
        self._backend = backend

    def get(self, name: str) -> _FakeCVBackend:
        return self._backend


@pytest.fixture()
def cv_hooks() -> Iterator[_FakeCVBackend]:
    """Install fake registry, loader and backend; restore on exit."""
    backend = _FakeCVBackend()
    registry = DatasetRegistry(
        (_dataset_config("grouped_fake", True), _dataset_config("flat_fake", False))
    )

    def fake_registry() -> DatasetRegistry:
        return registry

    def fake_loader(
        config: DatasetConfig,
        external_dir: Path,
        progress_callback: object = None,
    ) -> LoadedDataset:
        return _fake_dataset(grouped=config.get("group_column") is not None)

    saved = (hooks.dataset_registry_factory, hooks.dataset_loader, hooks.registry_factory)
    hooks.dataset_registry_factory = fake_registry
    hooks.dataset_loader = fake_loader
    hooks.registry_factory = lambda: _FakeClassifierRegistry(backend)  # type: ignore[assignment]
    yield backend
    (hooks.dataset_registry_factory, hooks.dataset_loader, hooks.registry_factory) = saved


def test_every_fold_scores_its_held_out_groups(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Three folds, perfect ranking feature: three fold lines at AUC 1, and
    the inner early-stopping split is grouped on every fold."""
    assert main(["grouped_fake", "cleargbm", "3", "7"], external_dir=tmp_path) == EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "grouped_fake via cleargbm: 120 rows, 12 groups, 3 folds, seed 7"
    fold_lines = [line for line in out if line.startswith("fold ")]
    assert len(fold_lines) == 3
    assert all("auc 1.0000" in line for line in fold_lines)
    assert all("4 held-out groups, 40 rows" in line for line in fold_lines)
    assert out[-1] == "mean auc 1.0000 +/- 0.0000 over 3 folds"
    assert cv_hooks.inner_groups_seen == [True, True, True]


def test_defaults_are_five_folds_and_the_fixed_seed(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["grouped_fake", "lightgbm"], external_dir=tmp_path) == EXIT_OK
    out = capsys.readouterr().out
    assert "5 folds, seed 42" in out
    assert "over 5 folds" in out


def test_an_ungrouped_dataset_is_refused_with_the_reason(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Row-level CV of correlated rows would score memorization as skill;
    saying so beats a quietly wrong number."""
    assert main(["flat_fake", "cleargbm"], external_dir=tmp_path) == EXIT_BAD_USAGE
    assert "has no group column" in capsys.readouterr().out


def test_unknown_dataset_and_backend_are_usage_errors(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["missing", "cleargbm"], external_dir=tmp_path) == EXIT_BAD_USAGE
    assert "dataset must be one of: flat_fake, grouped_fake" in capsys.readouterr().out
    assert main(["grouped_fake", "catboost"], external_dir=tmp_path) == EXIT_BAD_USAGE
    assert "backend must be cleargbm or lightgbm" in capsys.readouterr().out


def test_a_bad_argument_count_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["only-one"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: cv_external <dataset> <backend> [folds] [seed]\n"


def test_module_entry_point_exits_with_the_run_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.cv_external")
    sys.argv = ["cv_external"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.cv_external", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.cv_external"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
