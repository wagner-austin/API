"""Grouped k-fold CV runner: whole groups per fold, spread reported.

The single-split AUC swung 0.79 -> 0.63 between two exports for no modelling
reason; the runner exists to report the spread instead of a lucky draw. These
drive it with a fake loader and backend through the worker hooks, the house
save-and-restore DI.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TypeGuard

import numpy as np
import pytest
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
)
from covenant_ml.backends.registry import BackendRegistration, ClassifierRegistry
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.registry import DatasetRegistry
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
    TargetColumnSpec,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    ClearGBMConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
    TrainProgress,
)
from numpy.typing import NDArray
from scripts.cv_external import (
    EXIT_BAD_USAGE,
    EXIT_OK,
    _has_mixed_label_groups,
    _split_for,
    main,
)

from covenant_radar_api.worker import _test_hooks


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


def _is_cleargbm_config(cfg: ClassifierTrainConfig) -> TypeGuard[ClearGBMConfig]:
    """Narrow to ClearGBMConfig by its growth_strategy key, unique to it."""
    return "growth_strategy" in cfg


class _FakeCVBackend:
    """Records the grouped inner split and files a model like the real ones."""

    def __init__(self) -> None:
        self.inner_groups_seen: list[bool] = []
        self.growth_strategies_seen: list[str] = []
        self.num_leaves_seen: list[int | None] = []
        self.min_data_in_bin_seen: list[int | None] = []

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: Callable[[TrainProgress], None] | None,
        groups: NDArray[np.int64] | None = None,
    ) -> TrainOutcome:
        self.inner_groups_seen.append(groups is not None)
        if _is_cleargbm_config(config):
            self.growth_strategies_seen.append(config["growth_strategy"])
            self.num_leaves_seen.append(config["num_leaves"])
            self.min_data_in_bin_seen.append(config.get("min_data_in_bin"))
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

    def backend_name(self) -> BackendName:
        return "cleargbm"

    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError("the CV runner does not query backend capabilities")

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        raise NotImplementedError("the CV runner trains from raw arrays, it does not prepare")

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        raise NotImplementedError("the CV runner scores folds itself from predict_proba")

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        raise NotImplementedError("train() already writes the model file")

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        raise NotImplementedError("the CV runner reports AUC, not importances")

    def get_default_search_space(self) -> SearchSpace:
        raise NotImplementedError("the CV runner does not tune hyperparameters")

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        raise NotImplementedError("the CV runner does not tune hyperparameters")


@pytest.fixture()
def cv_hooks() -> Generator[_FakeCVBackend, None, None]:
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
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        return _fake_dataset(grouped=config.get("group_column") is not None)

    def backend_factory() -> ClassifierBackend:
        return backend

    classifier_registry = ClassifierRegistry()
    classifier_registry.register("cleargbm", BackendRegistration(backend_factory))
    classifier_registry.register("lightgbm", BackendRegistration(backend_factory))

    def fake_classifier_registry() -> ClassifierRegistry:
        return classifier_registry

    saved = (
        _test_hooks.dataset_registry_factory,
        _test_hooks.dataset_loader,
        _test_hooks.registry_factory,
    )
    _test_hooks.dataset_registry_factory = fake_registry
    _test_hooks.dataset_loader = fake_loader
    _test_hooks.registry_factory = fake_classifier_registry
    yield backend
    (
        _test_hooks.dataset_registry_factory,
        _test_hooks.dataset_loader,
        _test_hooks.registry_factory,
    ) = saved


def test_every_fold_scores_its_held_out_groups(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Three folds, perfect ranking feature: three fold lines at AUC 1, and
    the inner early-stopping split is grouped on every fold."""
    assert main(["grouped_fake", "cleargbm", "3", "7"], external_dir=tmp_path) == EXIT_OK
    # The runner writes to stdout, and so does logging once another test in the
    # session has configured a stdout handler, so its lines are identified by
    # value rather than by position in the stream.
    out = capsys.readouterr().out.splitlines()
    assert "grouped_fake via cleargbm: 120 rows, 12 groups, 3 folds, seed 7" in out
    assert "mean auc 1.0000 +/- 0.0000 over 3 folds" in out
    fold_lines = [line for line in out if line.startswith("fold ")]
    assert len(fold_lines) == 3
    assert all("auc 1.0000" in line for line in fold_lines)
    assert all("4 held-out groups, 40 rows" in line for line in fold_lines)
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
    assert capsys.readouterr().out == (
        "flat_fake has no group column; grouped CV needs one -- a plain "
        "k-fold of correlated rows would score memorization as skill\n"
    )


def test_unknown_dataset_and_backend_are_usage_errors(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["missing", "cleargbm"], external_dir=tmp_path) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "dataset must be one of: flat_fake, grouped_fake (got missing)\n"
    )
    assert main(["grouped_fake", "catboost"], external_dir=tmp_path) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "backend must be cleargbm, cleargbm-leafwise or lightgbm (got catboost)\n"
    )


def test_leafwise_variant_runs_with_a_leaf_budget(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The cleargbm-leafwise protocol variant trains under the cleargbm
    backend name with leaf_wise growth and the fixed 31-leaf budget."""
    assert main(["grouped_fake", "cleargbm-leafwise", "3", "7"], external_dir=tmp_path) == EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert "grouped_fake via cleargbm-leafwise: 120 rows, 12 groups, 3 folds, seed 7" in out
    assert cv_hooks.growth_strategies_seen == ["leaf_wise", "leaf_wise", "leaf_wise"]
    assert cv_hooks.num_leaves_seen == [31, 31, 31]


def _int64_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create an int64 array from a tuple of ints without Any leakage.

    Args:
        values: The integer values.

    Returns:
        Array of int64 dtype.
    """
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def test_mixed_label_groups_use_plain_grouped_kfold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Groups holding both labels switch to the plain grouped instrument,
    and the switch is announced — stratification's any-positive group
    label is undefined for co-elution windows."""
    y = _int64_array((1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0))
    groups = _int64_array((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5))
    assert _has_mixed_label_groups(y, groups)

    splits = _split_for(y, groups, 3, 42)
    assert splits["n_folds"] == 3
    assert capsys.readouterr().out == (
        "groups carry mixed labels; label stratification is undefined -- "
        "using plain grouped k-fold\n"
    )


def test_uniform_label_groups_keep_the_stratified_protocol(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Label-uniform groups run the stratified splitter silently — the
    protocol every standing number ran under, unchanged."""
    y = _int64_array((1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0))
    groups = _int64_array((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5))
    assert not _has_mixed_label_groups(y, groups)

    splits = _split_for(y, groups, 3, 42)
    assert splits["n_folds"] == 3
    assert capsys.readouterr().out == ""


def test_a_bad_argument_count_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["only-one"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == (
        "usage: cv_external <dataset> <backend> [folds] [seed] [min_data_in_bin]\n"
    )


def test_min_data_in_bin_reaches_every_fold_config(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The optional floor threads into each fold's ClearGBM config and is
    announced in the header; without it the wire key stays absent."""
    assert main(["grouped_fake", "cleargbm", "3", "7", "8"], external_dir=tmp_path) == EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert (
        "grouped_fake via cleargbm: 120 rows, 12 groups, 3 folds, seed 7, min_data_in_bin 8"
    ) in out
    assert cv_hooks.min_data_in_bin_seen == [8, 8, 8]

    assert main(["grouped_fake", "cleargbm", "3", "7"], external_dir=tmp_path) == EXIT_OK
    assert cv_hooks.min_data_in_bin_seen == [8, 8, 8, None, None, None]


def test_min_data_in_bin_below_two_is_refused(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["grouped_fake", "cleargbm", "3", "7", "1"], external_dir=tmp_path) == (
        EXIT_BAD_USAGE
    )
    assert capsys.readouterr().out == (
        "min_data_in_bin must be >= 2 (a floor of 1 is the unset behavior)\n"
    )


def test_min_data_in_bin_with_lightgbm_is_refused(
    cv_hooks: _FakeCVBackend, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["grouped_fake", "lightgbm", "3", "7", "8"], external_dir=tmp_path) == (
        EXIT_BAD_USAGE
    )
    assert capsys.readouterr().out == (
        "min_data_in_bin is a ClearGBM protocol variant; the LightGBM backend's own "
        "floor is not exposed here\n"
    )


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
