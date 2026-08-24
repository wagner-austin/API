"""Registry-driven regression benchmark: four arms on one dataset.

The standing regression scoreboard's harness: ClearGBM (both growth
policies), LightGBM and XGBoost train on one verified regression dataset
under the matched P1 protocol (300 rounds, lr 0.05, depth 6 with the
leaf-wise arms budgeted at 31 leaves, 20 rows/leaf minimum, reg_lambda
1.0, early stopping 30 on validation) and score the same held-out test
split by RMSE, MAE and R², with per-arm fit wall clock recorded.

The split honours the dataset's own grouping law: when the registry
config names a ``group_column`` (rw_value's ``match``), whole groups land
in one split — 1,500 correlated snapshots of one match must never
straddle train and test, or memorization scores as skill. Row-independent
datasets (financial_distress) shuffle rows, reproducing the P1 protocol
exactly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from cleargbm.ensemble import predict_raw, train_gradient_boosting_regression
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from ..datasets.loader import DatasetLoader
from ..datasets.registry import make_default_regression_registry
from ..metrics_regression import compute_mae, compute_r_squared, compute_rmse


class RegressionBenchConfig(TypedDict):
    """Shared hyperparameters for every arm of the regression benchmark.

    Args:
        dataset: Registry name of the regression dataset.
        n_estimators: Boosting rounds for every arm.
        max_depth: Maximum tree depth (depth-wise arms).
        num_leaves: Leaf budget for the leaf-wise arms.
        learning_rate: Shrinkage for every arm.
        max_bins: Histogram bin count for every arm.
        min_samples_leaf: Minimum rows per leaf for every arm.
        early_stopping_rounds: Patience on validation loss.
    """

    dataset: str
    n_estimators: int
    max_depth: int
    num_leaves: int
    learning_rate: float
    max_bins: int
    min_samples_leaf: int
    early_stopping_rounds: int


class RegressionQuality(TypedDict):
    """Held-out quality for one arm at one seed.

    Args:
        rmse: Root mean squared error on the test split.
        mae: Mean absolute error on the test split.
        r_squared: Coefficient of determination on the test split.
    """

    rmse: float
    mae: float
    r_squared: float


class RegressionArmResult(TypedDict):
    """One arm's measurement at one seed.

    Args:
        model: Arm name (``"cleargbm"``, ``"cleargbm@leaf_wise"``,
            ``"lightgbm"`` or ``"xgboost"``).
        seed: Split seed.
        quality: Held-out quality record.
        fit_seconds: Wall-clock training time for this arm.
    """

    model: str
    seed: int
    quality: RegressionQuality
    fit_seconds: float


class RegressionManifest(TypedDict):
    """Complete regression benchmark manifest.

    Args:
        config: The shared hyperparameters.
        seeds: Every split seed measured.
        grouped: Whether the split was by group (the dataset's law).
        results: One record per arm per seed.
    """

    config: RegressionBenchConfig
    seeds: list[int]
    grouped: bool
    results: list[RegressionArmResult]


class _LGBMRegProto(Protocol):
    """Protocol for the LightGBM regressor members this module uses."""

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        eval_set: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        """Fit the regressor with a validation set for early stopping.

        Args:
            x: Feature matrix.
            y: Continuous targets.
            eval_set: Validation pairs for early stopping.
        """
        ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix.

        Returns:
            Predictions, shape (n_samples,).
        """
        ...


class _XGBRegProto(Protocol):
    """Protocol for the XGBoost regressor members this module uses.

    XGBoost's fit-time chatter is silenced per call (``verbose=False``)
    rather than per constructor, which is where its API puts the switch.
    """

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        eval_set: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
        verbose: bool,
    ) -> None:
        """Fit the regressor with a validation set for early stopping.

        Args:
            x: Feature matrix.
            y: Continuous targets.
            eval_set: Validation pairs for early stopping.
            verbose: Whether to print per-round evaluation lines.
        """
        ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix.

        Returns:
            Predictions, shape (n_samples,).
        """
        ...


class _LGBMRegCtor(Protocol):
    """Protocol for the LightGBM regressor constructor."""

    def __call__(
        self,
        *,
        objective: str,
        n_estimators: int,
        max_depth: int,
        num_leaves: int,
        learning_rate: float,
        max_bin: int,
        min_child_samples: int,
        reg_alpha: float,
        reg_lambda: float,
        early_stopping_rounds: int,
        n_jobs: int,
        random_state: int,
        verbose: int,
    ) -> _LGBMRegProto: ...


class _XGBRegCtor(Protocol):
    """Protocol for the XGBoost regressor constructor."""

    def __call__(
        self,
        *,
        objective: str,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        max_bin: int,
        min_child_weight: int,
        reg_alpha: float,
        reg_lambda: float,
        early_stopping_rounds: int,
        tree_method: str,
        n_jobs: int,
        random_state: int,
        verbosity: int,
    ) -> _XGBRegProto: ...


def _load_lightgbm_reg_ctor() -> _LGBMRegCtor:
    """Resolve LightGBM's regressor constructor as a Protocol-typed callable.

    Returns:
        The ``LGBMRegressor`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMRegressor"])
    constructor: _LGBMRegCtor = module.LGBMRegressor
    return constructor


def _load_xgboost_reg_ctor() -> _XGBRegCtor:
    """Resolve XGBoost's regressor constructor as a Protocol-typed callable.

    Returns:
        The ``XGBRegressor`` constructor.
    """
    module = __import__("xgboost", fromlist=["XGBRegressor"])
    constructor: _XGBRegCtor = module.XGBRegressor
    return constructor


class SplitIndices(TypedDict):
    """Row indices of one train/val/test partition.

    Args:
        train: Training row indices.
        val: Validation row indices.
        test: Test row indices.
    """

    train: NDArray[np.intp]
    val: NDArray[np.intp]
    test: NDArray[np.intp]


def split_rows(
    n_samples: int,
    groups: NDArray[np.int64] | None,
    seed: int,
) -> SplitIndices:
    """Partition rows 0.6/0.2/0.2 into train/val/test.

    With ``groups``, the units shuffled and partitioned are the GROUPS:
    every row of a group lands in one split. Without, rows shuffle
    directly — the P1 protocol.

    Args:
        n_samples: Total row count.
        groups: Group codes per row, or None for row-independent data.
        seed: Shuffle seed.

    Returns:
        The three index arrays.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        order: NDArray[np.intp] = rng.permutation(n_samples)
        n_train = int(n_samples * 0.6)
        n_val = int(n_samples * 0.2)
        return SplitIndices(
            train=order[:n_train],
            val=order[n_train : n_train + n_val],
            test=order[n_train + n_val :],
        )

    unique_groups: NDArray[np.int64] = np.unique(groups)
    group_order: NDArray[np.intp] = rng.permutation(len(unique_groups))
    shuffled: NDArray[np.int64] = unique_groups[group_order]
    n_groups = len(shuffled)
    g_train = int(n_groups * 0.6)
    g_val = int(n_groups * 0.2)
    train_groups = shuffled[:g_train]
    val_groups = shuffled[g_train : g_train + g_val]
    train_mask: NDArray[np.bool_] = np.isin(groups, train_groups)
    val_mask: NDArray[np.bool_] = np.isin(groups, val_groups)
    test_mask: NDArray[np.bool_] = ~(train_mask | val_mask)
    return SplitIndices(
        train=np.flatnonzero(train_mask),
        val=np.flatnonzero(val_mask),
        test=np.flatnonzero(test_mask),
    )


def _cleargbm_config(
    config: RegressionBenchConfig,
    seed: int,
    leaf_wise: bool,
) -> GradientBoostingConfig:
    """Build the ClearGBM training config for one arm run.

    Args:
        config: Shared hyperparameters.
        seed: Random seed for the run.
        leaf_wise: Whether the arm grows best-first.

    Returns:
        The full ClearGBM config.
    """
    return GradientBoostingConfig(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_samples_split=2 * config["min_samples_leaf"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=None,
        lambdarank_truncation_level=None,
        goss_top_rate=None,
        goss_other_rate=None,
        quantized_gradient_bins=None,
        max_bins=config["max_bins"],
        subsample=1.0,
        random_state=seed,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
        early_stopping_rounds=config["early_stopping_rounds"],
        growth_strategy="leaf_wise" if leaf_wise else "depth_wise",
        num_leaves=config["num_leaves"] if leaf_wise else None,
        objective="squared_error",
        scale_pos_weight=None,
    )


def _quality(
    y_test: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> RegressionQuality:
    """Score one arm's held-out predictions.

    Args:
        y_test: Held-out targets.
        predictions: Predicted values.

    Returns:
        The arm's quality record.
    """
    return RegressionQuality(
        rmse=compute_rmse(y_test, predictions),
        mae=compute_mae(y_test, predictions),
        r_squared=compute_r_squared(y_test, predictions),
    )


def run_regression_benchmark(
    config: RegressionBenchConfig,
    seeds: list[int],
    external_dir: Path,
) -> RegressionManifest:
    """Run all four arms across every seed, timing each fit.

    Args:
        config: Shared hyperparameters.
        seeds: Split seeds to measure.
        external_dir: Root directory holding the registered datasets.

    Returns:
        The complete manifest.
    """
    registry = make_default_regression_registry()
    dataset_config = registry.get(config["dataset"])
    dataset = DatasetLoader().load_regression(dataset_config, external_dir)
    x = dataset["x"]
    y = dataset["y"]
    groups = dataset["groups"]
    feature_names = dataset["meta"]["feature_names"]

    results: list[RegressionArmResult] = []
    for seed in seeds:
        indices = split_rows(len(y), groups, seed)
        x_train, y_train = x[indices["train"]], y[indices["train"]]
        x_val, y_val = x[indices["val"]], y[indices["val"]]
        x_test, y_test = x[indices["test"]], y[indices["test"]]

        for leaf_wise in [False, True]:
            started = time.perf_counter()
            model = train_gradient_boosting_regression(
                x_train,
                y_train,
                x_val,
                y_val,
                _cleargbm_config(config, seed, leaf_wise),
                feature_names,
            )
            fit_seconds = time.perf_counter() - started
            results.append(
                RegressionArmResult(
                    model="cleargbm@leaf_wise" if leaf_wise else "cleargbm",
                    seed=seed,
                    quality=_quality(y_test, predict_raw(model, x_test)),
                    fit_seconds=fit_seconds,
                )
            )

        lgbm = _load_lightgbm_reg_ctor()(
            objective="regression",
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            num_leaves=config["num_leaves"],
            learning_rate=config["learning_rate"],
            max_bin=config["max_bins"],
            min_child_samples=config["min_samples_leaf"],
            reg_alpha=0.0,
            reg_lambda=1.0,
            early_stopping_rounds=config["early_stopping_rounds"],
            n_jobs=1,
            random_state=seed,
            verbose=-1,
        )
        started = time.perf_counter()
        lgbm.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        fit_seconds = time.perf_counter() - started
        results.append(
            RegressionArmResult(
                model="lightgbm",
                seed=seed,
                quality=_quality(y_test, np.asarray(lgbm.predict(x_test), dtype=np.float64)),
                fit_seconds=fit_seconds,
            )
        )

        xgb = _load_xgboost_reg_ctor()(
            objective="reg:squarederror",
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            max_bin=config["max_bins"],
            min_child_weight=config["min_samples_leaf"],
            reg_alpha=0.0,
            reg_lambda=1.0,
            early_stopping_rounds=config["early_stopping_rounds"],
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
        started = time.perf_counter()
        xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        fit_seconds = time.perf_counter() - started
        results.append(
            RegressionArmResult(
                model="xgboost",
                seed=seed,
                quality=_quality(y_test, np.asarray(xgb.predict(x_test), dtype=np.float64)),
                fit_seconds=fit_seconds,
            )
        )
    return RegressionManifest(
        config=config,
        seeds=list(seeds),
        grouped=groups is not None,
        results=results,
    )


def encode_regression_manifest(manifest: RegressionManifest) -> JSONValue:
    """Encode the manifest to a JSON-serializable value.

    Args:
        manifest: The manifest to encode.

    Returns:
        JSON-shaped dictionary.
    """
    cfg = manifest["config"]
    return {
        "config": {
            "dataset": cfg["dataset"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "num_leaves": cfg["num_leaves"],
            "learning_rate": cfg["learning_rate"],
            "max_bins": cfg["max_bins"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "early_stopping_rounds": cfg["early_stopping_rounds"],
        },
        "seeds": list(manifest["seeds"]),
        "grouped": manifest["grouped"],
        "results": [
            {
                "model": r["model"],
                "seed": r["seed"],
                "quality": {
                    "rmse": r["quality"]["rmse"],
                    "mae": r["quality"]["mae"],
                    "r_squared": r["quality"]["r_squared"],
                },
                "fit_seconds": r["fit_seconds"],
            }
            for r in manifest["results"]
        ],
    }


__all__ = [
    "RegressionArmResult",
    "RegressionBenchConfig",
    "RegressionManifest",
    "RegressionQuality",
    "SplitIndices",
    "encode_regression_manifest",
    "run_regression_benchmark",
    "split_rows",
]
