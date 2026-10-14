"""Fixed small boosted learner, fitted only to explicitly supplied earlier rows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, require_float, require_int

from ..backends.lightgbm.backend import _get_lightgbm_imports, _LGBModelProtocol
from .forecast_model import WEATHER_FEATURES, features
from .forecast_records import ForecastFeatures, ForecastRow


def boosting_parameters() -> JSONObject:
    """Return the single declared configuration, with no data-dependent selection.

    Returns:
        JSON settings used by both fitting and the recorded report.
    """
    return {
        "n_estimators": 20,
        "learning_rate": 0.05,
        "num_leaves": 4,
        "max_depth": 2,
        "min_child_samples": 50,
        "reg_alpha": 0.0,
        "reg_lambda": 10.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 0,
        "n_jobs": 1,
        "objective": "binary",
        "metric": "binary_logloss",
        "device": "cpu",
        "boosting_type": "gbdt",
        "class_weight": None,
        "early_stopping": False,
    }


def feature_matrix(rows: Sequence[ForecastFeatures]) -> NDArray[np.float64]:
    """Materialize the existing fixed forecast features in their declared order.

    Args:
        rows: Outcome-free feature records.

    Returns:
        Float64 matrix with four columns.
    """
    values: list[tuple[float, ...]] = [features(row, True) for row in rows]
    return np.asarray(values, dtype=np.float64)


def fit_boosting(rows: Sequence[ForecastRow], artifact: Path) -> _LGBModelProtocol:
    """Fit and save one native model without evaluation data or early stopping.

    Args:
        rows: Earlier observations whose labels are available before prediction.
        artifact: New native model path in the experiment's reserved directory.

    Returns:
        Actual fitted LightGBM model.

    Raises:
        ValueError: BOOST-001 if both training classes are not present.
        FileExistsError: BOOST-002 if a model would overwrite existing evidence.
        OSError: Native model persistence fails.
    """
    if {row["yes"] for row in rows} != {0, 1}:
        raise ValueError("[BOOST-001] training requires both binary classes")
    if artifact.exists():
        raise FileExistsError("[BOOST-002] native model artifact already exists")
    parameters = boosting_parameters()
    constructor, _ = _get_lightgbm_imports()
    model = constructor(
        boosting_type="gbdt",
        objective="binary",
        metric="binary_logloss",
        device="cpu",
        class_weight=None,
        verbose=-1,
        n_estimators=require_int(parameters, "n_estimators"),
        learning_rate=require_float(parameters, "learning_rate"),
        num_leaves=require_int(parameters, "num_leaves"),
        max_depth=require_int(parameters, "max_depth"),
        min_child_samples=require_int(parameters, "min_child_samples"),
        reg_alpha=require_float(parameters, "reg_alpha"),
        reg_lambda=require_float(parameters, "reg_lambda"),
        subsample=require_float(parameters, "subsample"),
        colsample_bytree=require_float(parameters, "colsample_bytree"),
        random_state=require_int(parameters, "random_state"),
        n_jobs=require_int(parameters, "n_jobs"),
    )
    labels: list[int] = [row["yes"] for row in rows]
    y: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    model.fit(feature_matrix(rows), y, eval_set=[], feature_name=list(WEATHER_FEATURES))
    model.booster_.save_model(str(artifact))
    return model
