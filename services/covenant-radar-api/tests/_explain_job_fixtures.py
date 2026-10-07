"""Shared fixtures and helpers for test_explain_job splits.

The real Taiwan copy these tests explain over is ``tests/_real_datasets.py``'s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _create_xgboost_model(model_path: Path, n_features: int, n_samples: int = 100) -> None:
    """Create a simple XGBoost model for testing."""
    import xgboost as xgb

    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((n_samples, n_features))
    y: NDArray[np.int64] = rng.integers(0, 2, size=n_samples).astype(np.int64)

    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(x, y)
    model.save_model(str(model_path))
