"""Shared fixtures and helpers for test_explain_job splits."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import numpy as np
from numpy.typing import NDArray


def _copy_real_taiwan(external_root: Path) -> tuple[Path, int, list[str]]:
    """Copy full Taiwan dataset into external_root and return (path, n_rows, feature_names)."""
    src = Path(__file__).parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
    if not src.exists():
        raise FileNotFoundError("Taiwan dataset not found in repository data")
    dst_dir = external_root / "taiwan_data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.csv"
    copyfile(str(src), str(dst))
    header = (dst.read_text(encoding="utf-8").splitlines())[0]
    cols = [c.strip() for c in header.split(",")]
    feature_names = cols[1:]  # all columns after label
    n_rows = sum(1 for _ in dst.open(encoding="utf-8")) - 1
    return dst, n_rows, feature_names


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
