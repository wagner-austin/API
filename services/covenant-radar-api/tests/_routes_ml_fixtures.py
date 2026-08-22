"""Shared fixtures and helpers for test_routes_ml splits."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from covenant_domain import Deal, DealId, Measurement
from covenant_ml.testing import make_train_config
from covenant_ml.trainer_fit import (
    save_model,
    train_model,
)
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from covenant_radar_api.api.routes.ml import build_router

from .conftest import ContainerAndStore, make_route_test_client


class _XGBRegressorProto(Protocol):
    """Protocol for XGBRegressor interface used in test helpers."""

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> _XGBRegressorProto: ...

    def save_model(self, fname: str) -> None: ...


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    return make_route_test_client(build_router(cas.container))


def _create_and_save_model(model_path: Path) -> None:
    """Create a real trained XGBoost model for testing."""
    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    # Row 0: Low risk
    x_train[0, 0] = 2.0
    x_train[0, 1] = 5.0
    x_train[0, 2] = 1.5
    x_train[0, 3] = 0.1
    x_train[0, 4] = 0.2
    x_train[0, 5] = 0.0
    x_train[0, 6] = 0.0
    x_train[0, 7] = 0.0
    # Row 1: Low risk
    x_train[1, 0] = 2.5
    x_train[1, 1] = 4.0
    x_train[1, 2] = 1.3
    x_train[1, 3] = 0.2
    x_train[1, 4] = 0.3
    x_train[1, 5] = 1.0
    x_train[1, 6] = 1.0
    x_train[1, 7] = 1.0
    # Row 2: High risk
    x_train[2, 0] = 5.0
    x_train[2, 1] = 1.5
    x_train[2, 2] = 0.8
    x_train[2, 3] = 0.5
    x_train[2, 4] = 1.0
    x_train[2, 5] = 0.0
    x_train[2, 6] = 0.0
    x_train[2, 7] = 3.0
    # Row 3: High risk
    x_train[3, 0] = 6.0
    x_train[3, 1] = 1.0
    x_train[3, 2] = 0.6
    x_train[3, 3] = 0.8
    x_train[3, 4] = 1.5
    x_train[3, 5] = 1.0
    x_train[3, 6] = 1.0
    x_train[3, 7] = 4.0

    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[0] = 0
    y_train[1] = 0
    y_train[2] = 1
    y_train[3] = 1

    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    model = train_model(x_train, y_train, config)
    save_model(model, str(model_path))


def _add_test_deal(cas: ContainerAndStore, deal_id: str, sector: str, region: str) -> None:
    """Add a test deal to store."""
    cas.store.deals[deal_id] = Deal(
        id=DealId(value=deal_id),
        name="Test Deal",
        borrower="Test Corp",
        sector=sector,
        region=region,
        commitment_amount_cents=100_000_000,
        currency="USD",
        maturity_date_iso="2025-12-31",
    )
    cas.store._deal_order.append(deal_id)


def _add_test_measurements(cas: ContainerAndStore, deal_id: str) -> None:
    """Add test measurements for multiple periods."""
    periods = [
        ("2024-01-01", "2024-03-31"),
        ("2023-10-01", "2023-12-31"),
        ("2023-07-01", "2023-09-30"),
        ("2023-04-01", "2023-06-30"),
        ("2023-01-01", "2023-03-31"),
    ]
    metrics = {
        "total_debt": 10_000_000,
        "ebitda": 5_000_000,
        "interest_expense": 1_000_000,
        "current_assets": 8_000_000,
        "current_liabilities": 5_000_000,
    }
    for period_start, period_end in periods:
        for metric_name, value in metrics.items():
            cas.store.measurements.append(
                Measurement(
                    deal_id=DealId(value=deal_id),
                    period_start_iso=period_start,
                    period_end_iso=period_end,
                    metric_name=metric_name,
                    metric_value_scaled=value,
                )
            )


def _create_and_save_xgb_regressor(model_path: Path) -> None:
    """Create and save a real XGBoost regressor model for testing.

    Args:
        model_path: Path to save the model (.ubj format).
    """
    xgb_mod = __import__("xgboost")
    regressor: _XGBRegressorProto = xgb_mod.XGBRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.3, random_state=42
    )

    x_train: NDArray[np.float64] = np.arange(1.0, 13.0, dtype=np.float64).reshape(4, 3)
    y_train: NDArray[np.float64] = np.arange(1.5, 5.5, 1.0, dtype=np.float64)

    regressor.fit(x_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    regressor.save_model(str(model_path))
