"""Integration tests for ML routes with real implementations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)
from covenant_ml.predictor import load_model
from covenant_ml.trainer import save_model, train_model
from covenant_ml.types import TrainConfig, XGBModelProtocol
from covenant_persistence import (
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from platform_core.json_utils import load_json_str
from platform_workers.rq_harness import RQClientQueue, RQJobLike, RQRetryLike

from covenant_radar_api.api.routes.ml import ModelInfo, build_router

# Recursive JSON type matching platform_workers.rq_harness._JsonValue
_JsonValue = dict[str, "_JsonValue"] | list["_JsonValue"] | str | int | float | bool | None


class _InMemoryStore:
    """In-memory storage for ML test data."""

    def __init__(self) -> None:
        self.deals: dict[str, Deal] = {}
        self.measurements: list[Measurement] = []
        self.results: list[CovenantResult] = []
        self.covenants: dict[str, Covenant] = {}
        self.enqueued_jobs: list[tuple[str, tuple[str, ...]]] = []


class _InMemoryDealRepository:
    """In-memory implementation of DealRepository."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def create(self, deal: Deal) -> None:
        """Insert new deal."""
        deal_id = deal["id"]["value"]
        if deal_id in self._store.deals:
            raise ValueError(f"Deal already exists: {deal_id}")
        self._store.deals[deal_id] = deal

    def get(self, deal_id: DealId) -> Deal:
        """Get deal by ID."""
        key = deal_id["value"]
        if key not in self._store.deals:
            raise KeyError(f"Deal not found: {key}")
        return self._store.deals[key]

    def list_all(self) -> Sequence[Deal]:
        """List all deals."""
        return list(self._store.deals.values())

    def update(self, deal: Deal) -> None:
        """Update existing deal."""
        key = deal["id"]["value"]
        if key not in self._store.deals:
            raise KeyError(f"Deal not found: {key}")
        self._store.deals[key] = deal

    def delete(self, deal_id: DealId) -> None:
        """Delete deal."""
        key = deal_id["value"]
        if key not in self._store.deals:
            raise KeyError(f"Deal not found: {key}")
        del self._store.deals[key]


class _InMemoryMeasurementRepository:
    """In-memory implementation of MeasurementRepository."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def add_many(self, measurements: Sequence[Measurement]) -> int:
        """Insert measurements."""
        for m in measurements:
            self._store.measurements.append(m)
        return len(measurements)

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal."""
        result: list[Measurement] = []
        for m in self._store.measurements:
            if m["deal_id"]["value"] == deal_id["value"]:
                result.append(m)
        return result

    def list_for_deal_and_period(
        self,
        deal_id: DealId,
        period_start_iso: str,
        period_end_iso: str,
    ) -> Sequence[Measurement]:
        """List measurements for deal and period."""
        result: list[Measurement] = []
        for m in self._store.measurements:
            if (
                m["deal_id"]["value"] == deal_id["value"]
                and m["period_start_iso"] == period_start_iso
                and m["period_end_iso"] == period_end_iso
            ):
                result.append(m)
        return result


class _InMemoryCovenantResultRepository:
    """In-memory implementation of CovenantResultRepository."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store

    def save(self, result: CovenantResult) -> None:
        """Save result."""
        self._store.results.append(result)

    def save_many(self, results: Sequence[CovenantResult]) -> int:
        """Save multiple results."""
        for r in results:
            self._store.results.append(r)
        return len(results)

    def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
        """List results for a deal's covenants."""
        cov_ids: set[str] = set()
        for cov in self._store.covenants.values():
            if cov["deal_id"]["value"] == deal_id["value"]:
                cov_ids.add(cov["id"]["value"])
        result: list[CovenantResult] = []
        for r in self._store.results:
            if r["covenant_id"]["value"] in cov_ids:
                result.append(r)
        return result

    def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
        """List results for a covenant."""
        result: list[CovenantResult] = []
        for r in self._store.results:
            if r["covenant_id"]["value"] == covenant_id["value"]:
                result.append(r)
        return result


class _FakeJob:
    """Fake RQ job for testing."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def get_id(self) -> str:
        """Return job ID."""
        return self._job_id


class _InMemoryRQQueue:
    """In-memory RQ queue that records enqueued jobs."""

    def __init__(self, store: _InMemoryStore) -> None:
        self._store = store
        self._job_counter = 0

    def enqueue(
        self,
        func_ref: str,
        *args: _JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        """Enqueue a job and return a fake job object."""
        self._job_counter += 1
        job_id = f"job-{self._job_counter}"
        str_args = tuple(str(a) for a in args)
        self._store.enqueued_jobs.append((func_ref, str_args))
        job: RQJobLike = _FakeJob(job_id)
        return job


class _TestContainer:
    """Test container for ML routes with real XGBoost model."""

    def __init__(
        self,
        store: _InMemoryStore,
        model: XGBModelProtocol,
        model_path: str,
    ) -> None:
        self._store = store
        self._model = model
        self._model_path = model_path
        self._sector_encoder: dict[str, int] = {"Technology": 0, "Finance": 1, "Healthcare": 2}
        self._region_encoder: dict[str, int] = {"North America": 0, "Europe": 1, "Asia": 2}

    def deal_repo(self) -> DealRepository:
        """Return in-memory deal repository."""
        repo: DealRepository = _InMemoryDealRepository(self._store)
        return repo

    def measurement_repo(self) -> MeasurementRepository:
        """Return in-memory measurement repository."""
        repo: MeasurementRepository = _InMemoryMeasurementRepository(self._store)
        return repo

    def covenant_result_repo(self) -> CovenantResultRepository:
        """Return in-memory result repository."""
        repo: CovenantResultRepository = _InMemoryCovenantResultRepository(self._store)
        return repo

    def rq_queue(self) -> RQClientQueue:
        """Return in-memory RQ queue."""
        queue: RQClientQueue = _InMemoryRQQueue(self._store)
        return queue

    def get_model(self) -> XGBModelProtocol:
        """Return the XGBoost model."""
        return self._model

    def get_model_info(self) -> ModelInfo:
        """Return model info."""
        return ModelInfo(
            model_id="test-model-001",
            model_path=self._model_path,
            is_loaded=True,
        )

    def get_sector_encoder(self) -> dict[str, int]:
        """Return sector encoder."""
        return self._sector_encoder

    def get_region_encoder(self) -> dict[str, int]:
        """Return region encoder."""
        return self._region_encoder


def _create_real_model(tmp_path: Path) -> tuple[XGBModelProtocol, str]:
    """Create a real trained XGBoost model for testing."""
    # Create small training data using np.zeros and assignment to avoid list[Any]
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

    config = TrainConfig(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=10,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    )

    model = train_model(x_train, y_train, config)
    model_path = tmp_path / "test_model.ubj"
    save_model(model, str(model_path))

    # Load it back to verify
    loaded_model = load_model(str(model_path))
    return loaded_model, str(model_path)


def _add_test_deal(store: _InMemoryStore, deal_id: str, sector: str, region: str) -> None:
    """Add a test deal to store."""
    store.deals[deal_id] = Deal(
        id=DealId(value=deal_id),
        name="Test Deal",
        borrower="Test Corp",
        sector=sector,
        region=region,
        commitment_amount_cents=100_000_000,
        currency="USD",
        maturity_date_iso="2025-12-31",
    )


def _add_test_measurements(store: _InMemoryStore, deal_id: str) -> None:
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
            store.measurements.append(
                Measurement(
                    deal_id=DealId(value=deal_id),
                    period_start_iso=period_start,
                    period_end_iso=period_end,
                    metric_name=metric_name,
                    metric_value_scaled=value,
                )
            )


def _create_test_client(
    store: _InMemoryStore,
    model: XGBModelProtocol,
    model_path: str,
) -> TestClient:
    """Create test client with real model."""
    app = FastAPI()
    container = _TestContainer(store, model, model_path)
    router = build_router(container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestPredictEndpoint:
    """Tests for POST /ml/predict."""

    def test_predict_returns_probability_and_tier(self, tmp_path: Path) -> None:
        """Test prediction with real XGBoost model."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()
        _add_test_deal(store, "d1", "Technology", "North America")
        _add_test_measurements(store, "d1")

        client = _create_test_client(store, model, model_path)
        response = client.post("/ml/predict", content=b'{"deal_id": "d1"}')

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert isinstance(data, dict)
        assert data["deal_id"] == "d1"
        assert "probability" in data
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH")

    def test_predict_deal_not_found(self, tmp_path: Path) -> None:
        """Test prediction with nonexistent deal."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()

        client = _create_test_client(store, model, model_path)
        response = client.post("/ml/predict", content=b'{"deal_id": "nonexistent"}')

        # KeyError from deal_repo.get()
        assert response.status_code == 500

    def test_predict_missing_measurements(self, tmp_path: Path) -> None:
        """Test prediction with deal that has no measurements."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()
        _add_test_deal(store, "d1", "Technology", "North America")
        # No measurements added

        client = _create_test_client(store, model, model_path)
        response = client.post("/ml/predict", content=b'{"deal_id": "d1"}')

        # KeyError from missing metrics
        assert response.status_code == 500

    def test_predict_invalid_json(self, tmp_path: Path) -> None:
        """Test prediction with invalid JSON."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()

        client = _create_test_client(store, model, model_path)
        response = client.post("/ml/predict", content=b"not json")

        assert response.status_code == 500


class TestTrainEndpoint:
    """Tests for POST /ml/train."""

    def test_train_enqueues_job(self, tmp_path: Path) -> None:
        """Test training job is enqueued."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()

        client = _create_test_client(store, model, model_path)
        response = client.post(
            "/ml/train",
            content=b"""{
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }""",
        )

        assert response.status_code == 202
        data = load_json_str(response.text)
        assert isinstance(data, dict)
        assert "job_id" in data
        assert data["status"] == "queued"

        # Verify job was enqueued
        assert len(store.enqueued_jobs) == 1
        func_ref, _args = store.enqueued_jobs[0]
        assert func_ref == "covenant_radar_api.worker.train_job.run_training"

    def test_train_invalid_json(self, tmp_path: Path) -> None:
        """Test training with invalid JSON."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()

        client = _create_test_client(store, model, model_path)
        response = client.post("/ml/train", content=b"not json")

        assert response.status_code == 500

    def test_train_missing_field(self, tmp_path: Path) -> None:
        """Test training with missing field."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()

        client = _create_test_client(store, model, model_path)
        response = client.post("/ml/train", content=b'{"learning_rate": 0.1}')

        assert response.status_code == 500


class TestModelsActiveEndpoint:
    """Tests for GET /ml/models/active."""

    def test_get_model_info(self, tmp_path: Path) -> None:
        """Test getting active model info."""
        model, model_path = _create_real_model(tmp_path)
        store = _InMemoryStore()

        client = _create_test_client(store, model, model_path)
        response = client.get("/ml/models/active")

        assert response.status_code == 200
        data = load_json_str(response.text)
        assert isinstance(data, dict)
        assert data["model_id"] == "test-model-001"
        assert data["model_path"] == model_path
        assert data["is_loaded"] is True
