"""Integration tests for ML routes with real implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_domain import Deal, DealId, Measurement
from covenant_ml.testing import make_train_config
from covenant_ml.trainer import save_model, train_model
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_float,
    require_str,
)

from covenant_radar_api.api.routes.ml import build_router

from .conftest import ContainerAndStore


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    app = FastAPI()
    router = build_router(cas.container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


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


class TestPredictEndpoint:
    """Tests for POST /ml/predict."""

    def test_predict_returns_probability_and_tier(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test prediction with real XGBoost model."""
        # Create model file in the path expected by container
        model_path = Path(container_with_store.container.get_model_info()["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _create_and_save_model(model_path)

        _add_test_deal(container_with_store, "d1", "Technology", "North America")
        _add_test_measurements(container_with_store, "d1")

        client = _create_test_client(container_with_store)
        response = client.post("/ml/predict", content=b'{"deal_id": "d1"}')

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "deal_id") == "d1"
        # Verify probability is a float in valid range
        prob_val = float(str(data["probability"]))
        assert 0.0 <= prob_val <= 1.0
        assert require_str(data, "risk_tier") in ("LOW", "MEDIUM", "HIGH")

    def test_predict_deal_not_found(self, container_with_store: ContainerAndStore) -> None:
        """Test prediction with nonexistent deal."""
        model_path = Path(container_with_store.container.get_model_info()["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _create_and_save_model(model_path)

        client = _create_test_client(container_with_store)
        response = client.post("/ml/predict", content=b'{"deal_id": "nonexistent"}')

        # KeyError from deal_repo.get()
        assert response.status_code == 500

    def test_predict_missing_measurements(self, container_with_store: ContainerAndStore) -> None:
        """Test prediction with deal that has no measurements."""
        model_path = Path(container_with_store.container.get_model_info()["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _create_and_save_model(model_path)

        _add_test_deal(container_with_store, "d1", "Technology", "North America")
        # No measurements added

        client = _create_test_client(container_with_store)
        response = client.post("/ml/predict", content=b'{"deal_id": "d1"}')

        # KeyError from missing metrics
        assert response.status_code == 500

    def test_predict_invalid_json(self, container_with_store: ContainerAndStore) -> None:
        """Test prediction with invalid JSON."""
        model_path = Path(container_with_store.container.get_model_info()["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _create_and_save_model(model_path)

        client = _create_test_client(container_with_store)
        response = client.post("/ml/predict", content=b"not json")

        assert response.status_code == 500


class TestTrainEndpoint:
    """Tests for POST /ml/train."""

    def test_train_enqueues_job(self, container_with_store: ContainerAndStore) -> None:
        """Test training job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train",
            content=b"""{
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "device": "cuda",
                "scale_pos_weight": 1.5
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        # Verify job_id is the expected fake job id
        assert require_str(data, "job_id") == "test-job-id"
        # Verify enqueued payload contains device and scale_pos_weight
        enqueued = container_with_store.queue.jobs[-1]
        config_payload = narrow_json_to_dict(load_json_str(str(enqueued.args[0])))
        assert require_str(config_payload, "device") == "cuda"
        assert require_float(config_payload, "scale_pos_weight") == 1.5

    def test_train_enqueues_job_without_scale_weight(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Train endpoint enqueues job with defaults when scale_pos_weight omitted."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train",
            content=b"""{
                "learning_rate": 0.2,
                "max_depth": 4,
                "n_estimators": 50,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
                "random_state": 99
            }""",
        )

        assert response.status_code == 202
        enqueued = container_with_store.queue.jobs[-1]
        config_payload = narrow_json_to_dict(load_json_str(str(enqueued.args[0])))
        assert require_str(config_payload, "device") == "auto"
        assert "scale_pos_weight" not in config_payload

    def test_train_invalid_json(self, container_with_store: ContainerAndStore) -> None:
        """Test training with invalid JSON."""
        client = _create_test_client(container_with_store)
        response = client.post("/ml/train", content=b"not json")

        assert response.status_code == 500

    def test_train_missing_field(self, container_with_store: ContainerAndStore) -> None:
        """Test training with missing field."""
        client = _create_test_client(container_with_store)
        response = client.post("/ml/train", content=b'{"learning_rate": 0.1}')

        assert response.status_code == 500


class TestModelsActiveEndpoint:
    """Tests for GET /ml/models/active."""

    def test_get_model_info(self, container_with_store: ContainerAndStore) -> None:
        """Test getting active model info."""
        client = _create_test_client(container_with_store)
        response = client.get("/ml/models/active")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "model_id") == "default"
        assert require_str(data, "model_path").endswith("test_model.ubj")
        assert require_bool(data, "is_loaded") is False


class TestJobStatusEndpoint:
    """Tests for GET /ml/jobs/{job_id}."""

    def test_get_job_status_not_found(self, container_with_store: ContainerAndStore) -> None:
        """Test getting status of non-existent job."""
        from platform_workers.testing import hooks, make_fake_fetch_job_not_found

        hooks.fetch_job = make_fake_fetch_job_not_found()

        client = _create_test_client(container_with_store)
        response = client.get("/ml/jobs/nonexistent-job-id")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "job_id") == "nonexistent-job-id"
        assert require_str(data, "status") == "not_found"
        assert "result" not in data

    def test_get_job_status_queued(self, container_with_store: ContainerAndStore) -> None:
        """Test getting status of queued job."""
        from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

        fake_job = FakeFetchedJob(job_id="job-queued", status="queued", result=None)
        hooks.fetch_job = make_fake_fetch_job_found(fake_job)

        client = _create_test_client(container_with_store)
        response = client.get("/ml/jobs/job-queued")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "job_id") == "job-queued"
        assert require_str(data, "status") == "queued"

    def test_get_job_status_finished_with_result(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test getting status of finished job with result."""
        from platform_workers.testing import FakeFetchedJob, hooks, make_fake_fetch_job_found

        fake_job = FakeFetchedJob(
            job_id="job-finished",
            status="finished",
            result={"model_path": "/path/to/model.ubj"},
        )
        hooks.fetch_job = make_fake_fetch_job_found(fake_job)

        client = _create_test_client(container_with_store)
        response = client.get("/ml/jobs/job-finished")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "job_id") == "job-finished"
        assert require_str(data, "status") == "finished"
        # Verify result is present and is a dict with expected content
        result = data.get("result")
        assert type(result) is dict
        assert result.get("model_path") == "/path/to/model.ubj"


class TestTrainExternalEndpoint:
    """Tests for POST /ml/train-external."""

    def test_train_external_enqueues_job(self, container_with_store: ContainerAndStore) -> None:
        """Test external training job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external",
            content=b"""{
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        assert require_str(data, "job_id") == "test-job-id"

        # Verify job was enqueued with correct function
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_external_train_job" in enqueued.func

    def test_train_external_passes_raw_config(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """External training passes raw JSON config to job."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external",
            content=(
                b'{"dataset":"us","learning_rate":0.2,"max_depth":4,'
                b'"n_estimators":50,"subsample":0.9,"colsample_bytree":0.9,'
                b'"random_state":99}'
            ),
        )

        assert response.status_code == 202

        # Verify raw JSON was passed
        enqueued = container_with_store.queue.jobs[-1]
        assert "us" in str(enqueued.args[0])
