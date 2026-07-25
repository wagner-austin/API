"""Integration tests for ML routes with real implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from covenant_domain import Deal, DealId, Measurement
from covenant_ml.testing import make_train_config
from covenant_ml.trainer import save_model, train_model
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_float,
    require_int,
    require_list,
    require_str,
)

from covenant_radar_api.api.routes.ml import build_router
from covenant_radar_api.worker import _regression_hooks as regression_hooks

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

        # RecordNotFoundError from deal_repo.get() maps to 404.
        assert response.status_code == 404

    def test_predict_missing_measurements(self, container_with_store: ContainerAndStore) -> None:
        """Test prediction with deal that has no measurements."""
        model_path = Path(container_with_store.container.get_model_info()["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _create_and_save_model(model_path)

        _add_test_deal(container_with_store, "d1", "Technology", "North America")
        # No measurements added

        client = _create_test_client(container_with_store)
        response = client.post("/ml/predict", content=b'{"deal_id": "d1"}')

        # A bare KeyError from missing metrics is a defect, not an absent row,
        # so it stays a 500 rather than being softened into a 404.
        assert response.status_code == 500

    def test_predict_invalid_json(self, container_with_store: ContainerAndStore) -> None:
        """Test prediction with invalid JSON."""
        model_path = Path(container_with_store.container.get_model_info()["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _create_and_save_model(model_path)

        client = _create_test_client(container_with_store)
        response = client.post("/ml/predict", content=b"not json")

        assert response.status_code == 400


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

        assert response.status_code == 400

    def test_train_missing_field(self, container_with_store: ContainerAndStore) -> None:
        """Test training with missing field."""
        client = _create_test_client(container_with_store)
        response = client.post("/ml/train", content=b'{"learning_rate": 0.1}')

        assert response.status_code == 400


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
        from platform_workers.testing import hooks as workers_hooks
        from platform_workers.testing import make_fake_fetch_job_not_found

        workers_hooks.fetch_job = make_fake_fetch_job_not_found()

        client = _create_test_client(container_with_store)
        response = client.get("/ml/jobs/nonexistent-job-id")

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "job_id") == "nonexistent-job-id"
        assert require_str(data, "status") == "not_found"
        assert "result" not in data

    def test_get_job_status_queued(self, container_with_store: ContainerAndStore) -> None:
        """Test getting status of queued job."""
        from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
        from platform_workers.testing import hooks as workers_hooks

        fake_job = FakeFetchedJob(job_id="job-queued", status="queued", result=None)
        workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

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
        from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found
        from platform_workers.testing import hooks as workers_hooks

        fake_job = FakeFetchedJob(
            job_id="job-finished",
            status="finished",
            result={"model_path": "/path/to/model.ubj"},
        )
        workers_hooks.fetch_job = make_fake_fetch_job_found(fake_job)

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

    def test_train_external_invalid_dataset_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid dataset triggers edge validation and results in error (unhandled in tests)."""
        client = _create_test_client(container_with_store)
        # Missing/invalid dataset value to trigger ValueError in decoder
        response = client.post(
            "/ml/train-external",
            content=(
                b'{"dataset":"invalid","learning_rate":0.1,"max_depth":6,'
                b'"n_estimators":100,"subsample":0.8,"colsample_bytree":0.8,'
                b'"random_state":42}'
            ),
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_train_external_non_object_json_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Non-object JSON (e.g., list) triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external",
            content=b"[]",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400


class TestTrainExternalRegressionEndpoint:
    """Tests for POST /ml/train-external-regression."""

    def test_enqueues_regression_train_job(self, container_with_store: ContainerAndStore) -> None:
        """Regression train-external job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external-regression",
            content=b"""{
                "dataset": "financial_distress",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        assert require_str(data, "job_id") == "test-job-id"

        enqueued = container_with_store.queue.jobs[-1]
        assert "process_external_regression_train_job" in enqueued.func

    def test_lightgbm_reg_enqueues_job(self, container_with_store: ContainerAndStore) -> None:
        """LightGBM regressor training job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "lightgbm_reg",
                "device": "cpu",
                "learning_rate": 0.05,
                "max_depth": 5,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }""",
        )

        assert response.status_code == 202
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_external_regression_train_job" in enqueued.func

    def test_passes_raw_json_to_worker(self, container_with_store: ContainerAndStore) -> None:
        """Raw JSON is forwarded to the worker job."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external-regression",
            content=(
                b'{"dataset":"financial_distress","learning_rate":0.2,'
                b'"max_depth":4,"n_estimators":50,"subsample":0.9,'
                b'"colsample_bytree":0.9,"random_state":99}'
            ),
        )

        assert response.status_code == 202
        enqueued = container_with_store.queue.jobs[-1]
        assert "financial_distress" in str(enqueued.args[0])

    def test_invalid_dataset_returns_400(self, container_with_store: ContainerAndStore) -> None:
        """Invalid regression dataset triggers edge validation error."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external-regression",
            content=(
                b'{"dataset":"nonexistent","learning_rate":0.1,'
                b'"max_depth":3,"n_estimators":10,"subsample":0.8,'
                b'"colsample_bytree":0.8,"random_state":42}'
            ),
        )
        assert response.status_code == 400

    def test_invalid_backend_returns_400(self, container_with_store: ContainerAndStore) -> None:
        """Invalid regressor backend triggers edge validation error."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external-regression",
            content=(
                b'{"dataset":"financial_distress","backend":"xgboost",'
                b'"learning_rate":0.1,"max_depth":3,"n_estimators":10,'
                b'"subsample":0.8,"colsample_bytree":0.8,"random_state":42}'
            ),
        )
        assert response.status_code == 400

    def test_non_object_json_returns_400(self, container_with_store: ContainerAndStore) -> None:
        """Non-object JSON triggers JSONTypeError edge validation."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/train-external-regression",
            content=b'"just a string"',
        )
        assert response.status_code == 400


class TestOptimizeEndpoint:
    """Tests for POST /ml/optimize."""

    def test_optimize_enqueues_job(self, container_with_store: ContainerAndStore) -> None:
        """Test optimization job is enqueued with unified worker."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "taiwan",
                "n_trials": 50
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        assert require_str(data, "job_id") == "test-job-id"

        # Verify job was enqueued with unified function
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_optimize_job" in enqueued.func

    def test_optimize_with_all_options(self, container_with_store: ContainerAndStore) -> None:
        """Test optimization with all common options specified."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "us",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "device": "cuda",
                "feature_preset": "log_only",
                "random_state": 123
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"

        # Verify raw JSON body is forwarded to worker
        enqueued = container_with_store.queue.jobs[-1]
        config_str = str(enqueued.args[0])
        assert "us" in config_str
        assert "cuda" in config_str
        assert "log_only" in config_str

    def test_optimize_invalid_dataset_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid dataset triggers edge validation and results in error."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "invalid",
                "n_trials": 50
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_optimize_missing_n_trials_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Missing n_trials triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "taiwan"
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_optimize_invalid_device_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid device triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "taiwan",
                "n_trials": 50,
                "device": "tpu"
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_optimize_non_object_json_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Non-object JSON (e.g., list) triggers TypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"[]",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_optimize_mlp_backend_enqueues_unified_job(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test MLP optimization enqueues unified worker job with raw JSON."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "taiwan",
                "backend": "mlp",
                "n_trials": 50,
                "precision": "fp16",
                "optimizer": "adam"
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"

        # Verify unified job was enqueued
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_optimize_job" in enqueued.func

        # Verify raw JSON body is forwarded (backend-specific fields included)
        config_str = str(enqueued.args[0])
        assert "mlp" in config_str
        assert "fp16" in config_str

    def test_optimize_lightgbm_backend_enqueues_unified_job(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test LightGBM optimization enqueues unified worker job."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "polish",
                "backend": "lightgbm",
                "n_trials": 30,
                "device": "cuda"
            }""",
        )

        assert response.status_code == 202

        # Verify unified job was enqueued
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_optimize_job" in enqueued.func

    def test_optimize_lstm_backend_enqueues_unified_job(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test LSTM optimization enqueues unified worker job."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "us",
                "backend": "lstm",
                "n_trials": 25
            }""",
        )

        assert response.status_code == 202

        # Verify unified job was enqueued
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_optimize_job" in enqueued.func

    def test_optimize_all_backends_enqueue_unified_job(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test all 7 backends enqueue the same unified worker job."""
        client = _create_test_client(container_with_store)
        backends = [
            "xgboost",
            "mlp",
            "lstm",
            "lightgbm",
            "cleargbm",
            "logreg",
            "random_forest",
        ]
        for backend in backends:
            body = f'{{"dataset": "taiwan", "backend": "{backend}", "n_trials": 10}}'.encode()
            response = client.post("/ml/optimize", content=body)
            assert response.status_code == 202

            enqueued = container_with_store.queue.jobs[-1]
            assert "process_optimize_job" in enqueued.func

    def test_optimize_forwards_raw_json_body(self, container_with_store: ContainerAndStore) -> None:
        """Test raw JSON body is forwarded to worker unchanged."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize",
            content=b"""{
                "dataset": "taiwan",
                "backend": "mlp",
                "n_trials": 50,
                "timeout_seconds": 3600,
                "precision": "fp16",
                "optimizer": "adam",
                "n_epochs": 100
            }""",
        )

        assert response.status_code == 202

        # Verify raw body is forwarded (worker parses backend-specific fields)
        enqueued = container_with_store.queue.jobs[-1]
        config_str = str(enqueued.args[0])
        assert "taiwan" in config_str
        assert "mlp" in config_str
        assert "fp16" in config_str
        assert "3600" in config_str


class TestOptimizeRegressionEndpoint:
    """Tests for POST /ml/optimize-regression."""

    def test_optimize_regression_enqueues_job(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test regression optimization job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize-regression",
            content=b"""{
                "dataset": "financial_distress",
                "n_trials": 50
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        assert require_str(data, "job_id") == "test-job-id"

        enqueued = container_with_store.queue.jobs[-1]
        assert "process_regression_optimize_job" in enqueued.func

    def test_optimize_regression_with_all_options(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Test regression optimization with all common options."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "lightgbm_reg",
                "n_trials": 100,
                "timeout_seconds": 3600,
                "device": "cuda",
                "feature_preset": "log_only",
                "random_state": 123
            }""",
        )

        assert response.status_code == 202

        enqueued = container_with_store.queue.jobs[-1]
        config_str = str(enqueued.args[0])
        assert "financial_distress" in config_str
        assert "lightgbm_reg" in config_str
        assert "cuda" in config_str

    def test_optimize_regression_invalid_dataset_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid regression dataset triggers error."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize-regression",
            content=b"""{
                "dataset": "invalid",
                "n_trials": 50
            }""",
        )
        assert response.status_code == 400

    def test_optimize_regression_missing_n_trials_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Missing n_trials triggers JSONTypeError."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize-regression",
            content=b"""{
                "dataset": "financial_distress"
            }""",
        )
        assert response.status_code == 400

    def test_optimize_regression_invalid_backend_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid regressor backend triggers error."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "xgboost",
                "n_trials": 50
            }""",
        )
        assert response.status_code == 400

    def test_optimize_regression_forwards_raw_json_body(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Raw JSON body is forwarded to regression worker."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/optimize-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "n_trials": 50,
                "early_stopping_rounds": 20,
                "n_jobs": 4
            }""",
        )

        assert response.status_code == 202

        enqueued = container_with_store.queue.jobs[-1]
        config_str = str(enqueued.args[0])
        assert "financial_distress" in config_str
        assert "xgboost_reg" in config_str


class TestExplainEndpoint:
    """Tests for POST /ml/explain."""

    def test_explain_enqueues_job(self, container_with_store: ContainerAndStore) -> None:
        """Test explanation job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"""{
                "dataset": "taiwan",
                "backend": "xgboost",
                "model_path": "/models/xgboost.ubj",
                "explainer": "permutation"
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        assert require_str(data, "job_id") == "test-job-id"

        # Verify job was enqueued with correct function
        enqueued = container_with_store.queue.jobs[-1]
        assert "process_explain_job" in enqueued.func

    def test_explain_with_all_options(self, container_with_store: ContainerAndStore) -> None:
        """Test explanation with all options specified."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"""{
                "dataset": "us",
                "backend": "mlp",
                "model_path": "/models/mlp.pt",
                "explainer": "gradient",
                "target_class": 0,
                "n_samples": 500,
                "random_state": 123
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"

        # Verify job payload contains all options
        enqueued = container_with_store.queue.jobs[-1]
        config_str = str(enqueued.args[0])
        assert "us" in config_str
        assert "mlp" in config_str
        assert "gradient" in config_str

    def test_explain_invalid_dataset_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid dataset triggers edge validation and results in error."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"""{
                "dataset": "invalid",
                "backend": "xgboost",
                "model_path": "/models/model.ubj",
                "explainer": "permutation"
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_explain_invalid_backend_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid backend triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"""{
                "dataset": "taiwan",
                "backend": "invalid",
                "model_path": "/models/model.ubj",
                "explainer": "permutation"
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_explain_invalid_explainer_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid explainer triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"""{
                "dataset": "taiwan",
                "backend": "xgboost",
                "model_path": "/models/model.ubj",
                "explainer": "invalid"
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_explain_missing_required_field_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Missing required field triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"""{
                "dataset": "taiwan",
                "backend": "xgboost",
                "model_path": "/models/model.ubj"
            }""",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400

    def test_explain_non_object_json_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Non-object JSON (e.g., list) triggers TypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain",
            content=b"[]",
        )
        # AppError is unhandled in these route tests, so FastAPI returns 400
        assert response.status_code == 400


# =============================================================================
# Predict Regression Endpoint
# =============================================================================


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


class TestPredictRegressionEndpoint:
    """Tests for POST /ml/predict-regression."""

    def setup_method(self) -> None:
        """Save original regression workers_hooks."""
        self._orig_regressor_registry = regression_hooks.regressor_registry_factory

    def teardown_method(self) -> None:
        """Restore original regression workers_hooks."""
        regression_hooks.regressor_registry_factory = self._orig_regressor_registry

    def test_predict_regression_returns_predictions(
        self, container_with_store: ContainerAndStore, tmp_path: Path
    ) -> None:
        """Predict-regression returns predicted continuous values."""
        # Must live under the container's models_root: the route confines
        # caller-supplied model_path before loading.
        models_root = tmp_path / "models"
        models_root.mkdir(parents=True, exist_ok=True)
        model_path = models_root / "model.ubj"
        _create_and_save_xgb_regressor(model_path)

        client = _create_test_client(container_with_store)
        body = (
            '{"backend": "xgboost_reg",'
            f' "model_path": "{model_path.as_posix()}",'
            ' "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}'
        )
        response = client.post(
            "/ml/predict-regression",
            content=body.encode(),
        )

        assert response.status_code == 200
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "backend") == "xgboost_reg"
        assert require_int(data, "n_samples") == 2
        preds = require_list(data, "predictions")
        assert len(preds) == 2
        assert all(require_float({"v": v}, "v") > -1e10 for v in preds)

    def test_predict_regression_rejects_model_path_outside_models_root(
        self, container_with_store: ContainerAndStore, tmp_path: Path
    ) -> None:
        """A model_path escaping models_root returns 400 and loads nothing.

        This route loads in the API process, so an unconstrained path would
        open an arbitrary host file inside the web worker.

        Args:
            container_with_store: Fixture providing the wired container.
            tmp_path: Pytest temporary directory unique to this test.
        """
        outside = tmp_path / "outside.ubj"
        _create_and_save_xgb_regressor(outside)

        client = _create_test_client(container_with_store)
        body = (
            '{"backend": "xgboost_reg",'
            f' "model_path": "{outside.as_posix()}",'
            ' "features": [[1.0, 2.0, 3.0]]}'
        )
        response = client.post("/ml/predict-regression", content=body.encode())

        assert response.status_code == 400
        assert "must resolve inside the models root" in response.text

    def test_predict_regression_invalid_backend_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid backend triggers ValueError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/predict-regression",
            content=b'{"backend": "invalid", "model_path": "/tmp/m.ubj", "features": [[1.0]]}',
        )
        assert response.status_code == 400

    def test_predict_regression_missing_features_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Missing features field triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/predict-regression",
            content=b'{"backend": "xgboost_reg", "model_path": "/tmp/m.ubj"}',
        )
        assert response.status_code == 400

    def test_predict_regression_empty_features_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Empty features list triggers JSONTypeError in decoder."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/predict-regression",
            content=b'{"backend": "xgboost_reg", "model_path": "/tmp/m.ubj", "features": []}',
        )
        assert response.status_code == 400

    def test_predict_regression_non_object_json_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Non-object JSON triggers JSONTypeError."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/predict-regression",
            content=b'"just a string"',
        )
        assert response.status_code == 400


# =============================================================================
# Explain Regression Endpoint
# =============================================================================


class TestExplainRegressionEndpoint:
    """Tests for POST /ml/explain-regression."""

    def test_explain_regression_enqueues_job(self, container_with_store: ContainerAndStore) -> None:
        """Regression explanation job is enqueued."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": "/models/xgb_reg.ubj",
                "explainer": "permutation"
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"
        assert require_str(data, "job_id") == "test-job-id"

        enqueued = container_with_store.queue.jobs[-1]
        assert "process_regression_explain_job" in enqueued.func

    def test_explain_regression_with_all_options(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Regression explanation with all options specified."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "lightgbm_reg",
                "model_path": "/models/lgbm_reg.txt",
                "explainer": "shap_tree",
                "n_samples": 500,
                "random_state": 123
            }""",
        )

        assert response.status_code == 202
        data = narrow_json_to_dict(load_json_str(response.text))
        assert require_str(data, "status") == "queued"

        enqueued = container_with_store.queue.jobs[-1]
        config_str = str(enqueued.args[0])
        assert "financial_distress" in config_str
        assert "lightgbm_reg" in config_str
        assert "shap_tree" in config_str

    def test_explain_regression_invalid_backend_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid backend triggers error at API edge."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "invalid",
                "model_path": "/m",
                "explainer": "permutation"
            }""",
        )
        assert response.status_code == 400

    def test_explain_regression_invalid_explainer_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Invalid explainer triggers JSONTypeError at API edge."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": "/m",
                "explainer": "invalid"
            }""",
        )
        assert response.status_code == 400

    def test_explain_regression_missing_explainer_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Missing explainer triggers JSONTypeError at API edge."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain-regression",
            content=b"""{
                "dataset": "financial_distress",
                "backend": "xgboost_reg",
                "model_path": "/m"
            }""",
        )
        assert response.status_code == 400

    def test_explain_regression_non_object_json_returns_400(
        self, container_with_store: ContainerAndStore
    ) -> None:
        """Non-object JSON triggers error at API edge."""
        client = _create_test_client(container_with_store)
        response = client.post(
            "/ml/explain-regression",
            content=b"[]",
        )
        assert response.status_code == 400
