"""Integration tests for ML routes with real implementations."""

from __future__ import annotations

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_str,
)

from tests._routes_ml_fixtures import (
    _create_test_client,
)

from .conftest import ContainerAndStore


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
