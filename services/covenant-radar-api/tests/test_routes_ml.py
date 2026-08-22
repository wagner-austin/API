"""Integration tests for ML routes with real implementations."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_float,
    require_str,
)

from tests._routes_ml_fixtures import (
    _add_test_deal,
    _add_test_measurements,
    _create_and_save_model,
    _create_test_client,
)

from .conftest import ContainerAndStore


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
