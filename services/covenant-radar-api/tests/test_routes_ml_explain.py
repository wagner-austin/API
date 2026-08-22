"""Integration tests for ML routes with real implementations."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from covenant_radar_api.worker import _regression_hooks as regression_hooks
from tests._routes_ml_fixtures import (
    _create_and_save_xgb_regressor,
    _create_test_client,
)

from .conftest import ContainerAndStore


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
