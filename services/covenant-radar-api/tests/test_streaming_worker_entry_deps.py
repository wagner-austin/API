"""Tests for streaming worker entry point."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from covenant_ml.types import PredictorProtocol

from covenant_radar_api import streaming_worker_entry_hooks as _hooks
from covenant_radar_api.streaming._test_hooks_repositories import (
    FakeCovenantRepository,
    FakeCovenantResultRepository,
    FakeDealRepository,
    FakeMeasurementRepository,
)
from covenant_radar_api.streaming_worker_entry import (
    StreamingWorkerDeps,
    _create_worker,
    _load_model,
    _run_worker,
)
from covenant_radar_api.streaming_worker_entry_hooks import (
    FakeConnection,
)
from tests._worker_entry_fixtures import (
    _RecordingLogger,
)


@pytest.fixture()
def restore_hooks() -> Generator[None, None, None]:
    """Fixture that restores _hooks after test."""
    # Save original hooks
    orig_connection_factory = _hooks.connection_factory
    orig_repository_factory = _hooks.repository_factory
    orig_xgboost_loader = _hooks.xgboost_loader
    orig_logger_factory = _hooks.logger_factory

    yield

    # Restore original hooks
    _hooks.connection_factory = orig_connection_factory
    _hooks.repository_factory = orig_repository_factory
    _hooks.xgboost_loader = orig_xgboost_loader
    _hooks.logger_factory = orig_logger_factory


class TestLoadModel:
    """Tests for _load_model function."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError for non-existent path."""
        fake_path = tmp_path / "nonexistent_model.ubj"
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            _load_model(fake_path, "xgboost")

    def test_xgboost_uses_hook(
        self,
        tmp_path: Path,
        restore_hooks: None,
    ) -> None:
        """Test xgboost model type uses xgboost_loader hook."""
        from covenant_radar_api.streaming._test_hooks_model import FakePredictor

        model_file = tmp_path / "model.json"
        model_file.write_text("{}")
        fake_model = FakePredictor()

        class _FakeLoader:
            """Fake model loader that returns fake_model."""

            def __call__(self, model_path: str) -> PredictorProtocol:
                assert model_path == str(model_file)
                return fake_model

        _hooks.xgboost_loader = _FakeLoader()

        result = _load_model(model_file, "xgboost")
        assert result is fake_model

    def test_lightgbm_loads_model(self, tmp_path: Path) -> None:
        """Test lightgbm model type loads real LightGBM model."""
        import numpy as np
        from covenant_ml.backends.lightgbm.backend import _get_lightgbm_imports
        from numpy.typing import NDArray

        lgbm_ctor, _ = _get_lightgbm_imports()

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[0, 1] = 2.0
        x[1, 0] = 3.0
        x[1, 1] = 4.0
        x[2, 0] = 5.0
        x[2, 1] = 6.0
        x[3, 0] = 7.0
        x[3, 1] = 8.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        clf = lgbm_ctor(
            boosting_type="gbdt",
            num_leaves=2,
            max_depth=2,
            learning_rate=0.1,
            n_estimators=5,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_samples=1,
            random_state=42,
            n_jobs=-1,
            device="cpu",
            objective="binary",
            metric="binary_logloss",
            class_weight=None,
            verbose=-1,
        )
        clf.fit(x, y, eval_set=[(x, y)])

        model_path = tmp_path / "model.txt"
        clf.booster_.save_model(str(model_path))

        result = _load_model(model_path, "lightgbm")
        proba = result.predict_proba(x)
        assert proba.shape == (4, 2)

    def test_logreg_loads_model(self, tmp_path: Path) -> None:
        """Test logreg model type loads real LogReg model."""
        import numpy as np
        from covenant_ml.backends.logreg.backend import (
            _create_logreg_model,
            _get_joblib_imports,
        )
        from numpy.typing import NDArray

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[0, 1] = 2.0
        x[1, 0] = 3.0
        x[1, 1] = 4.0
        x[2, 0] = 5.0
        x[2, 1] = 6.0
        x[3, 0] = 7.0
        x[3, 1] = 8.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        lr = _create_logreg_model(
            penalty="l2",
            inverse_reg_strength=1.0,
            solver="lbfgs",
            max_iter=100,
            tol=1e-4,
            random_state=42,
            class_weight=None,
            l1_ratio=None,
            n_jobs=-1,
        )
        lr.fit(x, y)
        model_path = tmp_path / "model.joblib"
        dump_fn, _ = _get_joblib_imports()
        dump_fn(lr, str(model_path))

        result = _load_model(model_path, "logreg")
        proba = result.predict_proba(x)
        assert proba.shape == (4, 2)

    def test_random_forest_loads_model(self, tmp_path: Path) -> None:
        """Test random_forest model type loads real Random Forest model."""
        import numpy as np
        from covenant_ml.backends.random_forest.backend import _get_sklearn_imports
        from numpy.typing import NDArray

        rf_ctor, dump_fn, _ = _get_sklearn_imports()

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[0, 1] = 2.0
        x[1, 0] = 3.0
        x[1, 1] = 4.0
        x[2, 0] = 5.0
        x[2, 1] = 6.0
        x[3, 0] = 7.0
        x[3, 1] = 8.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        rf = rf_ctor(
            n_estimators=2,
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=-1,
            random_state=42,
            oob_score=False,
        )
        rf.fit(x, y)
        model_path = tmp_path / "model.joblib"
        dump_fn(rf, str(model_path))

        result = _load_model(model_path, "random_forest")
        proba = result.predict_proba(x)
        assert proba.shape == (4, 2)

    def test_mlp_loads_model(self, tmp_path: Path) -> None:
        """Test mlp model type loads MLP model from state dict + metadata."""
        import numpy as np
        from numpy.typing import NDArray
        from platform_core.json_utils import dump_json_str

        from covenant_radar_api.worker._model_loaders import _build_mlp_model

        # Build a real model and save its state dict
        n_features = 4
        hidden_size = 8
        model = _build_mlp_model(
            n_features=n_features,
            hidden_sizes=[hidden_size, hidden_size],
            dropout=0.0,
            device="cpu",
        )

        from platform_ml.torch_types import _import_torch

        torch = _import_torch()
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(model_path))

        # Create metadata JSON
        meta = dump_json_str(
            {
                "backend": "mlp",
                "n_features": n_features,
                "hidden_sizes": [hidden_size, hidden_size],
                "dropout": 0.0,
            }
        )
        meta_path = tmp_path / "model.meta.json"
        meta_path.write_text(meta)

        result = _load_model(model_path, "mlp")
        x_test: NDArray[np.float64] = np.zeros((2, n_features), dtype=np.float64)
        proba = result.predict_proba(x_test)
        assert proba.shape == (2, 2)


class TestCreateWorker:
    """Tests for _create_worker function."""

    def test_creates_streaming_worker(self) -> None:
        """Test _create_worker creates StreamingWorker from deps."""
        from covenant_radar_api.integrations.datadog.metrics import MetricsClient
        from covenant_radar_api.streaming._test_hooks import (
            FakeKafkaConsumer,
            FakeKafkaProducer,
        )
        from covenant_radar_api.streaming._test_hooks_model import (
            FakeMetricsSink,
            FakePredictor,
        )
        from covenant_radar_api.streaming.consumer import StreamingConsumer
        from covenant_radar_api.streaming.producer import StreamingProducer
        from covenant_radar_api.streaming.worker_events import make_default_worker_config

        fake_consumer = FakeKafkaConsumer()
        fake_producer = FakeKafkaProducer()

        deps: StreamingWorkerDeps = {
            "consumer": StreamingConsumer(fake_consumer, "test.measurements.v1"),
            "producer": StreamingProducer(
                fake_producer, "test.predictions.v1", "test.alerts.v1", "test.dlq.v1"
            ),
            "metrics": MetricsClient(FakeMetricsSink()),
            "model": FakePredictor(),
            "deal_repo": FakeDealRepository(),
            "covenant_repo": FakeCovenantRepository(),
            "measurement_repo": FakeMeasurementRepository(),
            "result_repo": FakeCovenantResultRepository(),
            "sector_encoder": {"Technology": 0},
            "region_encoder": {"North America": 0},
            "config": make_default_worker_config(),
            "db_conn": FakeConnection(),
        }

        worker = _create_worker(deps)
        assert worker.__class__.__name__ == "StreamingWorker"
        assert worker.is_running is False


class TestRunWorker:
    """Tests for _run_worker function."""

    def test_run_worker_returns_zero(self) -> None:
        """Test _run_worker runs worker and returns exit code 0."""
        import signal as sig

        from covenant_radar_api.integrations.datadog.metrics import MetricsClient
        from covenant_radar_api.streaming._test_hooks import (
            FakeKafkaConsumer,
            FakeKafkaProducer,
        )
        from covenant_radar_api.streaming._test_hooks_model import (
            FakeMetricsSink,
            FakePredictor,
        )
        from covenant_radar_api.streaming.consumer import StreamingConsumer
        from covenant_radar_api.streaming.producer import StreamingProducer
        from covenant_radar_api.streaming.worker import StreamingWorker
        from covenant_radar_api.streaming.worker_events import make_default_worker_config

        fake_consumer = FakeKafkaConsumer()
        fake_producer = FakeKafkaProducer()

        worker = StreamingWorker(
            consumer=StreamingConsumer(fake_consumer, "test.measurements.v1"),
            producer=StreamingProducer(
                fake_producer, "test.predictions.v1", "test.alerts.v1", "test.dlq.v1"
            ),
            metrics=MetricsClient(FakeMetricsSink()),
            model=FakePredictor(),
            deal_repo=FakeDealRepository(),
            covenant_repo=FakeCovenantRepository(),
            measurement_repo=FakeMeasurementRepository(),
            result_repo=FakeCovenantResultRepository(),
            sector_encoder={"Technology": 0},
            region_encoder={"North America": 0},
            config=make_default_worker_config(),
        )

        # Shut down after first poll so _run_worker returns
        fake_consumer.set_on_poll(lambda: worker.shutdown())

        # Save and restore signal handlers
        orig_sigint = sig.getsignal(sig.SIGINT)
        orig_sigterm = sig.getsignal(sig.SIGTERM)

        logger = _RecordingLogger()
        try:
            exit_code = _run_worker(worker, logger)
        finally:
            sig.signal(sig.SIGINT, orig_sigint)
            sig.signal(sig.SIGTERM, orig_sigterm)

        assert exit_code == 0
        # Verify logging happened
        assert len(logger.info_messages) >= 2
        assert "Starting" in logger.info_messages[0][0]
        assert "stopped" in logger.info_messages[1][0].lower()
