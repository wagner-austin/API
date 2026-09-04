"""Integration tests for training job with real XGBoost training."""

from __future__ import annotations

from pathlib import Path

from covenant_domain import (
    Deal,
    DealId,
    Measurement,
)
from covenant_persistence import (
    ConnectionProtocol,
)
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from platform_core.json_utils import (
    dump_json_str,
)

from covenant_radar_api.core import _test_hooks
from covenant_radar_api.worker.train_job import process_train_job, run_training
from tests._train_job_fixtures import (
    _add_covenant_results_for_deal,
    _add_deal,
    _add_measurements_for_deal,
    _TrainingProvider,
)


class TestProcessTrainJob:
    """Tests for process_train_job RQ entry point."""

    def test_process_train_job_loads_container_and_runs(self, tmp_path: Path) -> None:
        """Test that process_train_job loads container from env and runs training."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv
        from platform_workers.redis import RedisStrProto
        from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient
        from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

        store = InMemoryStore()

        # Add training data - need at least 10 samples for train/val/test split
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        # Create FakeEnv with test environment values
        fake_env = FakeEnv(
            {
                "REDIS_URL": "redis://test:6379/0",
                "DATABASE_URL": "postgresql://test@localhost/test",
                # APP__MODELS_ROOT is the name the loader reads --
                # `_parse_str("APP__MODELS_ROOT", "/data/models")`. This said
                # MODEL_OUTPUT_DIR, which nothing reads, so the redirect
                # never happened and `process_train_job` fell through to the
                # "/data/models" default. On Windows that resolves under the
                # current drive and is writable, so the test passed; on Linux
                # it is a directory at the filesystem root and
                # `mkdir(parents=True)` raised PermissionError.
                "APP__MODELS_ROOT": str(tmp_path),
                "APP__DATA_ROOT": str(tmp_path),
            }
        )

        # Override config hooks to use fake env
        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        # Override container hooks to use fakes
        fake_kv: FakeRedis = FakeRedis()
        fake_kv.sadd("rq:workers", "worker-1")
        fake_rq: FakeRedisBytesClient = FakeRedisBytesClient()
        fake_queue: FakeQueue = FakeQueue()

        def kv_factory(url: str) -> RedisStrProto:
            return fake_kv

        def connection_factory(dsn: str) -> ConnectionProtocol:
            return InMemoryConnection(store)

        def rq_client_factory(url: str) -> _RedisBytesClient:
            return fake_rq

        def queue_factory(name: str, connection: _RedisBytesClient) -> RQClientQueue:
            return fake_queue

        orig_kv = _test_hooks.kv_factory
        orig_conn = _test_hooks.connection_factory
        orig_rq = _test_hooks.rq_client_factory
        orig_queue = _test_hooks.queue_factory

        _test_hooks.kv_factory = kv_factory
        _test_hooks.connection_factory = connection_factory
        _test_hooks.rq_client_factory = rq_client_factory
        _test_hooks.queue_factory = queue_factory

        try:
            config_json = dump_json_str(
                {
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "n_estimators": 10,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                }
            )

            result = process_train_job(config_json)

            assert result["status"] == "complete"
            assert result["samples_total"] == 12

            # Verify model file was created in the temp directory
            model_path = Path(str(result["model_path"]))
            assert model_path.exists()
            assert model_path.suffix == ".ubj"
            # ...and that it is THIS test's temp directory. Without this the
            # test passes wherever the default happens to be writable, which
            # is how a redirect that never took effect went unnoticed.
            assert tmp_path in model_path.parents

            # Verify FakeRedis was only called with expected methods
            # sadd was called during setup, close is called during container.close()
            fake_kv.assert_only_called({"sadd", "close"})
        finally:
            # Restore all hooks
            config_hooks.get_env = orig_get_env
            _test_hooks.kv_factory = orig_kv
            _test_hooks.connection_factory = orig_conn
            _test_hooks.rq_client_factory = orig_rq
            _test_hooks.queue_factory = orig_queue


class TestModelLearning:
    """Integration tests that validate the model actually learns from training data."""

    def test_trained_model_predicts_breach_higher_for_risky_deals(self, tmp_path: Path) -> None:
        """Test that trained model predicts higher breach probability for risky deals.

        This validates the model actually learned from the training data by:
        1. Training on deals with clear patterns (high debt = breach, low debt = no breach)
        2. Loading the trained model
        3. Making predictions on new data with similar patterns
        4. Verifying risky deals have higher breach probability than safe deals
        """
        from covenant_domain.features import extract_features
        from covenant_ml.predictor import load_model, predict_probabilities

        store = InMemoryStore()

        # Create training data with clear patterns:
        # - Breach deals: high debt, low EBITDA (debt/EBITDA > 4)
        # - Safe deals: low debt, high EBITDA (debt/EBITDA < 2)

        # Required metrics for extract_features
        required_metrics = {
            "total_debt": 0,
            "ebitda": 0,
            "interest_expense": 1_000_000,
            "current_assets": 8_000_000,
            "current_liabilities": 5_000_000,
        }

        # Safe deals (no breach) - low debt ratios
        # Need 15+ samples per class to have enough after train/val/test split
        for i in range(15):
            deal_id = f"safe-{i}"
            _add_deal(store, deal_id, "Technology", "North America")
            # Low debt (2M), high EBITDA (5M) => ratio = 0.4
            safe_metrics = {
                **required_metrics,
                "total_debt": 2_000_000,
                "ebitda": 5_000_000,
            }
            for metric_name, metric_value in safe_metrics.items():
                store.measurements.append(
                    Measurement(
                        deal_id=DealId(value=deal_id),
                        period_start_iso="2024-01-01",
                        period_end_iso="2024-03-31",
                        metric_name=metric_name,
                        metric_value_scaled=metric_value,
                    )
                )
            _add_covenant_results_for_deal(store, deal_id, f"cov-safe-{i}", has_breach=False)

        # Risky deals (breach) - high debt ratios
        for i in range(15):
            deal_id = f"risky-{i}"
            _add_deal(store, deal_id, "Finance", "Europe")
            # High debt (25M), low EBITDA (5M) => ratio = 5.0
            risky_metrics = {
                **required_metrics,
                "total_debt": 25_000_000,
                "ebitda": 5_000_000,
            }
            for metric_name, metric_value in risky_metrics.items():
                store.measurements.append(
                    Measurement(
                        deal_id=DealId(value=deal_id),
                        period_start_iso="2024-01-01",
                        period_end_iso="2024-03-31",
                        metric_name=metric_name,
                        metric_value_scaled=metric_value,
                    )
                )
            _add_covenant_results_for_deal(store, deal_id, f"cov-risky-{i}", has_breach=True)

        # Train the model
        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 50,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
            }
        )

        result = run_training(config_json, provider)
        assert result["status"] == "complete"
        assert result["samples_total"] == 30

        # Load the trained model
        model_path = str(result["model_path"])
        model = load_model(model_path)

        # Create test features for a "safe" deal (low debt ratio)
        safe_deal = Deal(
            id=DealId(value="test-safe"),
            name="Safe Test Deal",
            borrower="Safe Corp",
            sector="Technology",
            region="North America",
            commitment_amount_cents=100_000_000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        safe_features = extract_features(
            deal=safe_deal,
            metrics_current={
                "total_debt": 2_000_000,
                "ebitda": 5_000_000,
                "interest_expense": 1_000_000,
                "current_assets": 8_000_000,
                "current_liabilities": 5_000_000,
            },
            metrics_1p_ago={},
            metrics_4p_ago={},
            recent_results=[],
            sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
            region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        )

        # Create test features for a "risky" deal (high debt ratio)
        risky_deal = Deal(
            id=DealId(value="test-risky"),
            name="Risky Test Deal",
            borrower="Risky Corp",
            sector="Finance",
            region="Europe",
            commitment_amount_cents=100_000_000,
            currency="USD",
            maturity_date_iso="2025-12-31",
        )
        risky_features = extract_features(
            deal=risky_deal,
            metrics_current={
                "total_debt": 25_000_000,
                "ebitda": 5_000_000,
                "interest_expense": 1_000_000,
                "current_assets": 8_000_000,
                "current_liabilities": 5_000_000,
            },
            metrics_1p_ago={},
            metrics_4p_ago={},
            recent_results=[],
            sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
            region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        )

        # Get predictions
        predictions = predict_probabilities(model, [safe_features, risky_features])

        safe_breach_prob = predictions[0]
        risky_breach_prob = predictions[1]

        # The model should predict higher breach probability for risky deal
        assert risky_breach_prob > safe_breach_prob, (
            f"Model did not learn: risky={risky_breach_prob:.3f} should be > "
            f"safe={safe_breach_prob:.3f}"
        )

        # Additional check: probabilities should be meaningfully different
        # (not just random noise)
        difference = risky_breach_prob - safe_breach_prob
        assert difference > 0.1, (
            f"Model predictions too similar: difference={difference:.3f} "
            f"(risky={risky_breach_prob:.3f}, safe={safe_breach_prob:.3f})"
        )


class TestUploadModelToDataBank:
    """Tests for _upload_model_to_data_bank function."""

    def test_success(self, tmp_path: Path) -> None:
        """Test successful model upload to data-bank via hook."""
        from covenant_radar_api.worker import _test_hooks as worker_hooks
        from covenant_radar_api.worker.train_job import _upload_model_to_data_bank

        model_path = tmp_path / "test_model.ubj"
        model_path.write_bytes(b"fake model bytes")

        upload_calls: list[tuple[Path, str, str]] = []

        class FakeUploader:
            """Fake uploader implementing DataBankUploaderProtocol."""

            def __call__(
                self,
                model_path: Path,
                data_bank_url: str,
                data_bank_key: str,
            ) -> str:
                upload_calls.append((model_path, data_bank_url, data_bank_key))
                return model_path.name

        orig_uploader = worker_hooks.data_bank_uploader
        worker_hooks.data_bank_uploader = FakeUploader()

        try:
            result = _upload_model_to_data_bank(
                model_path,
                "https://data-bank.example.com",
                "test-api-key",
            )

            assert result == "test_model.ubj"
            assert len(upload_calls) == 1
            assert upload_calls[0][0] == model_path
            assert upload_calls[0][1] == "https://data-bank.example.com"
            assert upload_calls[0][2] == "test-api-key"
        finally:
            worker_hooks.data_bank_uploader = orig_uploader


class TestProcessTrainJobWithDataBank:
    """Tests for process_train_job with data-bank integration."""

    def test_uploads_model_when_configured(self, tmp_path: Path) -> None:
        """Test that process job uploads model when data-bank is configured."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv
        from platform_workers.redis import RedisStrProto
        from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient
        from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

        from covenant_radar_api.worker import _test_hooks as worker_hooks

        store = InMemoryStore()

        # Add training data - need at least 10 samples
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        upload_calls: list[str] = []

        class FakeUploader:
            """Fake uploader that tracks calls."""

            def __call__(
                self,
                model_path: Path,
                data_bank_url: str,
                data_bank_key: str,
            ) -> str:
                upload_calls.append(model_path.name)
                return model_path.name

        orig_uploader = worker_hooks.data_bank_uploader
        worker_hooks.data_bank_uploader = FakeUploader()

        # Create FakeEnv with data-bank configured
        fake_env = FakeEnv(
            {
                "REDIS_URL": "redis://test:6379/0",
                "DATABASE_URL": "postgresql://test@localhost/test",
                # APP__MODELS_ROOT is the name the loader reads --
                # `_parse_str("APP__MODELS_ROOT", "/data/models")`. This said
                # MODEL_OUTPUT_DIR, which nothing reads, so the redirect
                # never happened and `process_train_job` fell through to the
                # "/data/models" default. On Windows that resolves under the
                # current drive and is writable, so the test passed; on Linux
                # it is a directory at the filesystem root and
                # `mkdir(parents=True)` raised PermissionError.
                "APP__MODELS_ROOT": str(tmp_path),
                "APP__DATA_ROOT": str(tmp_path),
                "DATA_BANK_API_URL": "https://data-bank.example.com",
                "DATA_BANK_API_KEY": "test-key",
            }
        )

        # Override config hooks to use fake env
        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        # Override container hooks to use fakes
        fake_kv: FakeRedis = FakeRedis()
        fake_kv.sadd("rq:workers", "worker-1")
        fake_rq: FakeRedisBytesClient = FakeRedisBytesClient()
        fake_queue: FakeQueue = FakeQueue()

        def kv_factory(url: str) -> RedisStrProto:
            return fake_kv

        def connection_factory(dsn: str) -> ConnectionProtocol:
            return InMemoryConnection(store)

        def rq_client_factory(url: str) -> _RedisBytesClient:
            return fake_rq

        def queue_factory(name: str, connection: _RedisBytesClient) -> RQClientQueue:
            return fake_queue

        orig_kv = _test_hooks.kv_factory
        orig_conn = _test_hooks.connection_factory
        orig_rq = _test_hooks.rq_client_factory
        orig_queue = _test_hooks.queue_factory

        _test_hooks.kv_factory = kv_factory
        _test_hooks.connection_factory = connection_factory
        _test_hooks.rq_client_factory = rq_client_factory
        _test_hooks.queue_factory = queue_factory

        try:
            config_json = dump_json_str(
                {
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "n_estimators": 10,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                }
            )

            result = process_train_job(config_json)

            assert result["status"] == "complete"
            assert result["samples_total"] == 12
            assert "model_file_id" in result
            assert result["model_file_id"] == "active_xgb.ubj"
            assert len(upload_calls) == 1
            assert upload_calls[0] == "active_xgb.ubj"
        finally:
            # Restore all hooks
            config_hooks.get_env = orig_get_env
            _test_hooks.kv_factory = orig_kv
            _test_hooks.connection_factory = orig_conn
            _test_hooks.rq_client_factory = orig_rq
            _test_hooks.queue_factory = orig_queue
            worker_hooks.data_bank_uploader = orig_uploader
