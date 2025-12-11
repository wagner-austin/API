"""Integration tests for training job with real XGBoost training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)
from covenant_persistence import (
    ConnectionProtocol,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
    PostgresCovenantResultRepository,
    PostgresDealRepository,
    PostgresMeasurementRepository,
)
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    dump_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)

from covenant_radar_api.core import _test_hooks
from covenant_radar_api.worker.train_job import _parse_train_config, process_train_job, run_training


class _TrainingProvider:
    """Training data provider using InMemoryConnection."""

    def __init__(self, store: InMemoryStore, output_dir: Path) -> None:
        self._conn = InMemoryConnection(store)
        self._output_dir = output_dir
        self._sector_encoder: dict[str, int] = {"Technology": 0, "Finance": 1, "Healthcare": 2}
        self._region_encoder: dict[str, int] = {"North America": 0, "Europe": 1, "Asia": 2}

    def deal_repo(self) -> DealRepository:
        """Return deal repository."""
        repo: DealRepository = PostgresDealRepository(self._conn)
        return repo

    def measurement_repo(self) -> MeasurementRepository:
        """Return measurement repository."""
        repo: MeasurementRepository = PostgresMeasurementRepository(self._conn)
        return repo

    def covenant_result_repo(self) -> CovenantResultRepository:
        """Return result repository."""
        repo: CovenantResultRepository = PostgresCovenantResultRepository(self._conn)
        return repo

    def get_sector_encoder(self) -> dict[str, int]:
        """Return sector encoder."""
        return self._sector_encoder

    def get_region_encoder(self) -> dict[str, int]:
        """Return region encoder."""
        return self._region_encoder

    def get_model_output_dir(self) -> Path:
        """Return model output directory."""
        return self._output_dir


def _add_deal(store: InMemoryStore, deal_id: str, sector: str, region: str) -> None:
    """Add a deal to store."""
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
    store._deal_order.append(deal_id)


def _add_measurements_for_deal(store: InMemoryStore, deal_id: str) -> None:
    """Add measurements for multiple periods for a deal."""
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


def _add_covenant_results_for_deal(
    store: InMemoryStore, deal_id: str, cov_id: str, has_breach: bool
) -> None:
    """Add covenant and results for a deal."""
    store.covenants[cov_id] = Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name="Test Covenant",
        formula="debt / ebitda",
        threshold_value_scaled=4_000_000,
        threshold_direction="<=",
        frequency="QUARTERLY",
    )
    store._covenant_order.append(cov_id)

    status: Literal["OK", "NEAR_BREACH", "BREACH"] = "BREACH" if has_breach else "OK"
    store.covenant_results.append(
        CovenantResult(
            covenant_id=CovenantId(value=cov_id),
            period_start_iso="2024-01-01",
            period_end_iso="2024-03-31",
            calculated_value_scaled=2_000_000,
            status=status,
        )
    )


class TestRunTraining:
    """Tests for run_training job function with real XGBoost training."""

    def test_train_with_valid_data(self, tmp_path: Path) -> None:
        """Test training with valid training data produces a model."""
        store = InMemoryStore()

        # Add multiple deals with varying outcomes (minimum 10 samples required)
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            # Alternate breach outcomes for class balance
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "scale_pos_weight": 1.2,
            }
        )

        result = run_training(config_json, provider)

        assert result["status"] == "complete"
        assert result["samples_total"] == 12

        # Verify model file was created and has valid path
        model_path = Path(str(result["model_path"]))
        assert model_path.exists()
        assert model_path.suffix == ".ubj"

        # Verify model_id is a valid UUID
        model_id = require_str(result, "model_id")
        import uuid

        uuid.UUID(model_id)  # Raises ValueError if invalid

        # Verify config is returned with correct values
        config = narrow_json_to_dict(result["config"])
        assert require_float(config, "learning_rate") == 0.1
        assert require_int(config, "max_depth") == 3
        assert require_float(config, "reg_alpha") == 0.0
        assert require_float(config, "reg_lambda") == 1.0
        assert require_float(config, "scale_pos_weight") == 1.2

    def test_train_with_no_data_raises(self, tmp_path: Path) -> None:
        """Test training with no data raises ValueError."""
        store = InMemoryStore()
        # Empty store - no deals

        provider = _TrainingProvider(store, tmp_path)
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

        with pytest.raises(ValueError, match="No training data"):
            run_training(config_json, provider)

    def test_train_with_insufficient_data_raises(self, tmp_path: Path) -> None:
        """Test training with insufficient data (1-9 samples) raises ValueError."""
        store = InMemoryStore()

        # Add only 5 deals (less than minimum 10)
        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        for i in range(5):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        provider = _TrainingProvider(store, tmp_path)
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

        with pytest.raises(ValueError, match=r"Insufficient training data: 5 samples"):
            run_training(config_json, provider)

    def test_train_with_invalid_config_raises(self, tmp_path: Path) -> None:
        """Test training with invalid config JSON raises."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = "not valid json"

        with pytest.raises(InvalidJsonError):
            run_training(config_json, provider)

    def test_train_with_missing_config_field_raises(self, tmp_path: Path) -> None:
        """Test training with missing config field raises."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str({"learning_rate": 0.1})  # Missing other fields

        with pytest.raises(JSONTypeError, match="Missing required field"):
            run_training(config_json, provider)

    def test_train_with_invalid_scale_pos_weight_raises(self, tmp_path: Path) -> None:
        """Test training raises when scale_pos_weight has invalid type."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "scale_pos_weight": "invalid",
            }
        )

        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            run_training(config_json, provider)

    def test_train_with_invalid_early_stopping_type_raises(self, tmp_path: Path) -> None:
        """Test training raises when early_stopping_rounds is not numeric."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "early_stopping_rounds": "ten",
            }
        )

        with pytest.raises(JSONTypeError, match="Field 'early_stopping_rounds' must be a number"):
            run_training(config_json, provider)

    def test_train_with_config_not_object_raises(self, tmp_path: Path) -> None:
        """Test training with config that is not a JSON object raises."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = "[1, 2, 3]"  # JSON array, not object

        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            run_training(config_json, provider)

    def test_train_with_invalid_ratio_type_raises(self, tmp_path: Path) -> None:
        """Test training raises when optional ratio is not numeric."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "train_ratio": "heavy",
            }
        )

        with pytest.raises(JSONTypeError, match="Field 'train_ratio' must be a number"):
            run_training(config_json, provider)

    def test_train_with_invalid_device_raises(self, tmp_path: Path) -> None:
        """Test training raises when device is unsupported."""
        store = InMemoryStore()
        provider = _TrainingProvider(store, tmp_path)

        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "device": "tpu",
            }
        )

        with pytest.raises(ValueError, match="device must be one of: cpu, cuda, auto"):
            run_training(config_json, provider)

    def test_parse_train_config_accepts_cpu_cuda_and_auto(self) -> None:
        """Test _parse_train_config handles supported device values."""
        base = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
        }
        config_cpu = _parse_train_config(dump_json_str({**base, "device": "cpu"}))
        assert config_cpu["device"] == "cpu"
        config_cuda = _parse_train_config(dump_json_str({**base, "device": "cuda"}))
        assert config_cuda["device"] == "cuda"
        config_auto = _parse_train_config(dump_json_str({**base, "device": "auto"}))
        assert config_auto["device"] == "auto"

    def test_parse_train_config_rejects_non_string_device(self) -> None:
        """Test _parse_train_config raises for non-string device types."""
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "device": 99,
            }
        )

        with pytest.raises(JSONTypeError, match="device must be a string"):
            _parse_train_config(config_json)

    def test_train_with_explicit_split_ratios(self, tmp_path: Path) -> None:
        """Test training with explicitly provided split ratios."""
        store = InMemoryStore()

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

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "early_stopping_rounds": 5,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            }
        )

        result = run_training(config_json, provider)

        assert result["status"] == "complete"
        assert result["samples_total"] == 12
        # Config should reflect provided values
        config = narrow_json_to_dict(result["config"])
        assert require_float(config, "train_ratio") == 0.8
        assert require_float(config, "val_ratio") == 0.1
        assert require_float(config, "test_ratio") == 0.1
        assert require_int(config, "early_stopping_rounds") == 5
        assert require_float(config, "reg_alpha") == 1.0
        assert require_float(config, "reg_lambda") == 5.0

    def test_train_with_float_early_stopping(self, tmp_path: Path) -> None:
        """Test training with early_stopping_rounds as float (converts to int)."""
        store = InMemoryStore()

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

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "early_stopping_rounds": 8.0,  # Float value
            }
        )

        result = run_training(config_json, provider)

        assert result["status"] == "complete"
        config = narrow_json_to_dict(result["config"])
        assert require_int(config, "early_stopping_rounds") == 8

    def test_train_with_invalid_ratios_raises(self, tmp_path: Path) -> None:
        """Test training with ratios that don't sum to 1.0 raises ValueError."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        provider = _TrainingProvider(store, tmp_path)
        config_json = dump_json_str(
            {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 42,
                "train_ratio": 0.5,
                "val_ratio": 0.3,
                "test_ratio": 0.3,  # Sum = 1.1
            }
        )

        with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
            run_training(config_json, provider)

    def test_train_skips_deals_without_measurements(self, tmp_path: Path) -> None:
        """Test training skips deals that have no measurements."""
        store = InMemoryStore()

        sectors = ["Technology", "Finance", "Healthcare"]
        regions = ["North America", "Europe", "Asia"]

        # Add 12 deals with measurements
        for i in range(12):
            deal_id = f"d{i + 1}"
            sector = sectors[i % 3]
            region = regions[i % 3]
            _add_deal(store, deal_id, sector, region)
            _add_measurements_for_deal(store, deal_id)
            has_breach = i % 2 == 0
            _add_covenant_results_for_deal(store, deal_id, f"c{i + 1}", has_breach=has_breach)

        # Add deal without measurements - should be skipped
        _add_deal(store, "d-no-data", "Finance", "Europe")
        # No measurements for d-no-data

        provider = _TrainingProvider(store, tmp_path)
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

        result = run_training(config_json, provider)

        # Only 12 deals should be trained (d-no-data skipped)
        assert result["samples_total"] == 12

    def test_train_model_file_has_unique_name(self, tmp_path: Path) -> None:
        """Test that each training run produces a uniquely named model file."""
        store = InMemoryStore()

        # Add minimum 10 deals for training
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

        provider = _TrainingProvider(store, tmp_path)
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

        result1 = run_training(config_json, provider)
        result2 = run_training(config_json, provider)

        # Model IDs should be different
        assert result1["model_id"] != result2["model_id"]

        # Model paths should be different
        assert result1["model_path"] != result2["model_path"]

        # Both model files should exist
        assert Path(str(result1["model_path"])).exists()
        assert Path(str(result2["model_path"])).exists()


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
                "MODEL_OUTPUT_DIR": str(tmp_path),
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
