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

from covenant_radar_api.worker.train_job import run_training


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

        # Add multiple deals with varying outcomes
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        _add_deal(store, "d2", "Finance", "Europe")
        _add_measurements_for_deal(store, "d2")
        _add_covenant_results_for_deal(store, "d2", "c2", has_breach=True)

        _add_deal(store, "d3", "Healthcare", "Asia")
        _add_measurements_for_deal(store, "d3")
        _add_covenant_results_for_deal(store, "d3", "c3", has_breach=False)

        _add_deal(store, "d4", "Technology", "Europe")
        _add_measurements_for_deal(store, "d4")
        _add_covenant_results_for_deal(store, "d4", "c4", has_breach=True)

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

        assert result["status"] == "complete"
        assert result["samples_trained"] == 4

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

    def test_train_skips_deals_without_measurements(self, tmp_path: Path) -> None:
        """Test training skips deals that have no measurements."""
        store = InMemoryStore()

        # Deal with measurements
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        # Deal without measurements - should be skipped
        _add_deal(store, "d2", "Finance", "Europe")
        # No measurements for d2

        # Another deal with measurements
        _add_deal(store, "d3", "Healthcare", "Asia")
        _add_measurements_for_deal(store, "d3")
        _add_covenant_results_for_deal(store, "d3", "c3", has_breach=True)

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

        # Only 2 deals should be trained (d1 and d3), d2 skipped
        assert result["samples_trained"] == 2

    def test_train_model_file_has_unique_name(self, tmp_path: Path) -> None:
        """Test that each training run produces a uniquely named model file."""
        store = InMemoryStore()
        _add_deal(store, "d1", "Technology", "North America")
        _add_measurements_for_deal(store, "d1")
        _add_covenant_results_for_deal(store, "d1", "c1", has_breach=False)

        _add_deal(store, "d2", "Finance", "Europe")
        _add_measurements_for_deal(store, "d2")
        _add_covenant_results_for_deal(store, "d2", "c2", has_breach=True)

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
