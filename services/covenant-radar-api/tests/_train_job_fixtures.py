"""Shared fixtures and helpers for test_train_job splits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
