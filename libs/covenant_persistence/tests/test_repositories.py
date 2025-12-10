"""Tests for repositories module."""

from __future__ import annotations

from collections.abc import Sequence

from covenant_domain.models import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)

from covenant_persistence.repositories import (
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)


class TestDealRepositoryProtocol:
    """Tests that verify DealRepository protocol can be implemented."""

    def test_deal_repository_implementation(self) -> None:
        """A class implementing DealRepository methods satisfies the protocol."""
        deals: dict[str, Deal] = {}

        class TestDealRepo:
            def create(self, deal: Deal) -> None:
                deals[deal["id"]["value"]] = deal

            def get(self, deal_id: DealId) -> Deal:
                if deal_id["value"] not in deals:
                    raise KeyError(f"Deal not found: {deal_id['value']}")
                return deals[deal_id["value"]]

            def list_all(self) -> Sequence[Deal]:
                return list(deals.values())

            def update(self, deal: Deal) -> None:
                if deal["id"]["value"] not in deals:
                    raise KeyError(f"Deal not found: {deal['id']['value']}")
                deals[deal["id"]["value"]] = deal

            def delete(self, deal_id: DealId) -> None:
                if deal_id["value"] not in deals:
                    raise KeyError(f"Deal not found: {deal_id['value']}")
                del deals[deal_id["value"]]

        repo: DealRepository = TestDealRepo()

        deal: Deal = {
            "id": {"value": "deal-1"},
            "name": "Test Deal",
            "borrower": "Acme",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000000,
            "currency": "USD",
            "maturity_date_iso": "2025-12-31",
        }

        repo.create(deal)
        retrieved = repo.get({"value": "deal-1"})
        assert retrieved["name"] == "Test Deal"

        all_deals = repo.list_all()
        assert len(all_deals) == 1

        updated_deal: Deal = {
            "id": {"value": "deal-1"},
            "name": "Updated Deal",
            "borrower": "Acme",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 2000000,
            "currency": "USD",
            "maturity_date_iso": "2025-12-31",
        }
        repo.update(updated_deal)
        retrieved = repo.get({"value": "deal-1"})
        assert retrieved["name"] == "Updated Deal"

        repo.delete({"value": "deal-1"})
        assert len(repo.list_all()) == 0


class TestCovenantRepositoryProtocol:
    """Tests that verify CovenantRepository protocol can be implemented."""

    def test_covenant_repository_implementation(self) -> None:
        """A class implementing CovenantRepository methods satisfies the protocol."""
        covenants: dict[str, Covenant] = {}

        class TestCovenantRepo:
            def create(self, covenant: Covenant) -> None:
                covenants[covenant["id"]["value"]] = covenant

            def get(self, covenant_id: CovenantId) -> Covenant:
                if covenant_id["value"] not in covenants:
                    raise KeyError(f"Covenant not found: {covenant_id['value']}")
                return covenants[covenant_id["value"]]

            def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
                return [c for c in covenants.values() if c["deal_id"]["value"] == deal_id["value"]]

            def delete(self, covenant_id: CovenantId) -> None:
                if covenant_id["value"] not in covenants:
                    raise KeyError(f"Covenant not found: {covenant_id['value']}")
                del covenants[covenant_id["value"]]

        repo: CovenantRepository = TestCovenantRepo()

        covenant: Covenant = {
            "id": {"value": "cov-1"},
            "deal_id": {"value": "deal-1"},
            "name": "Test Covenant",
            "formula": "a / b",
            "threshold_value_scaled": 3500000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY",
        }

        repo.create(covenant)
        retrieved = repo.get({"value": "cov-1"})
        assert retrieved["name"] == "Test Covenant"

        deal_covenants = repo.list_for_deal({"value": "deal-1"})
        assert len(deal_covenants) == 1

        repo.delete({"value": "cov-1"})
        assert len(repo.list_for_deal({"value": "deal-1"})) == 0


class TestMeasurementRepositoryProtocol:
    """Tests that verify MeasurementRepository protocol can be implemented."""

    def test_measurement_repository_implementation(self) -> None:
        """A class implementing MeasurementRepository methods satisfies the protocol."""
        measurements: list[Measurement] = []

        class TestMeasurementRepo:
            def add_many(self, new_measurements: Sequence[Measurement]) -> int:
                measurements.extend(new_measurements)
                return len(new_measurements)

            def list_for_deal_and_period(
                self,
                deal_id: DealId,
                period_start_iso: str,
                period_end_iso: str,
            ) -> Sequence[Measurement]:
                return [
                    m
                    for m in measurements
                    if m["deal_id"]["value"] == deal_id["value"]
                    and m["period_start_iso"] == period_start_iso
                    and m["period_end_iso"] == period_end_iso
                ]

            def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
                return [m for m in measurements if m["deal_id"]["value"] == deal_id["value"]]

        repo: MeasurementRepository = TestMeasurementRepo()

        new_measurements: list[Measurement] = [
            {
                "deal_id": {"value": "deal-1"},
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "metric_name": "total_debt",
                "metric_value_scaled": 100000000,
            },
            {
                "deal_id": {"value": "deal-1"},
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "metric_name": "ebitda",
                "metric_value_scaled": 50000000,
            },
        ]

        count = repo.add_many(new_measurements)
        assert count == 2

        period_measurements = repo.list_for_deal_and_period(
            {"value": "deal-1"}, "2024-01-01", "2024-03-31"
        )
        assert len(period_measurements) == 2

        all_measurements = repo.list_for_deal({"value": "deal-1"})
        assert len(all_measurements) == 2


class TestCovenantResultRepositoryProtocol:
    """Tests that verify CovenantResultRepository protocol can be implemented."""

    def test_covenant_result_repository_implementation(self) -> None:
        """A class implementing CovenantResultRepository methods satisfies the protocol."""
        results: dict[str, CovenantResult] = {}

        class TestResultRepo:
            def save(self, result: CovenantResult) -> None:
                key = f"{result['covenant_id']['value']}:{result['period_start_iso']}"
                results[key] = result

            def save_many(self, new_results: Sequence[CovenantResult]) -> int:
                for result in new_results:
                    self.save(result)
                return len(new_results)

            def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
                # In a real implementation, this would join with covenants table
                return list(results.values())

            def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
                return [
                    r for r in results.values() if r["covenant_id"]["value"] == covenant_id["value"]
                ]

        repo: CovenantResultRepository = TestResultRepo()

        result: CovenantResult = {
            "covenant_id": {"value": "cov-1"},
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "calculated_value_scaled": 2500000,
            "status": "OK",
        }

        repo.save(result)
        covenant_results = repo.list_for_covenant({"value": "cov-1"})
        assert len(covenant_results) == 1
        assert covenant_results[0]["status"] == "OK"

        more_results: list[CovenantResult] = [
            {
                "covenant_id": {"value": "cov-1"},
                "period_start_iso": "2024-04-01",
                "period_end_iso": "2024-06-30",
                "calculated_value_scaled": 3000000,
                "status": "NEAR_BREACH",
            },
        ]
        count = repo.save_many(more_results)
        assert count == 1

        all_results = repo.list_for_deal({"value": "deal-1"})
        assert len(all_results) == 2
