"""Tests for fake implementations used in streaming worker tests.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import numpy as np
import pytest
from covenant_domain import CovenantId, DealId
from numpy.typing import NDArray

from covenant_radar_api.streaming._test_hooks_model import (
    FakeMetricsSink,
    FakePredictor,
)
from covenant_radar_api.streaming._test_hooks_repositories import (
    FakeCovenantRepository,
    FakeCovenantResultRepository,
    FakeDealRepository,
    FakeMeasurementRepository,
)

from ._test_worker_fixtures import (
    make_covenant,
    make_covenant_result,
    make_deal,
    make_measurement,
)


class TestFakeDealRepository:
    """Tests for FakeDealRepository."""

    def test_create_and_get(self) -> None:
        """Create and retrieve a deal."""
        repo = FakeDealRepository()
        deal = make_deal("deal-001")
        repo.create(deal)

        result = repo.get(DealId(value="deal-001"))
        assert result["name"] == "Test Deal"

    def test_create_duplicate_raises(self) -> None:
        """Creating duplicate deal raises ValueError."""
        repo = FakeDealRepository()
        deal = make_deal("deal-001")
        repo.create(deal)

        with pytest.raises(ValueError, match="Duplicate deal ID"):
            repo.create(deal)

    def test_get_not_found_raises(self) -> None:
        """Getting non-existent deal raises KeyError."""
        repo = FakeDealRepository()

        with pytest.raises(KeyError):
            repo.get(DealId(value="not-found"))

    def test_list_all(self) -> None:
        """List all deals."""
        repo = FakeDealRepository()
        repo.create(make_deal("deal-001", "Deal One"))
        repo.create(make_deal("deal-002", "Deal Two"))

        all_deals = repo.list_all()
        assert len(all_deals) == 2

    def test_update(self) -> None:
        """Update existing deal."""
        repo = FakeDealRepository()
        deal = make_deal("deal-001", "Original")
        repo.create(deal)

        updated = make_deal("deal-001", "Updated")
        repo.update(updated)

        result = repo.get(DealId(value="deal-001"))
        assert result["name"] == "Updated"

    def test_update_not_found_raises(self) -> None:
        """Updating non-existent deal raises KeyError."""
        repo = FakeDealRepository()

        with pytest.raises(KeyError):
            repo.update(make_deal("not-found"))

    def test_delete(self) -> None:
        """Delete a deal."""
        repo = FakeDealRepository()
        deal = make_deal("deal-001")
        repo.create(deal)

        repo.delete(DealId(value="deal-001"))

        with pytest.raises(KeyError):
            repo.get(DealId(value="deal-001"))

    def test_delete_not_found_raises(self) -> None:
        """Deleting non-existent deal raises KeyError."""
        repo = FakeDealRepository()

        with pytest.raises(KeyError):
            repo.delete(DealId(value="not-found"))


class TestFakeCovenantRepository:
    """Tests for FakeCovenantRepository."""

    def test_create_and_get(self) -> None:
        """Create and retrieve a covenant."""
        repo = FakeCovenantRepository()
        covenant = make_covenant("cov-001", "deal-001")
        repo.create(covenant)

        result = repo.get(CovenantId(value="cov-001"))
        assert result["formula"] == "total_debt / ebitda"

    def test_create_duplicate_raises(self) -> None:
        """Creating duplicate covenant raises ValueError."""
        repo = FakeCovenantRepository()
        covenant = make_covenant("cov-001")
        repo.create(covenant)

        with pytest.raises(ValueError, match="Duplicate covenant ID"):
            repo.create(covenant)

    def test_get_not_found_raises(self) -> None:
        """Getting non-existent covenant raises KeyError."""
        repo = FakeCovenantRepository()

        with pytest.raises(KeyError):
            repo.get(CovenantId(value="not-found"))

    def test_list_for_deal(self) -> None:
        """List covenants for a deal."""
        repo = FakeCovenantRepository()
        repo.create(make_covenant("cov-001", "deal-001"))
        repo.create(make_covenant("cov-002", "deal-001"))
        repo.create(make_covenant("cov-003", "deal-002"))

        covenants = repo.list_for_deal(DealId(value="deal-001"))
        assert len(covenants) == 2

    def test_delete(self) -> None:
        """Delete a covenant."""
        repo = FakeCovenantRepository()
        repo.create(make_covenant("cov-001"))

        repo.delete(CovenantId(value="cov-001"))

        with pytest.raises(KeyError):
            repo.get(CovenantId(value="cov-001"))

    def test_delete_not_found_raises(self) -> None:
        """Deleting non-existent covenant raises KeyError."""
        repo = FakeCovenantRepository()

        with pytest.raises(KeyError):
            repo.delete(CovenantId(value="not-found"))


class TestFakeMeasurementRepository:
    """Tests for FakeMeasurementRepository."""

    def test_add_many(self) -> None:
        """Add multiple measurements."""
        repo = FakeMeasurementRepository()
        measurements = [
            make_measurement("deal-001", "debt_to_equity"),
            make_measurement("deal-001", "current_ratio"),
        ]
        count = repo.add_many(measurements)
        assert count == 2

    def test_list_for_deal(self) -> None:
        """List measurements for a deal."""
        repo = FakeMeasurementRepository()
        repo.add_many(
            [
                make_measurement("deal-001", "debt_to_equity"),
                make_measurement("deal-002", "current_ratio"),
            ]
        )

        results = repo.list_for_deal(DealId(value="deal-001"))
        assert len(results) == 1
        assert results[0]["metric_name"] == "debt_to_equity"

    def test_list_for_deal_and_period(self) -> None:
        """List measurements for deal and period."""
        repo = FakeMeasurementRepository()
        repo.add_many(
            [
                make_measurement(
                    "deal-001", "debt_to_equity", period_start="2024-01-01", period_end="2024-03-31"
                ),
                make_measurement(
                    "deal-001", "current_ratio", period_start="2024-04-01", period_end="2024-06-30"
                ),
            ]
        )

        results = repo.list_for_deal_and_period(
            DealId(value="deal-001"),
            "2024-01-01",
            "2024-03-31",
        )
        assert len(results) == 1
        assert results[0]["metric_name"] == "debt_to_equity"


class TestFakeCovenantResultRepository:
    """Tests for FakeCovenantResultRepository."""

    def test_save(self) -> None:
        """Save a covenant result."""
        repo = FakeCovenantResultRepository()
        result = make_covenant_result("cov-001")
        repo.save(result)

        results = repo.list_for_covenant(CovenantId(value="cov-001"))
        assert len(results) == 1

    def test_save_replaces_existing(self) -> None:
        """Save replaces existing result for same key."""
        repo = FakeCovenantResultRepository()
        result1 = make_covenant_result("cov-001", status="OK")
        repo.save(result1)

        result2 = make_covenant_result("cov-001", status="BREACH")
        repo.save(result2)

        results = repo.list_for_covenant(CovenantId(value="cov-001"))
        assert len(results) == 1
        assert results[0]["status"] == "BREACH"

    def test_save_many(self) -> None:
        """Save multiple results."""
        repo = FakeCovenantResultRepository()
        results = [
            make_covenant_result("cov-001"),
            make_covenant_result("cov-002"),
        ]
        count = repo.save_many(results)
        assert count == 2

    def test_list_for_deal(self) -> None:
        """List results for a deal."""
        repo = FakeCovenantResultRepository()
        repo.save(make_covenant_result("cov-001"))
        repo.save(make_covenant_result("cov-002"))

        results = repo.list_for_deal(DealId(value="deal-001"))
        assert len(results) == 2

    def test_list_for_covenant(self) -> None:
        """List results for a covenant."""
        repo = FakeCovenantResultRepository()
        repo.save(make_covenant_result("cov-001", period_end="2024-03-31"))
        repo.save(make_covenant_result("cov-001", period_end="2024-06-30"))
        repo.save(make_covenant_result("cov-002", period_end="2024-03-31"))

        results = repo.list_for_covenant(CovenantId(value="cov-001"))
        assert len(results) == 2


class TestFakePredictor:
    """Tests for FakePredictor."""

    def test_predict_proba_returns_configured_probability(self) -> None:
        """Returns configured probability."""
        predictor = FakePredictor(default_probability=0.75)
        x: NDArray[np.float64] = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float64)

        result = predictor.predict_proba(x)

        assert result.shape == (2, 2)
        result_list: list[list[float]] = result.tolist()
        assert result_list[0][0] == 0.25  # 1 - 0.75
        assert result_list[0][1] == 0.75
        assert result_list[1][1] == 0.75

    def test_call_count_tracks_calls(self) -> None:
        """Tracks number of calls."""
        predictor = FakePredictor()
        assert predictor.call_count == 0

        x1: NDArray[np.float64] = np.array(((1.0,),), dtype=np.float64)
        predictor.predict_proba(x1)
        assert predictor.call_count == 1

        x2: NDArray[np.float64] = np.array(((2.0,),), dtype=np.float64)
        predictor.predict_proba(x2)
        assert predictor.call_count == 2


class TestFakeMetricsSink:
    """Tests for FakeMetricsSink."""

    def test_increment_records(self) -> None:
        """Records increment calls."""
        sink = FakeMetricsSink()
        sink.increment("test.metric", 1, ("tag:value",))

        assert len(sink.increments) == 1
        assert sink.increments[0] == ("test.metric", 1, ("tag:value",))

    def test_gauge_records(self) -> None:
        """Records gauge calls."""
        sink = FakeMetricsSink()
        sink.gauge("test.gauge", 42.5, ("tag:value",))

        assert len(sink.gauges) == 1
        assert sink.gauges[0] == ("test.gauge", 42.5, ("tag:value",))

    def test_histogram_records(self) -> None:
        """Records histogram calls."""
        sink = FakeMetricsSink()
        sink.histogram("test.histogram", 100.0, ("tag:value",))

        assert len(sink.histograms) == 1
        assert sink.histograms[0] == ("test.histogram", 100.0, ("tag:value",))
