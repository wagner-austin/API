"""Fake repository implementations for streaming worker testing.

These fakes implement the repository protocols from covenant_persistence
without requiring a real database connection.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Sequence

from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)


class FakeDealRepository:
    """Fake deal repository for testing.

    Stores deals in memory keyed by deal_id string.
    """

    def __init__(self) -> None:
        """Initialize empty deal store."""
        self._deals: dict[str, Deal] = {}

    def create(self, deal: Deal) -> None:
        """Insert new deal. Raises ValueError on duplicate ID.

        Args:
            deal: Deal to insert.

        Raises:
            ValueError: If deal already exists.
        """
        deal_id = deal["id"]["value"]
        if deal_id in self._deals:
            msg = f"Duplicate deal ID: {deal_id}"
            raise ValueError(msg)
        self._deals[deal_id] = deal

    def get(self, deal_id: DealId) -> Deal:
        """Get deal by ID. Raises KeyError if not found.

        Args:
            deal_id: Deal identifier.

        Returns:
            Deal data.

        Raises:
            KeyError: If deal not found.
        """
        value = deal_id["value"]
        if value not in self._deals:
            raise KeyError(value)
        return self._deals[value]

    def list_all(self) -> Sequence[Deal]:
        """List all deals.

        Returns:
            List of all deals.
        """
        return list(self._deals.values())

    def update(self, deal: Deal) -> None:
        """Update existing deal. Raises KeyError if not found.

        Args:
            deal: Deal to update.

        Raises:
            KeyError: If deal not found.
        """
        deal_id = deal["id"]["value"]
        if deal_id not in self._deals:
            raise KeyError(deal_id)
        self._deals[deal_id] = deal

    def delete(self, deal_id: DealId) -> None:
        """Delete deal. Raises KeyError if not found.

        Args:
            deal_id: Deal identifier.

        Raises:
            KeyError: If deal not found.
        """
        value = deal_id["value"]
        if value not in self._deals:
            raise KeyError(value)
        del self._deals[value]


class FakeCovenantRepository:
    """Fake covenant repository for testing.

    Stores covenants in memory keyed by covenant_id string.
    """

    def __init__(self) -> None:
        """Initialize empty covenant store."""
        self._covenants: dict[str, Covenant] = {}

    def create(self, covenant: Covenant) -> None:
        """Insert new covenant. Raises ValueError on duplicate ID.

        Args:
            covenant: Covenant to insert.

        Raises:
            ValueError: If covenant already exists.
        """
        cov_id = covenant["id"]["value"]
        if cov_id in self._covenants:
            msg = f"Duplicate covenant ID: {cov_id}"
            raise ValueError(msg)
        self._covenants[cov_id] = covenant

    def get(self, covenant_id: CovenantId) -> Covenant:
        """Get covenant by ID. Raises KeyError if not found.

        Args:
            covenant_id: Covenant identifier.

        Returns:
            Covenant data.

        Raises:
            KeyError: If covenant not found.
        """
        value = covenant_id["value"]
        if value not in self._covenants:
            raise KeyError(value)
        return self._covenants[value]

    def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
        """List all covenants for a deal.

        Args:
            deal_id: Deal identifier.

        Returns:
            List of covenants for the deal.
        """
        result: list[Covenant] = []
        deal_value = deal_id["value"]
        for covenant in self._covenants.values():
            if covenant["deal_id"]["value"] == deal_value:
                result.append(covenant)
        return result

    def delete(self, covenant_id: CovenantId) -> None:
        """Delete covenant. Raises KeyError if not found.

        Args:
            covenant_id: Covenant identifier.

        Raises:
            KeyError: If covenant not found.
        """
        value = covenant_id["value"]
        if value not in self._covenants:
            raise KeyError(value)
        del self._covenants[value]


class FakeMeasurementRepository:
    """Fake measurement repository for testing.

    Stores measurements in a list.
    """

    def __init__(self) -> None:
        """Initialize empty measurement store."""
        self._measurements: list[Measurement] = []

    def add_many(self, measurements: Sequence[Measurement]) -> int:
        """Insert measurements. Returns count inserted.

        Args:
            measurements: Measurements to insert.

        Returns:
            Number of measurements inserted.
        """
        self._measurements.extend(measurements)
        return len(measurements)

    def list_for_deal_and_period(
        self,
        deal_id: DealId,
        period_start_iso: str,
        period_end_iso: str,
    ) -> Sequence[Measurement]:
        """List measurements for deal and period.

        Args:
            deal_id: Deal identifier.
            period_start_iso: Period start date.
            period_end_iso: Period end date.

        Returns:
            List of measurements matching criteria.
        """
        result: list[Measurement] = []
        deal_value = deal_id["value"]
        for m in self._measurements:
            if (
                m["deal_id"]["value"] == deal_value
                and m["period_start_iso"] == period_start_iso
                and m["period_end_iso"] == period_end_iso
            ):
                result.append(m)
        return result

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal.

        Args:
            deal_id: Deal identifier.

        Returns:
            List of all measurements for the deal.
        """
        result: list[Measurement] = []
        deal_value = deal_id["value"]
        for m in self._measurements:
            if m["deal_id"]["value"] == deal_value:
                result.append(m)
        return result


class FakeCovenantResultRepository:
    """Fake covenant result repository for testing.

    Stores results in a list.
    """

    def __init__(self) -> None:
        """Initialize empty result store."""
        self._results: list[CovenantResult] = []

    def save(self, result: CovenantResult) -> None:
        """Insert or update result.

        Args:
            result: Result to save.
        """
        # Remove existing if same key
        cov_id = result["covenant_id"]["value"]
        period_start = result["period_start_iso"]
        period_end = result["period_end_iso"]

        filtered: list[CovenantResult] = []
        for r in self._results:
            if not (
                r["covenant_id"]["value"] == cov_id
                and r["period_start_iso"] == period_start
                and r["period_end_iso"] == period_end
            ):
                filtered.append(r)
        self._results = filtered
        self._results.append(result)

    def save_many(self, results: Sequence[CovenantResult]) -> int:
        """Insert or update multiple results.

        Args:
            results: Results to save.

        Returns:
            Number of results saved.
        """
        for result in results:
            self.save(result)
        return len(results)

    def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
        """List all results for a deal's covenants.

        Args:
            deal_id: Deal identifier.

        Returns:
            List of results for the deal.
        """
        # For testing, we return all results
        # In real implementation, this would filter by covenants belonging to the deal
        return list(self._results)

    def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
        """List results for a specific covenant.

        Args:
            covenant_id: Covenant identifier.

        Returns:
            List of results for the covenant.
        """
        result: list[CovenantResult] = []
        cov_value = covenant_id["value"]
        for r in self._results:
            if r["covenant_id"]["value"] == cov_value:
                result.append(r)
        return result


__all__ = [
    "FakeCovenantRepository",
    "FakeCovenantResultRepository",
    "FakeDealRepository",
    "FakeMeasurementRepository",
]
