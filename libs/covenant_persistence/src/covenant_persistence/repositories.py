"""Repository protocol definitions for covenant domain entities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from covenant_domain.models import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)


class DealRepository(Protocol):
    """Repository for Deal operations."""

    def create(self, deal: Deal) -> None:
        """Insert new deal. Raises on duplicate ID."""
        ...

    def get(self, deal_id: DealId) -> Deal:
        """Get deal by ID. Raises KeyError if not found."""
        ...

    def list_all(self) -> Sequence[Deal]:
        """List all deals."""
        ...

    def update(self, deal: Deal) -> None:
        """Update existing deal. Raises KeyError if not found."""
        ...

    def delete(self, deal_id: DealId) -> None:
        """Delete deal. Raises KeyError if not found."""
        ...


class CovenantRepository(Protocol):
    """Repository for Covenant operations."""

    def create(self, covenant: Covenant) -> None:
        """Insert new covenant. Raises on duplicate ID."""
        ...

    def get(self, covenant_id: CovenantId) -> Covenant:
        """Get covenant by ID. Raises KeyError if not found."""
        ...

    def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
        """List all covenants for a deal."""
        ...

    def delete(self, covenant_id: CovenantId) -> None:
        """Delete covenant. Raises KeyError if not found."""
        ...


class MeasurementRepository(Protocol):
    """Repository for Measurement operations."""

    def add_many(self, measurements: Sequence[Measurement]) -> int:
        """Insert measurements. Returns count inserted. Raises on duplicate."""
        ...

    def list_for_deal_and_period(
        self,
        deal_id: DealId,
        period_start_iso: str,
        period_end_iso: str,
    ) -> Sequence[Measurement]:
        """List measurements for deal and period."""
        ...

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal."""
        ...


class CovenantResultRepository(Protocol):
    """Repository for CovenantResult operations."""

    def save(self, result: CovenantResult) -> None:
        """Insert or update result."""
        ...

    def save_many(self, results: Sequence[CovenantResult]) -> int:
        """Insert or update multiple results. Returns count."""
        ...

    def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
        """List all results for a deal's covenants."""
        ...

    def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
        """List results for a specific covenant."""
        ...


__all__ = [
    "CovenantRepository",
    "CovenantResultRepository",
    "DealRepository",
    "MeasurementRepository",
]
