"""PostgreSQL implementations of repository protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from covenant_domain.models import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)

from .protocols import ConnectionProtocol


class PostgresDealRepository:
    """PostgreSQL implementation of DealRepository."""

    def __init__(self, conn: ConnectionProtocol) -> None:
        """Initialize with database connection."""
        self._conn = conn

    def create(self, deal: Deal) -> None:
        """Insert new deal. Raises on duplicate ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO deals (id, name, borrower, sector, region,
                             commitment_amount_cents, currency, maturity_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                deal["id"]["value"],
                deal["name"],
                deal["borrower"],
                deal["sector"],
                deal["region"],
                deal["commitment_amount_cents"],
                deal["currency"],
                deal["maturity_date_iso"],
            ),
        )
        self._conn.commit()

    def get(self, deal_id: DealId) -> Deal:
        """Get deal by ID. Raises KeyError if not found."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id::text, name, borrower, sector, region,
                   commitment_amount_cents, currency, maturity_date::text
            FROM deals WHERE id = %s
            """,
            (deal_id["value"],),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Deal not found: {deal_id['value']}")
        return _row_to_deal(row)

    def list_all(self) -> Sequence[Deal]:
        """List all deals."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id::text, name, borrower, sector, region,
                   commitment_amount_cents, currency, maturity_date::text
            FROM deals ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        return [_row_to_deal(row) for row in rows]

    def update(self, deal: Deal) -> None:
        """Update existing deal. Raises KeyError if not found."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE deals SET name = %s, borrower = %s, sector = %s, region = %s,
                           commitment_amount_cents = %s, currency = %s, maturity_date = %s
            WHERE id = %s
            """,
            (
                deal["name"],
                deal["borrower"],
                deal["sector"],
                deal["region"],
                deal["commitment_amount_cents"],
                deal["currency"],
                deal["maturity_date_iso"],
                deal["id"]["value"],
            ),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Deal not found: {deal['id']['value']}")
        self._conn.commit()

    def delete(self, deal_id: DealId) -> None:
        """Delete deal. Raises KeyError if not found."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM deals WHERE id = %s", (deal_id["value"],))
        if cursor.rowcount == 0:
            raise KeyError(f"Deal not found: {deal_id['value']}")
        self._conn.commit()


class PostgresCovenantRepository:
    """PostgreSQL implementation of CovenantRepository."""

    def __init__(self, conn: ConnectionProtocol) -> None:
        """Initialize with database connection."""
        self._conn = conn

    def create(self, covenant: Covenant) -> None:
        """Insert new covenant. Raises on duplicate ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO covenants (id, deal_id, name, formula,
                                  threshold_value_scaled, threshold_direction, frequency)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                covenant["id"]["value"],
                covenant["deal_id"]["value"],
                covenant["name"],
                covenant["formula"],
                covenant["threshold_value_scaled"],
                covenant["threshold_direction"],
                covenant["frequency"],
            ),
        )
        self._conn.commit()

    def get(self, covenant_id: CovenantId) -> Covenant:
        """Get covenant by ID. Raises KeyError if not found."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id::text, deal_id::text, name, formula,
                   threshold_value_scaled, threshold_direction, frequency
            FROM covenants WHERE id = %s
            """,
            (covenant_id["value"],),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Covenant not found: {covenant_id['value']}")
        return _row_to_covenant(row)

    def list_for_deal(self, deal_id: DealId) -> Sequence[Covenant]:
        """List all covenants for a deal."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id::text, deal_id::text, name, formula,
                   threshold_value_scaled, threshold_direction, frequency
            FROM covenants WHERE deal_id = %s ORDER BY created_at DESC
            """,
            (deal_id["value"],),
        )
        rows = cursor.fetchall()
        return [_row_to_covenant(row) for row in rows]

    def delete(self, covenant_id: CovenantId) -> None:
        """Delete covenant. Raises KeyError if not found."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM covenants WHERE id = %s", (covenant_id["value"],))
        if cursor.rowcount == 0:
            raise KeyError(f"Covenant not found: {covenant_id['value']}")
        self._conn.commit()


class PostgresMeasurementRepository:
    """PostgreSQL implementation of MeasurementRepository."""

    def __init__(self, conn: ConnectionProtocol) -> None:
        """Initialize with database connection."""
        self._conn = conn

    def add_many(self, measurements: Sequence[Measurement]) -> int:
        """Insert measurements. Returns count inserted. Raises on duplicate."""
        if len(measurements) == 0:
            return 0
        cursor = self._conn.cursor()
        count = 0
        for m in measurements:
            cursor.execute(
                """
                INSERT INTO measurements (deal_id, period_start, period_end,
                                         metric_name, metric_value_scaled)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    m["deal_id"]["value"],
                    m["period_start_iso"],
                    m["period_end_iso"],
                    m["metric_name"],
                    m["metric_value_scaled"],
                ),
            )
            count += 1
        self._conn.commit()
        return count

    def list_for_deal_and_period(
        self,
        deal_id: DealId,
        period_start_iso: str,
        period_end_iso: str,
    ) -> Sequence[Measurement]:
        """List measurements for deal and period."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT deal_id::text, period_start::text, period_end::text,
                   metric_name, metric_value_scaled
            FROM measurements
            WHERE deal_id = %s AND period_start = %s AND period_end = %s
            ORDER BY metric_name
            """,
            (deal_id["value"], period_start_iso, period_end_iso),
        )
        rows = cursor.fetchall()
        return [_row_to_measurement(row) for row in rows]

    def list_for_deal(self, deal_id: DealId) -> Sequence[Measurement]:
        """List all measurements for a deal."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT deal_id::text, period_start::text, period_end::text,
                   metric_name, metric_value_scaled
            FROM measurements WHERE deal_id = %s
            ORDER BY period_start DESC, metric_name
            """,
            (deal_id["value"],),
        )
        rows = cursor.fetchall()
        return [_row_to_measurement(row) for row in rows]


class PostgresCovenantResultRepository:
    """PostgreSQL implementation of CovenantResultRepository."""

    def __init__(self, conn: ConnectionProtocol) -> None:
        """Initialize with database connection."""
        self._conn = conn

    def save(self, result: CovenantResult) -> None:
        """Insert or update result."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO covenant_results (covenant_id, period_start, period_end,
                                         calculated_value_scaled, status)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (covenant_id, period_start, period_end)
            DO UPDATE SET calculated_value_scaled = EXCLUDED.calculated_value_scaled,
                         status = EXCLUDED.status
            """,
            (
                result["covenant_id"]["value"],
                result["period_start_iso"],
                result["period_end_iso"],
                result["calculated_value_scaled"],
                result["status"],
            ),
        )
        self._conn.commit()

    def save_many(self, results: Sequence[CovenantResult]) -> int:
        """Insert or update multiple results. Returns count."""
        if len(results) == 0:
            return 0
        count = 0
        for result in results:
            self.save(result)
            count += 1
        return count

    def list_for_deal(self, deal_id: DealId) -> Sequence[CovenantResult]:
        """List all results for a deal's covenants."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT cr.covenant_id::text, cr.period_start::text, cr.period_end::text,
                   cr.calculated_value_scaled, cr.status
            FROM covenant_results cr
            JOIN covenants c ON cr.covenant_id = c.id
            WHERE c.deal_id = %s
            ORDER BY cr.period_start DESC
            """,
            (deal_id["value"],),
        )
        rows = cursor.fetchall()
        return [_row_to_covenant_result(row) for row in rows]

    def list_for_covenant(self, covenant_id: CovenantId) -> Sequence[CovenantResult]:
        """List results for a specific covenant."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT covenant_id::text, period_start::text, period_end::text,
                   calculated_value_scaled, status
            FROM covenant_results WHERE covenant_id = %s
            ORDER BY period_start DESC
            """,
            (covenant_id["value"],),
        )
        rows = cursor.fetchall()
        return [_row_to_covenant_result(row) for row in rows]


def _row_to_deal(row: tuple[str | int | bool | None, ...]) -> Deal:
    """Convert database row to Deal TypedDict."""
    id_val = row[0]
    name = row[1]
    borrower = row[2]
    sector = row[3]
    region = row[4]
    commitment = row[5]
    currency = row[6]
    maturity = row[7]

    if not isinstance(id_val, str):
        raise TypeError(f"Expected str for id, got {type(id_val).__name__}")
    if not isinstance(name, str):
        raise TypeError(f"Expected str for name, got {type(name).__name__}")
    if not isinstance(borrower, str):
        raise TypeError(f"Expected str for borrower, got {type(borrower).__name__}")
    if not isinstance(sector, str):
        raise TypeError(f"Expected str for sector, got {type(sector).__name__}")
    if not isinstance(region, str):
        raise TypeError(f"Expected str for region, got {type(region).__name__}")
    if not isinstance(commitment, int):
        raise TypeError(f"Expected int for commitment, got {type(commitment).__name__}")
    if not isinstance(currency, str):
        raise TypeError(f"Expected str for currency, got {type(currency).__name__}")
    if not isinstance(maturity, str):
        raise TypeError(f"Expected str for maturity, got {type(maturity).__name__}")

    return Deal(
        id=DealId(value=id_val),
        name=name,
        borrower=borrower,
        sector=sector,
        region=region,
        commitment_amount_cents=commitment,
        currency=currency,
        maturity_date_iso=maturity,
    )


def _require_threshold_direction(value: str) -> Literal["<=", ">="]:
    """Validate and narrow threshold direction."""
    if value == "<=":
        return "<="
    if value == ">=":
        return ">="
    raise ValueError(f"Invalid threshold direction: {value}")


def _require_frequency(value: str) -> Literal["QUARTERLY", "ANNUAL"]:
    """Validate and narrow covenant frequency."""
    if value == "QUARTERLY":
        return "QUARTERLY"
    if value == "ANNUAL":
        return "ANNUAL"
    raise ValueError(f"Invalid frequency: {value}")


def _require_status(value: str) -> Literal["OK", "NEAR_BREACH", "BREACH"]:
    """Validate and narrow covenant status."""
    if value == "OK":
        return "OK"
    if value == "NEAR_BREACH":
        return "NEAR_BREACH"
    if value == "BREACH":
        return "BREACH"
    raise ValueError(f"Invalid status: {value}")


def _row_to_covenant(row: tuple[str | int | bool | None, ...]) -> Covenant:
    """Convert database row to Covenant TypedDict."""
    id_val = row[0]
    deal_id = row[1]
    name = row[2]
    formula = row[3]
    threshold = row[4]
    direction = row[5]
    frequency = row[6]

    if not isinstance(id_val, str):
        raise TypeError(f"Expected str for id, got {type(id_val).__name__}")
    if not isinstance(deal_id, str):
        raise TypeError(f"Expected str for deal_id, got {type(deal_id).__name__}")
    if not isinstance(name, str):
        raise TypeError(f"Expected str for name, got {type(name).__name__}")
    if not isinstance(formula, str):
        raise TypeError(f"Expected str for formula, got {type(formula).__name__}")
    if not isinstance(threshold, int):
        raise TypeError(f"Expected int for threshold, got {type(threshold).__name__}")
    if not isinstance(direction, str):
        raise TypeError(f"Expected str for direction, got {type(direction).__name__}")
    if not isinstance(frequency, str):
        raise TypeError(f"Expected str for frequency, got {type(frequency).__name__}")

    return Covenant(
        id=CovenantId(value=id_val),
        deal_id=DealId(value=deal_id),
        name=name,
        formula=formula,
        threshold_value_scaled=threshold,
        threshold_direction=_require_threshold_direction(direction),
        frequency=_require_frequency(frequency),
    )


def _row_to_measurement(row: tuple[str | int | bool | None, ...]) -> Measurement:
    """Convert database row to Measurement TypedDict."""
    deal_id = row[0]
    period_start = row[1]
    period_end = row[2]
    metric_name = row[3]
    metric_value = row[4]

    if not isinstance(deal_id, str):
        raise TypeError(f"Expected str for deal_id, got {type(deal_id).__name__}")
    if not isinstance(period_start, str):
        raise TypeError(f"Expected str for period_start, got {type(period_start).__name__}")
    if not isinstance(period_end, str):
        raise TypeError(f"Expected str for period_end, got {type(period_end).__name__}")
    if not isinstance(metric_name, str):
        raise TypeError(f"Expected str for metric_name, got {type(metric_name).__name__}")
    if not isinstance(metric_value, int):
        raise TypeError(f"Expected int for metric_value, got {type(metric_value).__name__}")

    return Measurement(
        deal_id=DealId(value=deal_id),
        period_start_iso=period_start,
        period_end_iso=period_end,
        metric_name=metric_name,
        metric_value_scaled=metric_value,
    )


def _row_to_covenant_result(row: tuple[str | int | bool | None, ...]) -> CovenantResult:
    """Convert database row to CovenantResult TypedDict."""
    covenant_id = row[0]
    period_start = row[1]
    period_end = row[2]
    calculated = row[3]
    status = row[4]

    if not isinstance(covenant_id, str):
        raise TypeError(f"Expected str for covenant_id, got {type(covenant_id).__name__}")
    if not isinstance(period_start, str):
        raise TypeError(f"Expected str for period_start, got {type(period_start).__name__}")
    if not isinstance(period_end, str):
        raise TypeError(f"Expected str for period_end, got {type(period_end).__name__}")
    if not isinstance(calculated, int):
        raise TypeError(f"Expected int for calculated, got {type(calculated).__name__}")
    if not isinstance(status, str):
        raise TypeError(f"Expected str for status, got {type(status).__name__}")

    return CovenantResult(
        covenant_id=CovenantId(value=covenant_id),
        period_start_iso=period_start,
        period_end_iso=period_end,
        calculated_value_scaled=calculated,
        status=_require_status(status),
    )


__all__ = [
    "PostgresCovenantRepository",
    "PostgresCovenantResultRepository",
    "PostgresDealRepository",
    "PostgresMeasurementRepository",
]
