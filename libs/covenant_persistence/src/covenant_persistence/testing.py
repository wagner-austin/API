"""Testing utilities for covenant_persistence.

Provides in-memory implementations of ConnectionProtocol and CursorProtocol
for testing services that use covenant_persistence without a real database.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from covenant_domain import (
    Covenant,
    CovenantId,
    CovenantResult,
    Deal,
    DealId,
    Measurement,
)

from .protocols import CursorProtocol

# =============================================================================
# Type Aliases
# =============================================================================

_Params = tuple[str | int | bool | None, ...]
_Row = tuple[str | int | bool | None, ...]
_QueryHandler = Callable[[str, _Params], None]


# =============================================================================
# In-Memory Store
# =============================================================================


class InMemoryStore:
    """Shared in-memory storage for all repositories.

    This store is used by InMemoryCursor to simulate PostgreSQL behavior.
    All data is stored in dictionaries keyed by ID.
    """

    def __init__(self) -> None:
        """Initialize empty data stores."""
        self.deals: dict[str, Deal] = {}
        self.covenants: dict[str, Covenant] = {}
        self.measurements: list[Measurement] = []
        self.covenant_results: list[CovenantResult] = []
        self._deal_order: list[str] = []
        self._covenant_order: list[str] = []


# =============================================================================
# In-Memory Cursor
# =============================================================================


class InMemoryCursor:
    """In-memory cursor that simulates psycopg cursor behavior."""

    def __init__(self, store: InMemoryStore) -> None:
        """Initialize cursor with store reference."""
        self._store = store
        self._results: list[_Row] = []
        self._rowcount = 0

    @property
    def rowcount(self) -> int:
        """Return number of rows affected by last execute."""
        return self._rowcount

    def execute(self, query: str, params: _Params = ()) -> None:
        """Execute SQL query against in-memory store."""
        self._results = []
        self._rowcount = 0
        query_lower = query.lower().strip()
        handler = self._get_handler(query_lower)
        if handler is not None:
            handler(query_lower, params)

    def _get_handler(self, query: str) -> _QueryHandler | None:
        """Get handler for query type."""
        handler = self._get_deal_handler(query)
        if handler is not None:
            return handler
        handler = self._get_covenant_handler(query)
        if handler is not None:
            return handler
        handler = self._get_measurement_handler(query)
        if handler is not None:
            return handler
        return self._get_result_handler(query)

    def _get_deal_handler(self, query: str) -> _QueryHandler | None:
        """Get handler for deal operations."""
        if query.startswith("insert into deals"):
            return self._insert_deal
        if query.startswith("select") and "from deals" in query:
            return self._select_deals
        if query.startswith("update deals"):
            return self._update_deal
        if query.startswith("delete from deals"):
            return self._delete_deal
        return None

    def _get_covenant_handler(self, query: str) -> _QueryHandler | None:
        """Get handler for covenant operations."""
        if query.startswith("insert into covenants"):
            return self._insert_covenant
        if query.startswith("select") and "from covenants" in query:
            return self._select_covenants
        if query.startswith("delete from covenants"):
            return self._delete_covenant
        return None

    def _get_measurement_handler(self, query: str) -> _QueryHandler | None:
        """Get handler for measurement operations."""
        if query.startswith("insert into measurements"):
            return self._insert_measurement
        if query.startswith("select") and "from measurements" in query:
            return self._select_measurements
        return None

    def _get_result_handler(self, query: str) -> _QueryHandler | None:
        """Get handler for covenant result operations."""
        if query.startswith("insert into covenant_results"):
            return self._insert_covenant_result
        if query.startswith("select") and "from covenant_results" in query:
            return self._select_covenant_results
        return None

    def _insert_deal(self, query: str, params: _Params) -> None:
        deal_id = str(params[0])
        if deal_id in self._store.deals:
            raise ValueError(f"Duplicate deal ID: {deal_id}")
        self._store.deals[deal_id] = Deal(
            id=DealId(value=deal_id),
            name=str(params[1]),
            borrower=str(params[2]),
            sector=str(params[3]),
            region=str(params[4]),
            commitment_amount_cents=int(params[5]) if params[5] is not None else 0,
            currency=str(params[6]),
            maturity_date_iso=str(params[7]),
        )
        self._store._deal_order.append(deal_id)
        self._rowcount = 1

    def _select_deals(self, query: str, params: _Params) -> None:
        if "where id = %s" in query:
            deal_id = str(params[0])
            if deal_id in self._store.deals:
                self._results = [_deal_to_row(self._store.deals[deal_id])]
        else:
            self._results = [
                _deal_to_row(self._store.deals[did])
                for did in reversed(self._store._deal_order)
                if did in self._store.deals
            ]

    def _update_deal(self, query: str, params: _Params) -> None:
        deal_id = str(params[7])
        if deal_id not in self._store.deals:
            self._rowcount = 0
            return
        self._store.deals[deal_id] = Deal(
            id=DealId(value=deal_id),
            name=str(params[0]),
            borrower=str(params[1]),
            sector=str(params[2]),
            region=str(params[3]),
            commitment_amount_cents=int(params[4]) if params[4] is not None else 0,
            currency=str(params[5]),
            maturity_date_iso=str(params[6]),
        )
        self._rowcount = 1

    def _delete_deal(self, query: str, params: _Params) -> None:
        deal_id = str(params[0])
        if deal_id in self._store.deals:
            del self._store.deals[deal_id]
            self._rowcount = 1
        else:
            self._rowcount = 0

    def _insert_covenant(self, query: str, params: _Params) -> None:
        cov_id = str(params[0])
        if cov_id in self._store.covenants:
            raise ValueError(f"Duplicate covenant ID: {cov_id}")
        self._store.covenants[cov_id] = Covenant(
            id=CovenantId(value=cov_id),
            deal_id=DealId(value=str(params[1])),
            name=str(params[2]),
            formula=str(params[3]),
            threshold_value_scaled=int(params[4]) if params[4] is not None else 0,
            threshold_direction=_require_direction(str(params[5])),
            frequency=_require_frequency(str(params[6])),
        )
        self._store._covenant_order.append(cov_id)
        self._rowcount = 1

    def _select_covenants(self, query: str, params: _Params) -> None:
        if "where id = %s" in query:
            cov_id = str(params[0])
            if cov_id in self._store.covenants:
                self._results = [_covenant_to_row(self._store.covenants[cov_id])]
        else:
            deal_id = str(params[0])
            rows: list[_Row] = []
            for cid in reversed(self._store._covenant_order):
                cov = self._store.covenants.get(cid)
                if cov is not None and cov["deal_id"]["value"] == deal_id:
                    rows.append(_covenant_to_row(cov))
            self._results = rows

    def _delete_covenant(self, query: str, params: _Params) -> None:
        cov_id = str(params[0])
        if cov_id in self._store.covenants:
            del self._store.covenants[cov_id]
            self._rowcount = 1
        else:
            self._rowcount = 0

    def _insert_measurement(self, query: str, params: _Params) -> None:
        m = Measurement(
            deal_id=DealId(value=str(params[0])),
            period_start_iso=str(params[1]),
            period_end_iso=str(params[2]),
            metric_name=str(params[3]),
            metric_value_scaled=int(params[4]) if params[4] is not None else 0,
        )
        for existing in self._store.measurements:
            if (
                existing["deal_id"]["value"] == m["deal_id"]["value"]
                and existing["period_start_iso"] == m["period_start_iso"]
                and existing["period_end_iso"] == m["period_end_iso"]
                and existing["metric_name"] == m["metric_name"]
            ):
                raise ValueError("Duplicate measurement")
        self._store.measurements.append(m)
        self._rowcount = 1

    def _select_measurements(self, query: str, params: _Params) -> None:
        deal_id = str(params[0])
        results: list[Measurement] = []
        if "period_start = %s" in query and "period_end = %s" in query:
            period_start = str(params[1])
            period_end = str(params[2])
            for m in self._store.measurements:
                if (
                    m["deal_id"]["value"] == deal_id
                    and m["period_start_iso"] == period_start
                    and m["period_end_iso"] == period_end
                ):
                    results.append(m)
            results.sort(key=lambda x: x["metric_name"])
        else:
            for m in self._store.measurements:
                if m["deal_id"]["value"] == deal_id:
                    results.append(m)
            results.sort(key=lambda x: (-_date_key(x["period_start_iso"]), x["metric_name"]))
        self._results = [_measurement_to_row(m) for m in results]

    def _insert_covenant_result(self, query: str, params: _Params) -> None:
        cov_id = str(params[0])
        period_start = str(params[1])
        period_end = str(params[2])
        calculated = int(params[3]) if params[3] is not None else 0
        status_str = str(params[4])
        self._store.covenant_results = [
            r
            for r in self._store.covenant_results
            if not (
                r["covenant_id"]["value"] == cov_id
                and r["period_start_iso"] == period_start
                and r["period_end_iso"] == period_end
            )
        ]
        self._store.covenant_results.append(
            CovenantResult(
                covenant_id=CovenantId(value=cov_id),
                period_start_iso=period_start,
                period_end_iso=period_end,
                calculated_value_scaled=calculated,
                status=_require_status(status_str),
            )
        )
        self._rowcount = 1

    def _select_covenant_results(self, query: str, params: _Params) -> None:
        results: list[CovenantResult] = []
        if "join covenants" in query:
            deal_id = str(params[0])
            cov_ids = {
                cid for cid, c in self._store.covenants.items() if c["deal_id"]["value"] == deal_id
            }
            for r in self._store.covenant_results:
                if r["covenant_id"]["value"] in cov_ids:
                    results.append(r)
        elif "where covenant_id = %s" in query:
            cov_id = str(params[0])
            for r in self._store.covenant_results:
                if r["covenant_id"]["value"] == cov_id:
                    results.append(r)
        results.sort(key=lambda x: x["period_start_iso"], reverse=True)
        self._results = [_covenant_result_to_row(r) for r in results]

    def fetchone(self) -> _Row | None:
        """Fetch one row from results."""
        if self._results:
            return self._results[0]
        return None

    def fetchall(self) -> Sequence[_Row]:
        """Fetch all rows from results."""
        return list(self._results)


# =============================================================================
# In-Memory Connection
# =============================================================================


class InMemoryConnection:
    """In-memory connection that satisfies ConnectionProtocol."""

    def __init__(self, store: InMemoryStore) -> None:
        """Initialize connection with store reference."""
        self._store = store
        self._closed = False

    def cursor(self) -> CursorProtocol:
        """Create a new cursor."""
        cursor: CursorProtocol = InMemoryCursor(self._store)
        return cursor

    def commit(self) -> None:
        """Commit transaction (no-op for in-memory)."""
        pass

    def rollback(self) -> None:
        """Rollback transaction (no-op for in-memory)."""
        pass

    def close(self) -> None:
        """Close connection."""
        self._closed = True


# =============================================================================
# Row Conversion Helpers
# =============================================================================


def _deal_to_row(d: Deal) -> _Row:
    return (
        d["id"]["value"],
        d["name"],
        d["borrower"],
        d["sector"],
        d["region"],
        d["commitment_amount_cents"],
        d["currency"],
        d["maturity_date_iso"],
    )


def _covenant_to_row(c: Covenant) -> _Row:
    return (
        c["id"]["value"],
        c["deal_id"]["value"],
        c["name"],
        c["formula"],
        c["threshold_value_scaled"],
        c["threshold_direction"],
        c["frequency"],
    )


def _measurement_to_row(m: Measurement) -> _Row:
    return (
        m["deal_id"]["value"],
        m["period_start_iso"],
        m["period_end_iso"],
        m["metric_name"],
        m["metric_value_scaled"],
    )


def _covenant_result_to_row(r: CovenantResult) -> _Row:
    return (
        r["covenant_id"]["value"],
        r["period_start_iso"],
        r["period_end_iso"],
        r["calculated_value_scaled"],
        r["status"],
    )


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_direction(value: str) -> Literal["<=", ">="]:
    if value == "<=":
        return "<="
    if value == ">=":
        return ">="
    raise ValueError(f"Invalid direction: {value}")


def _require_frequency(value: str) -> Literal["QUARTERLY", "ANNUAL"]:
    if value == "QUARTERLY":
        return "QUARTERLY"
    if value == "ANNUAL":
        return "ANNUAL"
    raise ValueError(f"Invalid frequency: {value}")


def _require_status(value: str) -> Literal["OK", "NEAR_BREACH", "BREACH"]:
    if value == "OK":
        return "OK"
    if value == "NEAR_BREACH":
        return "NEAR_BREACH"
    if value == "BREACH":
        return "BREACH"
    raise ValueError(f"Invalid status: {value}")


def _date_key(iso_date: str) -> int:
    return int(iso_date.replace("-", ""))


__all__ = [
    "InMemoryConnection",
    "InMemoryCursor",
    "InMemoryStore",
]
