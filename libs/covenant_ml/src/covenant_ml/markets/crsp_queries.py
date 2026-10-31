"""Exact extraction SQL for both evidence partitions, derived only from the frozen protocol.

Each partition holds what is knowable by its observation end. The
train-validation partition ends the day before ``holdout_start``; the final
partition ends at ``coverage_end``. Daily returns, terminal events and
membership starts are cut at that day. Membership end bounds are clipped to it
in the query itself, so a later vendor end date never reaches the partition and
the clipped expression is preserved in the query bytes. A terminal event enters
only when its valuation date (the latest present stored, amount or next-price
date) also lies within the partition; an event with no valuation date is kept
for the evaluator to refuse. The CIZ calendar continues to the end of the exit
month of the partition's last decision, which ``crsp_windows`` requires; it
carries trading-day flags and no security observation.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import NamedTuple

from .crsp_calendar import crsp_month
from .crsp_protocol import CrspProtocol

_SCHEMA = re.compile(r"[a-z_][a-z0-9_]{0,62}")


class CrspScope(NamedTuple):
    """One partition's inclusive date bounds, all computed from the protocol."""

    partition: str
    history_start: str
    universe_start: str
    observation_end: str
    calendar_start: str
    calendar_end: str
    last_decision: str


def _month_end(month: int) -> str:
    """Return the last day of a consecutive month number from ``crsp_month``."""
    following = month + 1
    return (date(following // 12, following % 12 + 1, 1) - timedelta(days=1)).isoformat()


def crsp_scopes(protocol: CrspProtocol) -> tuple[CrspScope, CrspScope]:
    """Derive the train-validation and final partition bounds.

    Args:
        protocol: Decoded frozen design.

    Returns:
        The train-validation scope, then the final scope.

    Raises:
        ValueError: CRSP-010 when the validation calendar would pass ``coverage_end``.
    """
    boundary = (date.fromisoformat(protocol["holdout_start"]) - timedelta(days=1)).isoformat()
    calendar_start = protocol["history_start"][:8] + "01"
    validation_calendar = _month_end(crsp_month(boundary) + 2)
    if validation_calendar > protocol["coverage_end"]:
        raise ValueError("[CRSP-010] the validation calendar must end within coverage")
    return (
        CrspScope(
            "train-validation",
            protocol["history_start"],
            protocol["train_start"],
            boundary,
            calendar_start,
            validation_calendar,
            boundary,
        ),
        CrspScope(
            "final",
            protocol["history_start"],
            protocol["train_start"],
            protocol["coverage_end"],
            calendar_start,
            protocol["coverage_end"],
            protocol["holdout_end"],
        ),
    )


def require_schema(value: str) -> str:
    """Accept only a plain lowercase PostgreSQL schema identifier.

    Args:
        value: The vendor library or ``pg_temp`` for a synthetic session.

    Returns:
        The identifier, safe to place unquoted in SQL.

    Raises:
        ValueError: CRSP-010 for any other spelling.
    """
    if _SCHEMA.fullmatch(value) is None:
        raise ValueError("[CRSP-010] source schema must be a plain lowercase identifier")
    return value


def _codes(values: tuple[int, ...]) -> str:
    """Render validated integer codes as an SQL list."""
    return ", ".join(str(value) for value in values)


def _copy(select: str) -> str:
    """Wrap one bounded query as CSV with its header on standard output."""
    return f"COPY ({select}) TO STDOUT WITH (FORMAT CSV, HEADER);\n"


def crsp_relation_sql(
    protocol: CrspProtocol, scope: CrspScope, schema: str, max_rows: int
) -> dict[str, str]:
    """Render one partition's four COPY statements in protocol column order.

    Every query restricts securities to those with an eligible share and
    exchange interval overlapping ``universe_start`` through the partition's
    observation end, orders by the relation's natural key and fetches one row
    beyond ``max_rows`` so truncation is detectable rather than silent.

    Args:
        protocol: Decoded frozen design supplying tables, columns and codes.
        scope: Partition bounds from ``crsp_scopes``.
        schema: Validated source schema.
        max_rows: Positive per-relation row limit.

    Returns:
        Exact SQL bytes to save and execute, keyed by relation role.

    Raises:
        ValueError: CRSP-010 for an invalid schema or row limit.
    """
    source = require_schema(schema)
    if max_rows < 1:
        raise ValueError("[CRSP-010] the row limit must be positive")
    end, start = f"DATE '{scope.observation_end}'", f"DATE '{scope.history_start}'"
    names = f"{source}.{protocol['membership']}"
    universe = (
        f"SELECT DISTINCT permno FROM {names}"
        f" WHERE shrcd IN ({_codes(protocol['share_codes'])})"
        f" AND exchcd IN ({_codes(protocol['exchange_codes'])})"
        f" AND namedt <= {end} AND nameendt >= DATE '{scope.universe_start}'"
    )
    limit = f" LIMIT {max_rows + 1}"
    membership = ", ".join(
        f"LEAST(nameendt, {end}) AS nameendt" if column == "nameendt" else column
        for column in protocol["membership_columns"]
    )
    valuation = "COALESCE(GREATEST(deldlydt, delamtdt, delnextdt), delistingdt)"
    return {
        "membership": _copy(
            f"SELECT {membership} FROM {names} WHERE permno IN ({universe})"
            f" AND namedt <= {end} AND nameendt >= {start}"
            " ORDER BY permno, namedt, nameendt" + limit
        ),
        "returns": _copy(
            f"SELECT {', '.join(protocol['returns_columns'])}"
            f" FROM {source}.{protocol['returns']} WHERE permno IN ({universe})"
            f" AND dlycaldt BETWEEN {start} AND {end} ORDER BY permno, dlycaldt" + limit
        ),
        "calendar": _copy(
            f"SELECT {', '.join(protocol['calendar_columns'])}"
            f" FROM {source}.{protocol['calendar']}"
            f" WHERE exchangegroup = '{protocol['calendar_group']}'"
            f" AND caldt BETWEEN DATE '{scope.calendar_start}' AND DATE '{scope.calendar_end}'"
            " ORDER BY caldt" + limit
        ),
        "delists": _copy(
            f"SELECT {', '.join(protocol['delists_columns'])}"
            f" FROM {source}.{protocol['delists']} WHERE permno IN ({universe})"
            f" AND delistingdt BETWEEN {start} AND {end} AND {valuation} <= {end}"
            " ORDER BY permno, delistingdt" + limit
        ),
    }


def crsp_driver_sql(setup: str | None, statement_seconds: int) -> str:
    """Render the one-session script running every partition in one snapshot.

    The optional setup file runs before the transaction; it exists only for a
    synthetic session that builds temporary relations. All eight COPY files and
    two snapshot readings share one ``REPEATABLE READ READ ONLY`` transaction.

    Args:
        setup: Relative name of a synthetic setup script, or None for vendor data.
        statement_seconds: Positive per-statement server timeout.

    Returns:
        Exact psql script bytes, using only paths relative to its directory.

    Raises:
        ValueError: CRSP-010 for a nonpositive statement timeout.
    """
    if statement_seconds < 1:
        raise ValueError("[CRSP-010] the statement timeout must be positive")
    lines = [] if setup is None else [f"\\i {setup}"]
    lines += [
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;",
        f"SET LOCAL statement_timeout = '{statement_seconds}s';",
        "SET LOCAL DateStyle = 'ISO, YMD';",
        "SET LOCAL TimeZone = 'UTC';",
        "\\o snapshot-first.csv",
        "COPY (SELECT pg_current_snapshot()::text, now()) TO STDOUT WITH (FORMAT CSV);",
    ]
    for partition in ("train-validation", "final"):
        for relation in ("membership", "returns", "calendar", "delists"):
            lines += [f"\\o {partition}/{relation}.csv", f"\\i {partition}/{relation}.sql"]
    lines += [
        "\\o snapshot-last.csv",
        "COPY (SELECT pg_current_snapshot()::text, now()) TO STDOUT WITH (FORMAT CSV);",
        "\\o",
        "COMMIT;",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["CrspScope", "crsp_driver_sql", "crsp_relation_sql", "crsp_scopes", "require_schema"]
