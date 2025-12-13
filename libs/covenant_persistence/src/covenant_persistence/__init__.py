"""PostgreSQL repository layer for covenant monitoring with Protocol-based psycopg."""

from __future__ import annotations

from .postgres import (
    PostgresCovenantRepository,
    PostgresCovenantResultRepository,
    PostgresDealRepository,
    PostgresMeasurementRepository,
    ensure_schema,
)
from .protocols import (
    ConnectCallable,
    ConnectionProtocol,
    CursorProtocol,
    connect,
)
from .repositories import (
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)

__all__ = [
    "ConnectCallable",
    "ConnectionProtocol",
    "CovenantRepository",
    "CovenantResultRepository",
    "CursorProtocol",
    "DealRepository",
    "MeasurementRepository",
    "PostgresCovenantRepository",
    "PostgresCovenantResultRepository",
    "PostgresDealRepository",
    "PostgresMeasurementRepository",
    "connect",
    "ensure_schema",
]
