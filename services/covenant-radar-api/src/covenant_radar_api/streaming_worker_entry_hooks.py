"""Test hooks for streaming worker entry dependency injection.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Protocol

from covenant_ml.types import PredictorProtocol
from covenant_persistence import (
    CovenantRepository,
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from covenant_persistence.protocols import ConnectCallable, ConnectionProtocol

# =============================================================================
# Logger Protocol
# =============================================================================


class LoggerProtocol(Protocol):
    """Protocol for logger used in streaming worker entry.

    Matches the interface of platform_core.logging.Logger for info/error methods.
    Uses extra dict for structured log fields.
    """

    def info(self, message: str, *, extra: dict[str, str] | None = None) -> None:
        """Log info message with optional extra fields.

        Args:
            message: Log message.
            extra: Additional structured log fields.
        """
        ...

    def error(self, message: str, *, extra: dict[str, str] | None = None) -> None:
        """Log error message with optional extra fields.

        Args:
            message: Log message.
            extra: Additional structured log fields.
        """
        ...


# =============================================================================
# Connection Factory Protocol
# =============================================================================


class ConnectionFactoryProtocol(Protocol):
    """Protocol for database connection factory.

    Creates a ConnectionProtocol from a database URL.
    """

    def __call__(self, database_url: str) -> ConnectionProtocol:
        """Create a database connection.

        Args:
            database_url: PostgreSQL connection string.

        Returns:
            Database connection implementing ConnectionProtocol.
        """
        ...


def _real_connection_factory(database_url: str) -> ConnectionProtocol:
    """Create a real psycopg database connection.

    Args:
        database_url: PostgreSQL connection string.

    Returns:
        psycopg connection implementing ConnectionProtocol.
    """
    psycopg = __import__("psycopg")
    connect_fn: ConnectCallable = psycopg.connect
    return connect_fn(database_url)


# =============================================================================
# Repository Factory Protocol
# =============================================================================


class RepositoryTuple(Protocol):
    """Protocol for tuple of repositories."""

    @property
    def deal_repo(self) -> DealRepository:
        """Deal repository."""
        ...

    @property
    def covenant_repo(self) -> CovenantRepository:
        """Covenant repository."""
        ...

    @property
    def measurement_repo(self) -> MeasurementRepository:
        """Measurement repository."""
        ...

    @property
    def result_repo(self) -> CovenantResultRepository:
        """Covenant result repository."""
        ...


class RepositoryFactoryProtocol(Protocol):
    """Protocol for repository factory.

    Creates all repositories from a database connection.
    """

    def __call__(
        self,
        conn: ConnectionProtocol,
    ) -> tuple[DealRepository, CovenantRepository, MeasurementRepository, CovenantResultRepository]:
        """Create repositories from connection.

        Args:
            conn: Database connection.

        Returns:
            Tuple of (deal_repo, covenant_repo, measurement_repo, result_repo).
        """
        ...


def _real_repository_factory(
    conn: ConnectionProtocol,
) -> tuple[DealRepository, CovenantRepository, MeasurementRepository, CovenantResultRepository]:
    """Create real PostgreSQL repositories.

    Args:
        conn: Database connection.

    Returns:
        Tuple of PostgreSQL repository implementations.
    """
    from covenant_persistence.postgres import (
        PostgresCovenantRepository,
        PostgresCovenantResultRepository,
        PostgresDealRepository,
        PostgresMeasurementRepository,
    )

    deal_repo: DealRepository = PostgresDealRepository(conn)
    covenant_repo: CovenantRepository = PostgresCovenantRepository(conn)
    measurement_repo: MeasurementRepository = PostgresMeasurementRepository(conn)
    result_repo: CovenantResultRepository = PostgresCovenantResultRepository(conn)

    return deal_repo, covenant_repo, measurement_repo, result_repo


# =============================================================================
# Model Loader Protocol
# =============================================================================


class ModelLoaderProtocol(Protocol):
    """Protocol for ML model loader."""

    def __call__(self, model_path: str) -> PredictorProtocol:
        """Load model from path.

        Args:
            model_path: Path to model file.

        Returns:
            Loaded model implementing PredictorProtocol.
        """
        ...


def _real_xgboost_loader(model_path: str) -> PredictorProtocol:
    """Load XGBoost model from file.

    Args:
        model_path: Path to XGBoost model file (.json or .ubj).

    Returns:
        Loaded XGBoost model.
    """
    from covenant_ml.predictor import load_model

    return load_model(model_path)


# =============================================================================
# Logger Factory Protocol
# =============================================================================


class LoggerFactoryProtocol(Protocol):
    """Protocol for logger factory."""

    def __call__(self, name: str) -> LoggerProtocol:
        """Get logger for module.

        Args:
            name: Module name.

        Returns:
            Logger instance.
        """
        ...


def _real_logger_factory(name: str) -> LoggerProtocol:
    """Get real logger from platform_core.

    Args:
        name: Module name.

    Returns:
        Logger instance.
    """
    from platform_core.logging import get_logger

    logger: LoggerProtocol = get_logger(name)
    return logger


# =============================================================================
# Module-Level Injectable Hooks
# =============================================================================

# Production code calls these; tests override before calling.
connection_factory: ConnectionFactoryProtocol = _real_connection_factory
repository_factory: RepositoryFactoryProtocol = _real_repository_factory
xgboost_loader: ModelLoaderProtocol = _real_xgboost_loader
logger_factory: LoggerFactoryProtocol = _real_logger_factory


# =============================================================================
# Fake Implementations for Testing
# =============================================================================


class FakeConnection:
    """Fake database connection for testing."""

    def __init__(self) -> None:
        """Initialize fake connection."""
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> FakeCursor:
        """Create fake cursor."""
        return FakeCursor()

    def commit(self) -> None:
        """Record commit."""
        self.committed = True

    def rollback(self) -> None:
        """Record rollback."""
        self.rolled_back = True

    def close(self) -> None:
        """Record close."""
        self.closed = True


class FakeCursor:
    """Fake database cursor for testing."""

    def __init__(self) -> None:
        """Initialize fake cursor."""
        self.executed_queries: list[tuple[str, tuple[str | int | bool | None, ...]]] = []
        self._rows: list[tuple[str | int | bool | None, ...]] = []
        self._rowcount = 0

    def execute(
        self,
        query: str,
        params: tuple[str | int | bool | None, ...] = (),
    ) -> None:
        """Record executed query."""
        self.executed_queries.append((query, params))

    def fetchone(self) -> tuple[str | int | bool | None, ...] | None:
        """Return first row or None."""
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self) -> list[tuple[str | int | bool | None, ...]]:
        """Return all rows."""
        result = list(self._rows)
        self._rows.clear()
        return result

    @property
    def rowcount(self) -> int:
        """Return row count."""
        return self._rowcount


def _fake_connection_factory(database_url: str) -> ConnectionProtocol:
    """Create fake connection for testing.

    Args:
        database_url: Ignored in fake.

    Returns:
        FakeConnection instance.
    """
    return FakeConnection()


__all__ = [
    "ConnectionFactoryProtocol",
    "FakeConnection",
    "FakeCursor",
    "LoggerFactoryProtocol",
    "LoggerProtocol",
    "ModelLoaderProtocol",
    "RepositoryFactoryProtocol",
    "_fake_connection_factory",
    "_real_connection_factory",
    "_real_logger_factory",
    "_real_repository_factory",
    "_real_xgboost_loader",
    "connection_factory",
    "logger_factory",
    "repository_factory",
    "xgboost_loader",
]
