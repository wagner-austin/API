"""Test hooks for seeding module dependency injection.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes before running.
"""

from __future__ import annotations

from typing import Protocol

from covenant_persistence import ConnectionProtocol


class ConnectionFactory(Protocol):
    """Protocol for creating database connections."""

    def __call__(self, dsn: str) -> ConnectionProtocol: ...


class UuidGenerator(Protocol):
    """Protocol for generating UUIDs."""

    def __call__(self) -> str: ...


class PsycopgModuleProtocol(Protocol):
    """Protocol for psycopg module with connect method."""

    def connect(self, dsn: str, autocommit: bool = False) -> ConnectionProtocol:
        """Connect to database."""
        ...


class LoadPsycopgModuleHook(Protocol):
    """Protocol for psycopg module loader hook."""

    def __call__(self) -> PsycopgModuleProtocol:
        """Load psycopg module."""
        ...


def _real_load_psycopg_module() -> PsycopgModuleProtocol:
    """Import psycopg and return it behind its protocol.

    Returns:
        The psycopg module.
    """
    module: PsycopgModuleProtocol = __import__("psycopg")
    return module


# Hook for loading psycopg module - tests rebind it to return a fake module.
load_psycopg_module: LoadPsycopgModuleHook = _real_load_psycopg_module


def _psycopg_connect_autocommit(dsn: str) -> ConnectionProtocol:
    """Connect to postgres with autocommit enabled.

    Uses autocommit=True to prevent failed transactions from blocking
    subsequent queries. Each statement commits immediately.
    """
    module = load_psycopg_module()
    conn: ConnectionProtocol = module.connect(dsn, autocommit=True)
    return conn


def _default_uuid_generator() -> str:
    """Default UUID generator using real uuid module."""
    import uuid

    generated: str = str(uuid.uuid4())
    return generated


# Module-level hooks - production defaults
# Tests override these before running
connection_factory: ConnectionFactory = _psycopg_connect_autocommit
uuid_generator: UuidGenerator = _default_uuid_generator


__all__ = [
    "ConnectionFactory",
    "UuidGenerator",
    "connection_factory",
    "uuid_generator",
]
