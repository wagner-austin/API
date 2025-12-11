"""CLI entry point for seeding covenant-radar-api database.

Usage:
    python -m scripts.seed
    python -m scripts.seed --verbose
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Protocol

from covenant_radar_api.seeding import _test_hooks as seeding_hooks
from covenant_radar_api.seeding import seed_database_with_defaults
from covenant_radar_api.seeding.runner import SeedResult


class WriteFunc(Protocol):
    """Protocol for stdout write function."""

    def __call__(self, text: str) -> int: ...


class GetEnvFunc(Protocol):
    """Protocol for environment variable getter."""

    def __call__(self, key: str) -> str | None: ...


def _default_get_env(key: str) -> str | None:
    """Default environment getter using os.environ."""
    from platform_core.config._test_hooks import get_env as platform_get_env

    return platform_get_env(key)


def _default_write(text: str) -> int:
    """Default write function using sys.stdout."""
    written: int = sys.stdout.write(text)
    return written


# Hooks for testing
get_env: GetEnvFunc = _default_get_env
write: WriteFunc = _default_write


def _get_database_url() -> str:
    """Get database URL from environment.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    url = get_env("DATABASE_URL")
    if url is None:
        raise RuntimeError("DATABASE_URL environment variable is required")
    return url


def _print_result(result: SeedResult, verbose: bool) -> None:
    """Print seed result to stdout."""
    write("Seeded database:\n")
    write(f"  Deals: {result['deals_created']}\n")
    write(f"  Covenants: {result['covenants_created']}\n")
    write(f"  Measurements: {result['measurements_created']}\n")
    write(f"  Results: {result['results_created']}\n")
    if verbose:
        total = (
            result["deals_created"]
            + result["covenants_created"]
            + result["measurements_created"]
            + result["results_created"]
        )
        write(f"  Total entities: {total}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for seed script.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success).
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    verbose = "--verbose" in args or "-v" in args

    dsn = _get_database_url()
    conn = seeding_hooks.connection_factory(dsn)

    result = seed_database_with_defaults(conn)
    _print_result(result, verbose)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
