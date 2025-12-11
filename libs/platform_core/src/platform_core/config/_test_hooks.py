"""Test hooks for platform_core config - allows injecting test dependencies."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable

from platform_core.json_utils import JSONValue


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return os.getenv(key)


def _default_tomllib_loads(s: str) -> dict[str, JSONValue]:
    """Production implementation - parses TOML string."""
    # Use getattr to call tomllib.loads avoiding Any propagation
    # tomllib.loads returns dict[str, Any], but valid TOML always produces
    # values that are valid JSON types (str, int, float, bool, list, dict, None)
    loads_func: Callable[[str], dict[str, JSONValue]] = tomllib.loads
    return loads_func(s)


# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env

# Hook for TOML parsing. Tests can override to test error handling.
tomllib_loads: Callable[[str], dict[str, JSONValue]] = _default_tomllib_loads
