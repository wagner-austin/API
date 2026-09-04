"""Test hooks for platform_core config - allows injecting test dependencies."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable

from platform_core.json_utils import JSONValue


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return os.getenv(key)


def _default_get_environment() -> dict[str, str]:
    """Production implementation - copies the whole process environment.

    Distinct from :func:`_default_get_env`, which answers a configuration
    question about one named key. This answers a process question: what a
    CHILD should inherit. ``subprocess`` replaces the environment wholesale
    when given one, so a caller adding a single variable has to start from
    the parent's -- and this module is the only one permitted to read it, so
    the alternative is every launcher growing its own ``os.environ`` access.

    Returns:
        A copy, so a caller's overlay cannot reach back into this process.
    """
    return dict(os.environ)


def _default_tomllib_loads(s: str) -> dict[str, JSONValue]:
    """Production implementation - parses TOML string."""
    # Use getattr to call tomllib.loads avoiding Any propagation
    # tomllib.loads returns dict[str, Any], but valid TOML always produces
    # values that are valid JSON types (str, int, float, bool, list, dict, None)
    loads_func: Callable[[str], dict[str, JSONValue]] = tomllib.loads
    return loads_func(s)


# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env

# Hook for reading the whole environment, for forwarding to a child process.
get_environment: Callable[[], dict[str, str]] = _default_get_environment

# Hook for TOML parsing. Tests can override to test error handling.
tomllib_loads: Callable[[str], dict[str, JSONValue]] = _default_tomllib_loads
