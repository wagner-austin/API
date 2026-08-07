"""Environment variable resolution hook.

Tests override ``get_env`` to inject deterministic values without
touching the real OS environment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from platform_core.config import _optional_env_str


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads via platform_core.

    Args:
        key: Environment variable name.

    Returns:
        Environment variable value or None if not set.
    """
    return _optional_env_str(key)


get_env: Callable[[str], str | None] = _default_get_env


__all__ = ["get_env"]


class LoadDotenvProtocol(Protocol):
    """Loads the ``.env`` file (or a fake in tests)."""

    def __call__(self) -> None:
        """Populate the process env from ``.env``."""
        ...


def _real_load_dotenv() -> None:
    """Production ``.env`` loader — thin wrapper around :mod:`dotenv`.

    The service main invokes this at process boot so shell-provided env
    vars still take precedence (dotenv leaves already-set values
    alone).
    """
    from dotenv import load_dotenv

    load_dotenv()


load_dotenv: LoadDotenvProtocol = _real_load_dotenv
