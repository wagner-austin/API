"""Environment variable resolution hook.

Tests override ``get_env`` to inject deterministic values without
touching the real OS environment.
"""

from __future__ import annotations

from collections.abc import Callable

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
