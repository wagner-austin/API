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


class ChildEnvironmentProtocol(Protocol):
    """Reads the whole environment a child process should inherit."""

    def __call__(self) -> dict[str, str]:
        """Copy the process environment.

        Returns:
            A copy, so a caller's overlay cannot reach back into this
            process.
        """
        ...


def _real_child_environment() -> dict[str, str]:
    """Production implementation — platform_core's canonical reader.

    Answers a process question rather than a configuration one: what
    a CHILD should inherit. The display-capture launch path overlays
    ``DISPLAY`` on this copy and hands it to Playwright, so the
    Chromium child finds its X server without this process ever
    mutating its own environment. ``platform_core.config._test_hooks``
    is the one module permitted to read ``os.environ``, and the
    import is function-level so its hook stays late-bound.

    Returns:
        A copy of the process environment.
    """
    from platform_core.config._test_hooks import get_environment

    return get_environment()


child_environment: ChildEnvironmentProtocol = _real_child_environment


__all__ = [
    "ChildEnvironmentProtocol",
    "_real_child_environment",
    "child_environment",
    "get_env",
]


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
