from __future__ import annotations

from platform_core.config import _optional_env_str


def version_string() -> str:
    env_val = _optional_env_str("HANDWRITING_VERSION")
    return env_val if env_val is not None else "0.0.0"


__all__ = ["version_string"]
