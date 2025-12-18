from __future__ import annotations

from typing import Final

_NAMES: Final[tuple[str, ...]] = ("bloom",)


def list_available_post_effects() -> list[str]:
    """List registered post-effect names.

    Returns:
        list[str]: Stable list of post-effect identifiers.
    """

    return list(_NAMES)


__all__ = [
    "list_available_post_effects",
]
