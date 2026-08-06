"""Fixture physics module for the claim-rule tests.

``__all__`` deliberately exposes exactly one constant and one
formula; the extra symbols exist so tests can bind claims to values
of the wrong shape (bool, str, non-int-returning callable) without
tripping the reverse-coverage rule.
"""

from __future__ import annotations

ANSWER = 42
TRUTHY = True
NAME = "fixture"
#: Bytes constant for the ``bytes`` claim-kind tests. Kept OUT of
#: ``__all__`` so reverse coverage stays a two-symbol surface; the
#: bytes tests bind it explicitly by address.
GREETING = b"A1"


def double(value: int) -> int:
    """Return twice the value.

    Args:
        value: Input to double.

    Returns:
        ``2 * value``.
    """
    return 2 * value


def label(value: int) -> str:
    """Return a non-int result for probe-shape tests.

    Args:
        value: Input to label.

    Returns:
        The value with a ``v`` prefix.
    """
    return f"v{value}"


__all__ = ["ANSWER", "double"]
