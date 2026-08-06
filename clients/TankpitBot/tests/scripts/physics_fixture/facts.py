"""Fixture physics module for the claim-rule tests.

``__all__`` deliberately exposes exactly one constant and one
formula; the extra symbols exist so tests can bind claims to values
of the wrong shape (bool, str, non-int-returning callable) without
tripping the reverse-coverage rule.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TypedDict

ANSWER = 42
TRUTHY = True
NAME = "fixture"
#: Bytes constant for the ``bytes`` claim-kind tests. Kept OUT of
#: ``__all__`` so reverse coverage stays a two-symbol surface; the
#: bytes tests bind it explicitly by address.
GREETING = b"A1"


class Colour(IntEnum):
    """Enum for the ``members`` claim-kind tests."""

    RED = 0
    BLUE = 1


#: Mapping, sequence and set for the ``members`` claim-kind tests. All
#: kept OUT of ``__all__`` for the same reason as ``GREETING``.
COLOUR_FUEL: dict[Colour, int] = {Colour.RED: 10, Colour.BLUE: 20}
COLOUR_NAMES: tuple[str, ...] = ("red", "blue")
ODD_CODES: frozenset[int] = frozenset({3, 1, 2})
#: A mapping whose VALUES are enum members, not ints. Nothing in the
#: currently-bound modules has this shape, so without it the enum arm
#: of the normalizer's element unwrapping is never executed.
COLOUR_BY_NAME: dict[str, Colour] = {"red": Colour.RED, "blue": Colour.BLUE}


class SampleRecord(TypedDict):
    """Record type for the ``keys`` claim-kind tests."""

    left: int
    right: str


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
