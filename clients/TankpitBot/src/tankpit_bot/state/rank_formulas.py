"""Rank-derived game constants (fuel capacity and built-in radar radius).

Both formulas are derived from ``self_state["rank"]`` on the wire, so
the bot knows them at tick 1 with no probing.

The mining chain (see ``wiki/pages/game-economy.md``, ``wiki/pages/
radar-mechanics.md``, ``wiki/pages/client-constants.md``):

* Fuel capacity was extracted from the client fuel-gauge draw (``Gc``
  in ``tpclient.js``): fill width ``7 * fuel / 100`` px against a
  capacity region of ``7 * (10 + rank)`` px. Equal iff
  ``fuel == 100 * (10 + rank)``. Verified at ranks 1/3/6/7 via user
  max-deposit arithmetic (deposit floor ``100`` server-enforced): a
  private deposited exactly 1000 (= 1100 - 100), a sergeant 1200
  (= 1300 - 100), a major 1500 (= 1600 - 100), a colonel 1598
  (= 1700 - 100 - two fuel walked).
* Built-in radar radius was verified at ranks 1/3/4/6/7 via manual
  axial reveals on the user's own tanks: private 2 (existing wire
  corpus), lieutenant 3 (111,129 -> 111,126), colonel 4 (165,125 ->
  165,129), then sergeant 3 (128,120 -> 128,123) and major 4 (234,5
  -> 238,5) chosen specifically to discriminate the two candidate
  step-boundary formulas. The steps fall at sergeant and major, so
  ``radius = 2 + rank // 3``.

Rank range is ``0..8`` (recruit .. general); the wire field is a
``uint8`` bounded by the client's rank table.
"""

from __future__ import annotations


def fuel_capacity(rank: int) -> int:
    """Return the tank's fuel capacity at the given rank.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Fuel capacity ``1000 + 100 * rank``: 1000 at recruit, 1100 at
        private, ..., 1800 at general.
    """
    return 1000 + 100 * rank


def free_radar_radius(rank: int) -> int:
    """Return the chebyshev radius of the built-in radar at the given rank.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Chebyshev radius ``2 + rank // 3``: 2 (5x5) at recruit/private/
        corporal, 3 (7x7) at sergeant/lieutenant/captain, 4 (9x9) at
        major/colonel/general. Only the extra radar sweeps the full
        viewport regardless of rank.
    """
    return 2 + rank // 3


__all__ = [
    "free_radar_radius",
    "fuel_capacity",
]
