"""Rank-derived limits and the built-in radar radius.

All three formulas are derived from ``self_state["rank"]`` on the
wire, so the bot knows them at tick 1 with no probing.

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

DEPOSIT_FLOOR = 100
"""Server-enforced minimum fuel left in the tank after a max deposit;
the client also refuses to initiate a deposit at or below this level
(``ce()`` gate in tpclient.js). Verified at four ranks 2026-07-06.
Wiki: [[game-economy]]#deposit-floor."""


def fuel_capacity(rank: int) -> int:
    """Return the tank's fuel capacity at the given rank.

    Recruits share the private-tier cap: the first rank-0 live
    session (bot-20260725-211120) carried 31 wire readings at exactly
    1100 and zero above it, including a pickup landing 943 -> 1100 —
    falsifying the old ``1000 + 100 * rank`` extrapolation at the one
    rank the 2026-07-06 deposit verifications (ranks 1/3/6/7) could
    not discriminate.

    Wiki: [[game-economy]]#fuel-capacity.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Fuel capacity ``1000 + 100 * max(rank, 1)``: 1100 at recruit
        AND private, 1200 at corporal, ..., 1800 at general.
    """
    return 1000 + 100 * max(rank, 1)


def damage_tier(fuel: int, rank: int) -> int:
    """Return the wire damage tier for a tank's fuel at the given rank.

    Tanks do NOT heal over time (user correction 2026-07-23): fuel IS
    the health pool, recovered only by pickups, and the rendered
    damage shade is a pure fuel-quartile indicator. Corpus-fitted the
    same day over every 0x2E sync carrying both fields (19,658
    samples, 246 sessions, zero exceptions; all fuel-carrying syncs
    are rank 1, boundaries exactly 275/550/825 = capacity quartiles):
    tier 3 is the top quartile (healthy, lightest shade), tier 0 the
    bottom (near death, darkest). Wiki: [[deactivation-format]].

    Args:
        fuel: Absolute wire fuel reading (0 through capacity).
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Damage tier ``min(3, 4 * fuel // capacity)``: quartile index
        of the fuel level, clamped so full fuel stays at tier 3.
    """
    return min(3, 4 * fuel // fuel_capacity(rank))


def free_radar_radius(rank: int) -> int:
    """Return the chebyshev radius of the built-in radar at the given rank.

    Wiki: [[radar-mechanics]]#free-radar-radius.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Chebyshev radius ``2 + rank // 3``: 2 (5x5) at recruit/private/
        corporal, 3 (7x7) at sergeant/lieutenant/captain, 4 (9x9) at
        major/colonel/general. Only the extra radar sweeps the full
        viewport regardless of rank.
    """
    return 2 + rank // 3


def inventory_capacity(rank: int) -> int:
    """Return the per-slot inventory capacity at the given rank.

    Recruits share the private-tier cap: the first rank-0 live
    session (bot-20260725-211120) sustained 0x49 snapshots at exactly
    25 in four slots at once — including slots the kill mercy bundle
    never touches — and never above 25, falsifying the tankpit.com
    rules-table's "recruit 20" at the wire. Each of ``dual_shots``,
    ``missile_shots``, ``homing_shots``, ``extra_radars``, and
    ``armor_shields`` shares the same rank-derived cap; the server
    refuses further pickup with ``0x52`` code-7 when a slot would
    exceed the cap.

    Wiki: [[game-economy]]#inventory-capacity.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Per-slot cap ``20 + 5 * max(rank, 1)``: 25 at recruit AND
        private, 30 at corporal, 35 at sergeant, 40 at lieutenant, 45
        at captain, 50 at major, 55 at colonel, 60 at general.
    """
    return 20 + 5 * max(rank, 1)


__all__ = [
    "DEPOSIT_FLOOR",
    "damage_tier",
    "free_radar_radius",
    "fuel_capacity",
    "inventory_capacity",
]
