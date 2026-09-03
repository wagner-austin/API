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

    The rank must be the tank's TRUE current rank: a mid-session
    promotion raises the server's caps at the promoting kill
    instantly, while the wire 0x3D/0x47 rank field stays stale for
    the rest of the session and no 0x2B arrives (measured
    bot-20260725-211120 — the session's over-1000 readings at wire
    rank 0 were a recruit -> private promotion at kill #1, briefly
    mis-corrected the same day as a recruit-cap law and reverted on
    user ground truth).

    Wiki: [[game-economy]]#fuel-capacity.

    Args:
        rank: The tank's true rank, ``0`` (recruit) through ``8``
            (general).

    Returns:
        Fuel capacity ``1000 + 100 * rank``: 1000 at recruit, 1100 at
        private, ..., 1800 at general.
    """
    return 1000 + 100 * rank


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

    The tankpit.com official rules table (recruit 20, +5 per rank) is
    the source, live-confirmed at private via sustained 25s. The rank
    must be the tank's TRUE current rank — a mid-session promotion
    raises the cap at the promoting kill instantly while the wire
    rank field stays stale (bot-20260725-211120: slot counts crossed
    20 only after the promoting kill). Each of ``dual_shots``,
    ``missile_shots``, ``homing_shots``, ``extra_radars``, and
    ``armor_shields`` shares the same rank-derived cap; the server
    refuses further pickup with ``0x52`` code-7 when a slot would
    exceed the cap.

    Wiki: [[game-economy]]#inventory-capacity.

    Args:
        rank: The tank's true rank, ``0`` (recruit) through ``8``
            (general).

    Returns:
        Per-slot cap ``20 + 5 * rank``: 20 at recruit, 25 at private,
        30 at corporal, 35 at sergeant, 40 at lieutenant, 45 at
        captain, 50 at major, 55 at colonel, 60 at general.
    """
    return 20 + 5 * rank


RESERVE_REFERENCE_RANK = 4
"""The rank the fuel-reserve config values are tuned at.

``make_default_ai_config``'s reserves were hand-tuned against a
lieutenant (capacity 1400) and said so in its docstring while being
applied verbatim to every rank — at private (capacity 1100) the
flat break floor plus the conservative hits-to-kill bound consumed
the whole tank, and run bot-20260901-210631 broke off a winnable
fight at literally full fuel ([[flag-triage-20260902]] row 6,
21:43:44: fuel 1100/1100, projected 360 < floor 408). Scaling by
capacity preserves the tuning exactly at this rank and shrinks it
proportionally below."""


def rank_scaled_reserve(reference: int, rank: int) -> int:
    """Scale a reference-rank fuel reserve to a tank's true rank.

    ``reference * capacity(rank) // capacity(REFERENCE_RANK)`` —
    integer-exact at the reference rank, proportional elsewhere. Fuel
    is the life pool and every reserve is a fraction of life, not an
    absolute: a private's whole tank is 79% of a lieutenant's, so a
    floor tuned at lieutenant overprices every fight below it
    (the [[flag-triage-20260902]] row 6 full-tank break) and
    underprices none above.

    Args:
        reference: Reserve value as tuned at
            :data:`RESERVE_REFERENCE_RANK`.
        rank: The tank's true rank, ``0`` (recruit) through ``8``
            (general).

    Returns:
        The reserve scaled to the rank's fuel capacity.
    """
    return reference * fuel_capacity(rank) // fuel_capacity(RESERVE_REFERENCE_RANK)


__all__ = [
    "DEPOSIT_FLOOR",
    "RESERVE_REFERENCE_RANK",
    "damage_tier",
    "free_radar_radius",
    "fuel_capacity",
    "inventory_capacity",
    "rank_scaled_reserve",
]
