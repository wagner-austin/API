"""Presence-scoped incoming rate: whose hits still price the fight.

[[flag-triage-20260902]] row 11: the break projection must be fed
only by shooters who can still fire on us. The world service owns the
exclusion policy over the policy-free damage book: registry-dead and
registry-silent shooters leave the window; unknown shooters never do.
"""

from __future__ import annotations

from tankpit_bot.ledger.damage_book import (
    confirm_incoming_damage,
    record_incoming_shot,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    TankStateDict,
    WorldStateDict,
    make_self_state,
    make_tank_state,
)
from tankpit_bot.types.constants import TankLiveness

_NOW = 10_000_000
_WINDOW_MS = 10_000
_PRESENCE_TTL_MS = 7_000


def _shooter(
    tank_id: int,
    *,
    last_wire_seen_ms: int,
    liveness: TankLiveness = "alive",
) -> TankStateDict:
    """One registry row for a shooter.

    Args:
        tank_id: Registry id.
        last_wire_seen_ms: Newest wire-sourced observation stamp.
        liveness: Registry liveness verdict.

    Returns:
        A registered enemy at (50,50).
    """
    return make_tank_state(
        tank_id=tank_id,
        x=50,
        y=50,
        team=2,
        rank=2,
        damage_state=3,
        name="Raider",
        is_bot=False,
        is_self=False,
        timestamp_ms=last_wire_seen_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        liveness=liveness,
    )


def _service_with_hits(
    tanks: dict[str, TankStateDict],
    shooter_ids: list[int],
) -> WorldService:
    """A service whose damage book holds one confirmed dual per shooter.

    Every hit is confirmed just inside the window, so only the
    presence policy — never the window prune — decides who counts.

    Args:
        tanks: Registry rows keyed by id string.
        shooter_ids: One confirmed -90 hit is recorded per id, in
            order, at one-second spacing ending 1 s before ``_NOW``.

    Returns:
        The populated service (self is tank id 601).
    """
    ws = WorldService()
    ws.world_state = WorldStateDict(
        **{
            **ws.world_state,
            "self_state": make_self_state(
                tank_id=601,
                x=100,
                y=100,
                team=1,
                rank=1,
                fuel=900,
                leaderboard_position=0,
            ),
            "tanks": tanks,
        }
    )
    for offset, shooter_id in enumerate(shooter_ids):
        ts = _NOW - 1000 * (len(shooter_ids) - offset)
        record_incoming_shot(ws.damage_book, shooter_id, "Raider", 1, ts)
        confirm_incoming_damage(ws.damage_book, -90, ts + 100)
    return ws


def test_a_wire_fresh_shooter_still_prices_the_fight() -> None:
    """A registered shooter heard from within the presence TTL counts."""
    ws = _service_with_hits(
        {"700": _shooter(700, last_wire_seen_ms=_NOW - 2000)},
        [700],
    )

    assert ws.get_incoming_damage_window(_NOW, _WINDOW_MS, _PRESENCE_TTL_MS) == (1, 90)


def test_a_silent_pair_mate_leaves_the_rate_window() -> None:
    """The row-11 defect shape: the disengaged pair-mate stops pricing.

    Shooter 700 is fresh and duelling; shooter 800 landed a hit but
    has been wire-silent past the presence TTL — they left the fight.
    Before this law both hits fed the projection and the pair's
    combined rate priced a one-on-one duel; now only the duellist's
    hit counts.
    """
    ws = _service_with_hits(
        {
            "700": _shooter(700, last_wire_seen_ms=_NOW - 2000),
            "800": _shooter(800, last_wire_seen_ms=_NOW - _PRESENCE_TTL_MS - 1000),
        },
        [700, 800],
    )

    assert ws.get_incoming_damage_window(_NOW, _WINDOW_MS, _PRESENCE_TTL_MS) == (1, 90)


def test_a_deactivated_shooter_leaves_the_rate_window() -> None:
    """The 2026-07-31 arena-soak law survives the generalization."""
    ws = _service_with_hits(
        {"700": _shooter(700, last_wire_seen_ms=_NOW - 1000, liveness="deactivated")},
        [700],
    )

    assert ws.get_incoming_damage_window(_NOW, _WINDOW_MS, _PRESENCE_TTL_MS) == (0, 0)


def test_an_unknown_shooter_always_counts() -> None:
    """A registry gap can never under-report live danger."""
    ws = _service_with_hits({}, [999])

    assert ws.get_incoming_damage_window(_NOW, _WINDOW_MS, _PRESENCE_TTL_MS) == (1, 90)


def test_presence_flips_back_within_the_window() -> None:
    """A silent shooter heard from again re-prices their earlier hits.

    Excluded entries stay in the log until the window prunes them, so
    presence is re-evaluated on every read — the same flip-back
    guarantee the dead-shooter exclusion documented for respawns.
    """
    silent = _shooter(800, last_wire_seen_ms=_NOW - _PRESENCE_TTL_MS - 1000)
    ws = _service_with_hits({"800": silent}, [800])
    assert ws.get_incoming_damage_window(_NOW, _WINDOW_MS, _PRESENCE_TTL_MS) == (0, 0)

    ws.world_state["tanks"]["800"] = _shooter(800, last_wire_seen_ms=_NOW)

    assert ws.get_incoming_damage_window(_NOW, _WINDOW_MS, _PRESENCE_TTL_MS) == (1, 90)
