"""Engagement doctrines: timing policy over WHEN a consented fight opens.

Operator order 2026-09-01 ("will we have pluggable strategies that we
can swap in per bot or per group of bots at a whim?"): a doctrine is
DATA selecting between existing tested gates. Consent and the human
rank window stay senior and untouched — every case here starts from a
CONSENTED human, which the skirmish default acquires.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.threat_acquisition import find_acquisition_target
from tankpit_bot.fleetshare.types import EngagementDoctrine
from tankpit_bot.ledger.damage_book import record_incoming_shot
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
    make_tank_state,
)

_HUMAN_ID = 1229


def _consented_human_world() -> tuple[WorldService, WorldStateDict, SelfStateDict]:
    """A consented human at (110,100), self at (100,100), fuel for the fight."""
    ws = WorldService()
    record_incoming_shot(ws.damage_book, _HUMAN_ID, "Triumvirate", 1, 100000)
    world = make_empty_world_state()
    self_state = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    tank = make_tank_state(
        tank_id=_HUMAN_ID,
        x=110,
        y=100,
        team=1,
        rank=1,
        damage_state=0,
        name="Triumvirate",
        is_bot=False,
        is_self=False,
    )
    tank["timestamp_ms"] = 100000
    tank["last_viewport_observation_ms"] = 100000
    world["tanks"][str(_HUMAN_ID)] = tank
    return ws, world, self_state


def _acquire(
    ws: WorldService,
    world: WorldStateDict,
    self_state: SelfStateDict,
    doctrine: EngagementDoctrine,
) -> int:
    """Run acquisition under the doctrine; -1 when nothing acquired."""
    result = find_acquisition_target(
        ws,
        world,
        self_state,
        blocked={},
        killed={},
        terrain=None,
        now_ms=100000,
        map_intel_horizon_ms=5000,
        engagement_reserve_fuel=650,
        doctrine=doctrine,
    )
    return -1 if result is None else result["tank_id"]


def test_skirmish_acquires_the_consented_human() -> None:
    """The default doctrine is today's behavior: consent is enough."""
    ws, world, self_state = _consented_human_world()

    assert _acquire(ws, world, self_state, "skirmish") == _HUMAN_ID


def test_passive_never_initiates_against_humans() -> None:
    """Consent notwithstanding, a passive bot never opens the fight."""
    ws, world, self_state = _consented_human_world()

    assert _acquire(ws, world, self_state, "passive") == -1


def test_duelist_takes_an_unclaimed_duel() -> None:
    """First come: with no sibling engaged, the duelist fights."""
    ws, world, self_state = _consented_human_world()

    assert _acquire(ws, world, self_state, "duelist") == _HUMAN_ID


def test_duelist_yields_a_claimed_duel() -> None:
    """A sibling already on the human keeps every other duelist out."""
    ws, world, self_state = _consented_human_world()
    ws.fleet_engaged_target_ids = {_HUMAN_ID: 99000}

    assert _acquire(ws, world, self_state, "duelist") == -1


def test_swarm_holds_until_the_muster_quorum_stands() -> None:
    """Nobody engaged and no war-ready sibling: the swarm bot farms on."""
    ws, world, self_state = _consented_human_world()

    assert _acquire(ws, world, self_state, "swarm") == -1


def test_swarm_strikes_when_the_quorum_stands() -> None:
    """One war-ready sibling makes two: the swarm hits together."""
    ws, world, self_state = _consented_human_world()
    ws.fleet_war_ready_count = 1

    assert _acquire(ws, world, self_state, "swarm") == _HUMAN_ID


def test_swarm_reinforces_an_engaged_sibling_without_a_quorum() -> None:
    """A fight a sibling already holds needs no muster — join now."""
    ws, world, self_state = _consented_human_world()
    ws.fleet_engaged_target_ids = {_HUMAN_ID: 99000}

    assert _acquire(ws, world, self_state, "swarm") == _HUMAN_ID


def test_doctrine_never_touches_practice_bots() -> None:
    """A passive bot still farms practice bots exactly as before."""
    ws, world, self_state = _consented_human_world()
    bot = make_tank_state(
        tank_id=530,
        x=105,
        y=100,
        team=1,
        rank=1,
        damage_state=0,
        name="orange-4",
        is_bot=True,
        is_self=False,
    )
    bot["timestamp_ms"] = 100000
    bot["last_viewport_observation_ms"] = 100000
    world["tanks"]["530"] = bot

    assert _acquire(ws, world, self_state, "passive") == 530
