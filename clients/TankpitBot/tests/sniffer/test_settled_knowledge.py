"""The settled-knowledge watermark: who counts as "a human is about".

[[flag-triage-20260902]] rows 3-5: scan staleness is a fact question —
only a foreign human can change ground unobserved — and the
``knowledge_floor_ms`` sweep is the instrument that answers it from
the tank registry.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.knowledge_floors import FORAGE_COVERAGE_TTL_MS
from tankpit_bot.state.types import (
    TankStateDict,
    WorldStateDict,
    make_self_state,
    make_tank_state,
)

_NOW = 10_000_000


def _tank(tank_id: int, name: str, timestamp_ms: int) -> TankStateDict:
    """Build one registry observation.

    Args:
        tank_id: Registry id.
        name: Wire display name — the human/practice-bot discriminator.
        timestamp_ms: Observation stamp.

    Returns:
        An alive tank observation at (50,50).
    """
    return make_tank_state(
        tank_id=tank_id,
        x=50,
        y=50,
        team=2,
        rank=2,
        damage_state=3,
        name=name,
        is_bot=False,
        is_self=False,
        timestamp_ms=timestamp_ms,
        liveness="alive",
    )


def _service(tanks: dict[str, TankStateDict]) -> WorldService:
    """Build a service whose registry holds exactly ``tanks``.

    Args:
        tanks: Registry rows keyed by id string.

    Returns:
        A service with self at tank id 601.
    """
    ws = WorldService()
    ws.world_state = WorldStateDict(
        **{
            **ws.world_state,
            "self_state": make_self_state(
                tank_id=601,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=0,
            ),
            "tanks": tanks,
        }
    )
    return ws


def test_an_empty_registry_leaves_the_room_settled() -> None:
    """No tanks at all: the floor is 0 and knowledge is permanent."""
    ws = _service({})

    assert ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS) == 0
    assert ws.last_foreign_human_seen_ms == 0


def test_a_foreign_human_restores_the_clock() -> None:
    """A human-named stranger seen now yields exactly the old TTL."""
    ws = _service({"900": _tank(900, "Sigma", _NOW)})

    assert ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS) == _NOW - FORAGE_COVERAGE_TTL_MS
    assert ws.last_foreign_human_seen_ms == _NOW


def test_practice_bots_never_unsettle_the_room() -> None:
    """Server bots are named by color-number and never collect."""
    ws = _service({"540": _tank(540, "red-5", _NOW)})

    assert ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS) == 0


def test_the_bots_own_tank_never_counts() -> None:
    """Self carries a human-style account name; identity excludes it."""
    ws = _service({"601": _tank(601, "Artax", _NOW)})

    assert ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS) == 0


def test_fleet_siblings_never_count() -> None:
    """Sibling accounts are human-named; the merge's id set excludes them.

    Without this, a 4-bot World fleet reads its own members as humans
    and no fleet room ever settles.
    """
    ws = _service({"602": _tank(602, "Arterial", _NOW)})
    ws.fleet_sibling_tank_ids = {602}

    assert ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS) == 0


def test_the_watermark_survives_the_humans_removal() -> None:
    """A departed human still bounds trust in scans from before.

    The registry is a living-tanks list — a 0x58 removes the row — but
    the watermark is monotonic session state, so ground scanned before
    the human's last sighting keeps aging out once.
    """
    departed = _NOW - FORAGE_COVERAGE_TTL_MS * 3
    ws = _service({"900": _tank(900, "Sigma", departed)})
    first = ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS)
    ws.world_state = WorldStateDict(**{**ws.world_state, "tanks": {}})

    second = ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS)

    assert first == departed
    assert second == departed
    assert ws.last_foreign_human_seen_ms == departed


def test_the_watermark_is_monotonic_across_sightings() -> None:
    """A fresher sighting advances it; an older one never regresses it."""
    ws = _service({"900": _tank(900, "Sigma", _NOW - 5000)})
    ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS)
    ws.world_state = WorldStateDict(
        **{**ws.world_state, "tanks": {"901": _tank(901, "Yuppler", _NOW - 9000)}}
    )

    ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS)

    assert ws.last_foreign_human_seen_ms == _NOW - 5000


def test_a_missing_self_state_still_sweeps() -> None:
    """Pre-join, every human-named tank is foreign by definition."""
    ws = WorldService()
    ws.world_state = WorldStateDict(
        **{**ws.world_state, "tanks": {"900": _tank(900, "Sigma", _NOW)}}
    )

    assert ws.knowledge_floor_ms(_NOW, FORAGE_COVERAGE_TTL_MS) == _NOW - FORAGE_COVERAGE_TTL_MS
