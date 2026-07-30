"""Tests for the damage-aware engagement break."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_break import (
    assess_engagement_break,
    estimate_hits_to_kill,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict, make_enemy_threat
from tankpit_bot.inventory import InventoryState
from tankpit_bot.ledger.damage_book import (
    confirm_incoming_damage,
    record_incoming_shot,
)
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.state.types import TankStateDict, make_container_state, make_tank_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _threat(*, damage_state: int, rank: int = 1) -> EnemyThreatDict:
    return make_enemy_threat(
        tank_id=50,
        x=120,
        y=100,
        distance=20,
        damage_state=damage_state,
        rank=rank,
        team=2,
        name="Runner",
        is_bot=True,
    )


def _ctx(*, fuel: int, inventory: InventoryState | None = None) -> DecideCtx:
    world, self_state = make_world(fuel=fuel)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory if inventory is not None else make_inventory(),
        100000,
        None,
        "",
    )


def test_hits_to_kill_scales_with_the_quartile() -> None:
    """Near-death rounds to 4 dual hits at private cap; full health to 13."""
    ctx = _ctx(fuel=1100)
    assert estimate_hits_to_kill(ctx, _threat(damage_state=0)) == 4
    assert estimate_hits_to_kill(ctx, _threat(damage_state=3)) == 13


def test_hits_to_kill_falls_back_to_homing_damage_without_duals() -> None:
    """At zero duals the per-hit damage is homing's 45."""
    ctx = _ctx(fuel=1100, inventory=make_inventory(dual_count=0))
    assert estimate_hits_to_kill(ctx, _threat(damage_state=0)) == 7


def test_full_health_target_under_sustained_fire_breaks() -> None:
    """13 hits to kill at 100/tick incoming projects a strand -> break."""
    assessment = assess_engagement_break(_ctx(fuel=1100), _threat(damage_state=3), 5, 500)
    assert assessment["incoming_rate_per_tick"] == 100
    assert assessment["hits_to_kill"] == 13
    assert assessment["projected_fuel_at_kill"] == 1100 - 13 * 120
    assert assessment["escape_floor"] == 200 + 100 + 200
    assert assessment["break_engagement"] is True


def test_near_death_target_under_the_same_fire_is_finished() -> None:
    """4 hits to kill projects clear of the floor -> keep fighting."""
    assessment = assess_engagement_break(_ctx(fuel=1100), _threat(damage_state=0), 5, 500)
    assert assessment["hits_to_kill"] == 4
    assert assessment["projected_fuel_at_kill"] == 1100 - 4 * 120
    assert assessment["break_engagement"] is False


def test_quiet_fight_never_breaks_even_at_low_fuel() -> None:
    """Zero measured incoming leaves the break inert."""
    assessment = assess_engagement_break(_ctx(fuel=400), _threat(damage_state=3), 0, 0)
    assert assessment["incoming_rate_per_tick"] == 0
    assert assessment["break_engagement"] is False


def test_a_two_hit_spike_is_not_sustained_fire() -> None:
    """Below the 3-hit floor the rate is discarded entirely."""
    assessment = assess_engagement_break(_ctx(fuel=500), _threat(damage_state=3), 2, 180)
    assert assessment["incoming_rate_per_tick"] == 0
    assert assessment["break_engagement"] is False


def _pursuit_target(*, damage_state: int) -> TankStateDict:
    return make_tank_state(
        tank_id=50,
        x=150,
        y=150,
        team=2,
        rank=1,
        name="Runner",
        is_self=False,
        is_bot=False,
        damage_state=damage_state,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=80000,
    )


def _seed_confirmed_incoming(count: int) -> None:
    """Confirm ``count`` dual hits into the live world-service book."""
    book = get_world_service().damage_book
    for i in range(count):
        ts = 95000 + i * 1000
        record_incoming_shot(book, 60, "ganker", 1, ts)
        confirm_incoming_damage(book, -90, ts + 100)


def _engage_ctx(*, fuel: int) -> DecideCtx:
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(damage_state=3)}
    world, self_state = make_world(
        fuel=fuel,
        tanks=tanks,
        containers={
            "140,100": make_container_state(
                x=140,
                y=100,
                is_fuel=True,
                volume=700,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "last_shot_target_id": 50,
        }
    )
    return DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
    )


def test_engage_breaks_under_sustained_fire_and_keeps_the_lock() -> None:
    """A losing pursuit hands the tick to refuel with the lock held.

    The bot-20260728-075336 shape: healthy fleeing target, ~90/tick
    measured incoming. The break fires, COLLECT's larder step aims the
    escape at the remembered 700-fuel container, and the combat lock
    survives the detour (never-drop).
    """
    reset_world_state()
    _seed_confirmed_incoming(5)
    try:
        decision = decide_hunt_mode(_engage_ctx(fuel=800))
    finally:
        reset_world_state()

    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 140
    assert decision["command"]["target_y"] == 100
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_engage_keeps_fighting_when_no_fire_is_measured() -> None:
    """The identical pursuit with an empty damage book still shoots."""
    reset_world_state()
    try:
        decision = decide_hunt_mode(_engage_ctx(fuel=800))
    finally:
        reset_world_state()

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"


def test_break_sets_the_escape_latch_on_the_delegated_decision() -> None:
    """The break decision carries ``break_escape_until_fuel`` = floor.

    The latch is what makes the break STICK across ticks -- without it
    the projection's sliding hit window oscillated into the 21:59
    map-fire loop (map_open deferred by the escape hop, closed by the
    next tick's shot, reopened by the next break, 27-36 fuel/tick).
    """
    reset_world_state()
    _seed_confirmed_incoming(5)
    try:
        decision = decide_hunt_mode(_engage_ctx(fuel=800))
    finally:
        reset_world_state()

    assert decision["updated_ai_state"]["break_escape_until_fuel"] > 0


def test_latched_break_escapes_even_when_the_projection_recovers() -> None:
    """An active latch keeps escaping with NO measured incoming fire.

    The oscillation regression: tick N broke and started the escape;
    tick N+1's empty damage book would have re-approved the fight.
    With the latch (fuel below the stored floor) the tick must stay
    on the escape -- same fuel-hop shape, lock still held.
    """
    reset_world_state()
    try:
        base_ctx = _engage_ctx(fuel=300)
        latched_ctx = DecideCtx(
            base_ctx.world,
            base_ctx.self_state,
            AIStateDict(**{**base_ctx.ai_state, "break_escape_until_fuel": 372}),
            base_ctx.inventory,
            base_ctx.timestamp_ms,
            base_ctx.terrain,
            base_ctx.combat_feedback,
            base_ctx.map_fuel_dots,
        )
        decision = decide_hunt_mode(latched_ctx)
    finally:
        reset_world_state()

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 372


def test_break_latch_releases_when_fuel_recovers_to_the_floor() -> None:
    """Fuel at the stored floor clears the latch and the fight resumes."""
    reset_world_state()
    try:
        base_ctx = _engage_ctx(fuel=800)
        latched_ctx = DecideCtx(
            base_ctx.world,
            base_ctx.self_state,
            AIStateDict(**{**base_ctx.ai_state, "break_escape_until_fuel": 372}),
            base_ctx.inventory,
            base_ctx.timestamp_ms,
            base_ctx.terrain,
            base_ctx.combat_feedback,
            base_ctx.map_fuel_dots,
        )
        decision = decide_hunt_mode(latched_ctx)
    finally:
        reset_world_state()

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 0
