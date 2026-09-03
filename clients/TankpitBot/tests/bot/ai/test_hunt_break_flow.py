"""The break latch through HUNT: leave, stay left, come back on fuel.

The hunt-mode flow around ``assess_engagement_break`` (whose
projection arithmetic is pinned in ``test_combat_break.py``): the
delegated refuel keeps the lock (never-drop), the escape latch
outlives a projection that recovers mid-flight, the close phase
honors it, and the latch releases only at the human-resume fuel
floor. Split from ``test_combat_break.py`` 2026-09-03 at the
600-line ceiling — helpers here build FULL hunt contexts (world,
registry, containers), where the projection file's helpers build
bare arithmetic fixtures.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_container_state, make_tank_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
    seed_confirmed_incoming,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _pursuit_target(*, damage_state: int) -> TankStateDict:
    return make_tank_state(
        tank_id=50,
        x=150,
        y=150,
        team=2,
        rank=1,
        name="red-7",
        is_self=False,
        is_bot=True,
        damage_state=damage_state,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        # Left the viewport 8 s ago -- inside the ~12 s homing trace
        # ([[shoot-event-format]]#reroute-ttl-ms), so pursuit fire is
        # still live; the trace-expired behavior has its own pin.
        last_viewport_observation_ms=92000,
    )


def _engage_ctx(*, fuel: int, damage_state: int = 3) -> DecideCtx:
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {"50": _pursuit_target(damage_state=damage_state)}
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
        ws=ws,
    )


def test_engage_blocks_the_unwinnable_two_attacker_fight() -> None:
    """A fight no fuel level can fund is BLOCKED, not endlessly sortied.

    The bot-20260728-075336 shape: healthy fleeing target, ~90/tick
    measured incoming. Making the break projection whole would need
    more fuel than the tank can hold (fuel_at_break + shortfall >=
    capacity), so escaping-and-returning would loop forever -- the
    break blocks the target with the standard TTL and replans.
    """
    ctx = _engage_ctx(fuel=800)
    seed_confirmed_incoming(ctx.ws, 5)
    decision = decide_hunt_mode(ctx)

    assert decision["updated_ai_state"]["combat_target_id"] == -1
    assert "50" in decision["updated_ai_state"]["blocked_combat_targets"]


def test_engage_breaks_under_moderate_fire_and_keeps_the_lock() -> None:
    """A fundable losing fight hands the tick to refuel with the lock held.

    Moderate incoming (27/tick, the practice-room single-attacker
    rate): the projection fails at the current fuel but recovering the
    shortfall fits inside the tank, so the break escapes to the
    remembered 700-fuel larder container with the lock held
    (never-drop) and latches the release level.
    """
    ctx = _engage_ctx(fuel=500, damage_state=1)
    seed_confirmed_incoming(ctx.ws, 5, weapon=0, damage=-45)
    decision = decide_hunt_mode(ctx)

    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 140
    assert decision["command"]["target_y"] == 100
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_engage_keeps_fighting_when_no_fire_is_measured() -> None:
    """The identical pursuit with an empty damage book still shoots."""
    decision = decide_hunt_mode(_engage_ctx(fuel=800))

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["behavior"]["reason_kind"] == "shoot_target"


def test_break_sets_the_escape_latch_on_the_delegated_decision() -> None:
    """The break decision carries ``break_escape_until_fuel`` = floor.

    The latch is what makes the break STICK across ticks -- without it
    the projection's sliding hit window oscillated into the 21:59
    map-fire loop (map_open deferred by the escape hop, closed by the
    next tick's shot, reopened by the next break, 27-36 fuel/tick).
    """
    ctx = _engage_ctx(fuel=500, damage_state=1)
    seed_confirmed_incoming(ctx.ws, 5, weapon=0, damage=-45)
    decision = decide_hunt_mode(ctx)

    # Release = fuel_at_break + (floor - projected): the level at which
    # the SAME projection clears. A bare-floor release was a zero-width
    # band (current fuel at break is usually above the floor already,
    # live receipts 23:23:45-55: three instant releases in ten
    # seconds), so the latch must exceed the fuel at break time.
    assert decision["updated_ai_state"]["break_escape_until_fuel"] > 500


def test_latched_break_escapes_even_when_the_projection_recovers() -> None:
    """An active latch keeps escaping with NO measured incoming fire.

    The oscillation regression: tick N broke and started the escape;
    tick N+1's empty damage book would have re-approved the fight.
    With the latch (fuel below the stored floor) the tick must stay
    on the escape -- same fuel-hop shape, lock still held.
    """
    ws = WorldService()
    base_ctx = _engage_ctx(fuel=300)
    ws.map_fuel_dots = base_ctx.map_fuel_dots
    latched_ctx = DecideCtx(
        base_ctx.world,
        base_ctx.self_state,
        AIStateDict(**{**base_ctx.ai_state, "break_escape_until_fuel": 372}),
        base_ctx.inventory,
        base_ctx.timestamp_ms,
        base_ctx.terrain,
        base_ctx.combat_feedback,
        ws=ws,
    )
    decision = decide_hunt_mode(latched_ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 372


def test_latched_close_phase_stays_on_the_escape() -> None:
    """A holding latch gates CLOSE ticks too (flag s2-7, bot-20260730-000030).

    The latch check used to live only on the ENGAGE path, so a CLOSE
    tick kept trading shots with orange-8 while the escape's larder
    hop deferred for a map open -- four shoot/map_open cycles, fuel
    572->462 under fire beside the minefield. The entry gate hands
    every latched tick to the lock-held refuel regardless of phase,
    so the deferred hop re-dispatches against the opened map on the
    very next tick.
    """
    ws = WorldService()
    visible_target = make_tank_state(
        tank_id=50,
        x=150,
        y=150,
        team=2,
        rank=1,
        name="Runner",
        is_self=False,
        is_bot=False,
        damage_state=3,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=100000,
    )
    world, self_state = make_world(
        fuel=300,
        tanks={"50": visible_target},
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
            "mode_state": "CLOSE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "break_escape_until_fuel": 372,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )
    decision = decide_hunt_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["command"]["cmd_type"] != "shoot"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 372


def test_latch_without_a_lock_does_not_hijack_the_tick() -> None:
    """Latch active but no combat lock: the tick runs its phase normally.

    The latch's job is finishing an escape from a LOCKED fight; once
    the lock is gone the phases proceed (here: acquisition opens the
    map) and the latch simply waits for its fuel release.
    """
    ws = WorldService()
    world, self_state = make_world(fuel=300)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "break_escape_until_fuel": 372,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        None,
        "",
        ws=ws,
    )
    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 372


def test_close_phase_fight_under_fire_breaks_at_entry() -> None:
    """The break assessment gates CLOSE ticks too — the Artax death pin.

    Run bot-20260730-004144, 01:06:55: the whole Yuppler fight ran in
    CLOSE phase, where no break assessment existed — the first break
    of the fight fired at fuel 216, four seconds before deactivation
    ("he just stood there and tanked like 4 shots"). With the
    assessment at HUNT entry, the same fight shape breaks on the
    first sustained-fire window while escape is still cheap.
    """
    ws = WorldService()
    seed_confirmed_incoming(ws, 5, weapon=0, damage=-45)
    base_ctx = _engage_ctx(fuel=500, damage_state=1)
    ws.map_fuel_dots = base_ctx.map_fuel_dots
    close_ctx = DecideCtx(
        base_ctx.world,
        base_ctx.self_state,
        AIStateDict(**{**base_ctx.ai_state, "mode_state": "CLOSE"}),
        base_ctx.inventory,
        base_ctx.timestamp_ms,
        base_ctx.terrain,
        base_ctx.combat_feedback,
        ws=ws,
    )
    decision = decide_hunt_mode(close_ctx)

    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["break_escape_until_fuel"] > 500


def _human_pursuit_tank() -> TankStateDict:
    return make_tank_state(
        tank_id=50,
        x=150,
        y=150,
        team=2,
        rank=1,
        name="Yuppler",
        is_self=False,
        is_bot=False,
        damage_state=3,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=80000,
    )


def test_unwinnable_human_fight_refuels_and_keeps_the_lock() -> None:
    """A human fight is never blocked as unwinnable — refuel and resume.

    User ruling 2026-07-30 after the Artax death ("the bot can fight
    against a human and win... it should have collected fuel and then
    kept fighting and collected as necessary"): the one-kill
    projection that condemns a practice-bot fight does not apply to
    attrition fights against humans. Past-capacity projections latch
    the RESUME floor (2026-07-31: refuel just enough to fund re-entry,
    never a full-tank restock trip) and escape to fuel with the lock
    held. Fuel 500 sits below the rank-2 half-capacity band (600), so
    the break is live.
    """
    base_ctx = _engage_ctx(fuel=500)
    seed_confirmed_incoming(base_ctx.ws, 5)
    base_ctx.world["tanks"]["50"] = _human_pursuit_tank()
    decision = decide_hunt_mode(base_ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    # The resume floor at defaults: max(200 + 100 + 450, 1200 // 2 +
    # 100) = 750 — one good container away, then back in the fight.
    # capacity//2 + rank-scaled hunt reserve at rank 2: 600 + 85.
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 685


def test_human_fight_above_half_capacity_keeps_fighting() -> None:
    """The break band holds a human fight in HUNT at healthy fuel.

    Same fire and target as the latch test, but at fuel 800 (above
    the rank-2 band of 600): no break fires, no COLLECT delegation --
    the tick stays a HUNT decision on the held lock (user ruling
    2026-07-31: "it dont really hunt to kill, it kinda just does
    damage then leaves").
    """
    base_ctx = _engage_ctx(fuel=800)
    seed_confirmed_incoming(base_ctx.ws, 5)
    base_ctx.world["tanks"]["50"] = _human_pursuit_tank()
    decision = decide_hunt_mode(base_ctx)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["combat_target_id"] == 50
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 0


def test_close_phase_unwinnable_fight_blocks_at_entry() -> None:
    """A fight no fuel level can fund is blocked from CLOSE ticks too."""
    ws = WorldService()
    seed_confirmed_incoming(ws, 5)
    base_ctx = _engage_ctx(fuel=800)
    ws.map_fuel_dots = base_ctx.map_fuel_dots
    close_ctx = DecideCtx(
        base_ctx.world,
        base_ctx.self_state,
        AIStateDict(**{**base_ctx.ai_state, "mode_state": "CLOSE"}),
        base_ctx.inventory,
        base_ctx.timestamp_ms,
        base_ctx.terrain,
        base_ctx.combat_feedback,
        ws=ws,
    )
    decision = decide_hunt_mode(close_ctx)

    assert decision["updated_ai_state"]["combat_target_id"] == -1
    assert "50" in decision["updated_ai_state"]["blocked_combat_targets"]


def test_break_latch_releases_when_fuel_recovers_to_the_floor() -> None:
    """Fuel at the stored floor clears the latch and the fight resumes."""
    ws = WorldService()
    base_ctx = _engage_ctx(fuel=800)
    ws.map_fuel_dots = base_ctx.map_fuel_dots
    latched_ctx = DecideCtx(
        base_ctx.world,
        base_ctx.self_state,
        AIStateDict(**{**base_ctx.ai_state, "break_escape_until_fuel": 372}),
        base_ctx.inventory,
        base_ctx.timestamp_ms,
        base_ctx.terrain,
        base_ctx.combat_feedback,
        ws=ws,
    )
    decision = decide_hunt_mode(latched_ctx)

    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["updated_ai_state"]["break_escape_until_fuel"] == 0
