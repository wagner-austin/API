"""Unlimited-distance human pursuit and the stale-human map refresh."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import (
    consent_human,
    make_inventory,
    make_map_known_enemy,
    make_scanned_ai_state,
    make_world,
)


def test_unaffordable_human_outranks_affordable_bot_at_acquisition() -> None:
    """A rank-window human beyond the horizon preempts nearby bot farming.

    User ruling 2026-07-29 ("unlimited distance for humans... this is
    the real deal"), born from the Yuppler encounter: Yuppler at dist
    95 was rejected ``unaffordable`` while the bot farmed red-3 at
    dist 19. With an affordable practice bot AND an unaffordable
    human on the fresh map, the decision must be a dot-relay leg
    toward the HUMAN, not a teleport at the bot.
    """
    ws = WorldService()
    consent_human(ws, 90)
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
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
        # (150,100) closes distance to Yuppler and is affordable.
        ((150, 100),),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"


def test_human_pursuit_falls_back_to_bot_when_no_leg_helps() -> None:
    """With no progress dot and a full tank, the bot farms while waiting.

    The pursuit must not deadlock the session: when no dot closes
    distance to the human and refuel-in-place is pointless (already
    at capacity), the affordable bot is engaged and the next map
    re-evaluates the pursuit.
    """
    ws = WorldService()
    consent_human(ws, 90)
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


def test_recruit_human_is_not_pursued() -> None:
    """A rank-0 human stays protected -- no relay chain toward them.

    The rank window rejects recruits BEFORE the affordability gate
    (reason ``protected_human_rank``, not ``unaffordable``), so the
    pursuit helper can never travel toward one and the affordable bot
    is farmed normally.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_tank_state(
            tank_id=90,
            x=240,
            y=100,
            team=2,
            rank=0,
            name="Yuppler",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=99800,
        ),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
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
        ((150, 100),),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


def test_locked_human_beyond_funds_relays_with_lock_held() -> None:
    """A locked human who teleported beyond funds is chased leg by leg.

    User ruling 2026-07-29: "even if they teleport super far away."
    The return costs 840 + the 650 engagement floor at fuel 700, so
    the plain resume cannot fund it -- the decision must be a relay
    leg toward the human with ``combat_target_id`` retained
    (never-drop rides through every leg).
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=700, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "combat_target_id": 90,
            "combat_target_x": 240,
            "combat_target_y": 100,
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
        ((150, 100),),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"
    assert decision["updated_ai_state"]["combat_target_id"] == 90


def test_locked_bot_beyond_funds_still_refuels_in_place() -> None:
    """The relay-resume is human-only: a bot lock keeps the plain refuel.

    Practice bots never flee across the map, so the 2026-07-27
    refuel-then-resume (get richer in place, return when fundable)
    remains the right shape for them -- guards the ``is_human_name``
    gate on the new relay branch.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="red-9"),
    }
    world, self_state = make_world(fuel=700, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "combat_target_id": 90,
            "combat_target_x": 240,
            "combat_target_y": 100,
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
        ((150, 100),),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["combat_target_id"] == 90


def test_locked_human_with_no_relay_leg_falls_back_to_refuel() -> None:
    """When no relay leg helps, the locked-human resume uses plain refuel.

    At fuel capacity with no progress dot, ``_relay_toward`` returns
    ``None`` (refuel-in-place is pointless at a full tank), so the
    resume falls through to the 2026-07-27 refuel-for-hunt path --
    whose collect cascade also declines at capacity and terminates
    via the blocked-target replan rather than deadlocking the tick.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "combat_target_id": 90,
            "combat_target_x": 240,
            "combat_target_y": 100,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    # The tick advances (no relay teleport was possible); the exact
    # fallback shape is refuel_for_hunt's contract, not re-tested here.
    assert decision["behavior"]["reason_kind"] != "dot_relay"


def test_relay_leg_cost_is_capped_at_the_engagement_budget() -> None:
    """A max-progress dot costing more than one kill budget is skipped.

    Regression for the 2026-07-29 21:17:40 broke-arrival: the uncapped
    picker paid 787 fuel in one leg (1100 -> 313) and landed next to
    Yuppler unable to fight, stranding the pursuit in a minutes-long
    restock. The near-enemy dot here costs ~780 (affordable under the
    old floor-only rule at a full tank) and must lose to the cheaper
    progress dot at ~300.
    """
    ws = WorldService()
    ws.map_data_ingested_ms = 99500  # data heard 500 ms ago: honestly fresh
    consent_human(ws, 90)
    tanks: dict[str, TankStateDict] = {
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
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
        # (230,100): most progress, cost ~780 -- beyond the 450 leg cap.
        # (150,100): cost 300 -- the correct capped leg.
        ((230, 100), (150, 100)),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 150
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"


def test_stale_known_human_forces_a_map_refresh_over_bot_farming() -> None:
    """A map-stale rank-window human is refreshed before settling for bots.

    The freshness asymmetry that hid Yuppler (2026-07-29 21:19):
    practice bots stay wire-fresh by moving; a quiet human goes stale
    ``map_open_cooldown_ms`` after every map open, and with a fresh
    bot always available acquisition never reopened the map. Here the
    bot is map-fresh via the wire, the human's timestamp has aged
    out, and the map itself is older than the cooldown -- the
    decision must be a map refresh, not a teleport at red-5.
    """
    ws = WorldService()
    consent_human(ws, 90)
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler", timestamp_ms=80000),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 90000,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "map_open"


def test_fresh_map_showing_stale_human_farms_normally() -> None:
    """A FRESH map that still shows the human stale means they left.

    No refresh can cure a human absent from the latest snapshot, so
    the affordable bot is engaged -- the refresh rule cannot loop.
    """
    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler", timestamp_ms=80000),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "", ws=ws)

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "teleport_target"
    assert decision["updated_ai_state"]["combat_target_id"] == 60


def test_stale_human_exists_filters_and_reasons() -> None:
    """The stale-human predicate skips allies, bots, and non-stale humans.

    Direct contract test: an allied human (same team), a wire-fresh
    practice bot, and a BLOCKED human must all fail to trigger the
    refresh; only a human whose sole curable defect is stale map data
    returns True.
    """
    ws = WorldService()
    consent_human(ws, 90)
    from tankpit_bot.bot.ai.threat_acquisition import stale_human_exists

    ally_human = make_tank_state(
        tank_id=70,
        x=110,
        y=100,
        team=1,
        rank=2,
        name="FriendlyHuman",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=80000,
    )
    fresh_bot = make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5")
    blocked_human = make_map_known_enemy(tank_id=80, x=200, y=100, name="Blocked")
    stale_human = make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler", timestamp_ms=80000)

    def check(tanks: dict[str, TankStateDict], blocked: dict[str, int]) -> bool:
        world, self_state = make_world(fuel=1100, tanks=tanks)
        return stale_human_exists(
            ws,
            world,
            self_state,
            blocked,
            {},
            None,
            100000,
            5000,
            engagement_reserve_fuel=650,
        )

    assert check({"70": ally_human, "60": fresh_bot}, {}) is False
    assert check({"80": blocked_human}, {"80": 100000}) is False
    assert check({"90": stale_human}, {}) is True


def test_relay_skips_progress_dot_below_the_fuel_floor() -> None:
    """A capped-cost dot that would dip below the reserve is skipped.

    At fuel 400, the 300-cost progress dot passes the 450 leg cap but
    would leave 100 < the 200 floor -- the cheaper dot wins instead.
    """
    ws = WorldService()
    ws.map_data_ingested_ms = 99500  # data heard 500 ms ago: honestly fresh
    consent_human(ws, 90)
    tanks: dict[str, TankStateDict] = {
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=400, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
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
        ((150, 100), (110, 100)),
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 110
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "dot_relay"
