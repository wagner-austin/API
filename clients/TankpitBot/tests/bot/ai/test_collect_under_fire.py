"""Tests for the under-fire escape rungs of the collect cascade."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.types.modes import AIMode, AIModeState


def test_trapped_escape_walks_instead_of_paying_for_an_unsafe_hop() -> None:
    """A landing beside the attacker cannot bypass candidate safety."""
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
    from tankpit_bot.bot.ai.context import DecideCtx
    from tankpit_bot.ledger.damage_book import confirm_incoming_damage, record_incoming_shot
    from tankpit_bot.state.types import make_container_state
    from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    book = ws.damage_book
    for i in range(4):
        ts = 95000 + i * 1000
        record_incoming_shot(book, 60, "Yuppler", 1, ts)
        confirm_incoming_damage(book, -90, ts + 100)
    world, self_state = make_world(
        fuel=800,
        containers={
            "110,100": make_container_state(
                x=110,
                y=100,
                is_fuel=True,
                volume=400,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": AIMode.COLLECT,
            "mode_state": AIModeState.SEARCH,
            "mode_started_ms": 90000,
            "last_map_open_ms": 99000,
            "combat_target_id": 50,
            "combat_target_x": 112,
            "combat_target_y": 100,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        CombatFeedback.NONE,
        ws=ws,
    )
    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected trapped-escape movement decision")
    assert decision["command"]["cmd_type"] == "move"
    assert decision["updated_ai_state"]["combat_target_id"] == 50


class TestEscapePlanContinuity:
    """Tests for committed-plan continuity inside the under-fire branch."""

    def test_equipment_plan_on_own_tile_is_finished_not_rederived(self) -> None:
        """Standing on the locked equipment under fire dispatches the pickup.

        Flag s8-2 (run bot-20260730-025337, 03:00:00): the escape hop
        landed ON its locked equipment and the next derivation
        re-selected a teleport to the tile the tank was standing on,
        burning a map-open tick. The committed plan completes here, so
        the pickup is the escape continuation.
        """
        from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.intent import set_resource_target
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.state.types import make_container_state
        from tests.bot.ai._support import (
            make_inventory,
            make_scanned_ai_state,
            make_world,
            seed_confirmed_incoming,
        )
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        seed_confirmed_incoming(ws, 3)
        world, self_state = make_world(
            fuel=668,
            containers={
                "100,100": make_container_state(
                    x=100,
                    y=100,
                    is_fuel=False,
                    volume=0,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
            },
        )
        ai_state = AIStateDict(
            **{
                **set_resource_target(make_scanned_ai_state(), "equipment", 100, 100),
                "mode": AIMode.COLLECT,
                "mode_state": AIModeState.SEARCH,
                "mode_started_ms": 90000,
                "suppress_landing_scan": False,
            }
        )
        inventory = make_inventory(dual_count=3, default_count=30)
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            InMemoryTerrainMap(),
            CombatFeedback.NONE,
            ws=ws,
        )

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected a decision from the under-fire branch")
        assert decision["command"]["cmd_type"] == "pickup_equipment"
        assert decision["behavior"]["reason_kind"] == "equipment_locked"

    def test_fuel_plan_at_cardinal_reach_is_finished_under_fire(self) -> None:
        """A fuel plan one tile away under fire dispatches its pickup."""
        from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.intent import set_resource_target
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.state.types import make_container_state
        from tests.bot.ai._support import (
            make_inventory,
            make_scanned_ai_state,
            make_world,
            seed_confirmed_incoming,
        )
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        seed_confirmed_incoming(ws, 3)
        world, self_state = make_world(
            fuel=668,
            containers={
                "101,100": make_container_state(
                    x=101,
                    y=100,
                    is_fuel=True,
                    volume=400,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
            },
        )
        ai_state = AIStateDict(
            **{
                **set_resource_target(make_scanned_ai_state(), "fuel", 101, 100),
                "mode": AIMode.COLLECT,
                "mode_state": AIModeState.SEARCH,
                "mode_started_ms": 90000,
                "suppress_landing_scan": False,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            CombatFeedback.NONE,
            ws=ws,
        )

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected a decision from the under-fire branch")
        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["behavior"]["reason_kind"] == "fuel_locked"

    def test_far_plan_does_not_hijack_the_escape(self) -> None:
        """A plan out of serve reach leaves the escape rungs in charge.

        The continuity gate is completion-only: a lock the tank would
        have to TRAVEL to must not preempt the under-fire walk law, so
        the in-viewport fuel walk still wins the tick.
        """
        from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.intent import set_resource_target
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.state.types import make_container_state
        from tests.bot.ai._support import (
            make_inventory,
            make_scanned_ai_state,
            make_world,
            seed_confirmed_incoming,
        )
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        seed_confirmed_incoming(ws, 3)
        world, self_state = make_world(
            fuel=668,
            containers={
                "106,100": make_container_state(
                    x=106,
                    y=100,
                    is_fuel=False,
                    volume=0,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
                "101,100": make_container_state(
                    x=101,
                    y=100,
                    is_fuel=True,
                    volume=400,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
            },
        )
        ai_state = AIStateDict(
            **{
                **set_resource_target(make_scanned_ai_state(), "equipment", 106, 100),
                "mode": AIMode.COLLECT,
                "mode_state": AIModeState.SEARCH,
                "mode_started_ms": 90000,
                "suppress_landing_scan": False,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(dual_count=3, default_count=30),
            100000,
            InMemoryTerrainMap(),
            CombatFeedback.NONE,
            ws=ws,
        )

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected a decision from the under-fire branch")
        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["behavior"]["reason_kind"] == "fuel_collect"


class TestMovementDeadEscape:
    """Tests for the movement-dead walk-rung skip under fire."""

    def test_rejected_movement_skips_the_walk_and_hops_out(self) -> None:
        """Two cant_go refusals in-window kill the walk rung.

        Run bot-20260730-110x ticks 95-107: twelve consecutive
        rejected walk-pickups under purple-1's fire (every direction
        refused, fuel 972->663) because the escape kept planning
        walks the server would not allow. With the movement-dead
        floor met, the in-viewport walkable fuel is skipped and the
        larder hop takes the tick.
        """
        from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.state.types import make_container_state
        from tests.bot.ai._support import (
            make_inventory,
            make_scanned_ai_state,
            make_world,
            seed_confirmed_incoming,
        )
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        seed_confirmed_incoming(ws, 3)
        ws.record_movement_rejection(96000)
        ws.record_movement_rejection(98000)
        world, self_state = make_world(
            fuel=668,
            containers={
                "101,100": make_container_state(
                    x=101,
                    y=100,
                    is_fuel=True,
                    volume=400,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
                "140,100": make_container_state(
                    x=140,
                    y=100,
                    is_fuel=True,
                    volume=500,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": AIMode.COLLECT,
                "mode_state": AIModeState.SEARCH,
                "mode_started_ms": 90000,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            CombatFeedback.NONE,
            ws=ws,
        )

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected a decision from the under-fire branch")
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["reason_kind"] == "fuel_hop"

    def test_single_rejection_keeps_the_walk_rung(self) -> None:
        """Below the movement-dead floor the walk law stands."""
        from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.state.types import make_container_state
        from tests.bot.ai._support import (
            make_inventory,
            make_scanned_ai_state,
            make_world,
            seed_confirmed_incoming,
        )
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        ws = WorldService()
        seed_confirmed_incoming(ws, 3)
        ws.record_movement_rejection(98000)
        world, self_state = make_world(
            fuel=668,
            containers={
                "101,100": make_container_state(
                    x=101,
                    y=100,
                    is_fuel=True,
                    volume=400,
                    timestamp_ms=100000,
                    failed_pickups=0,
                ),
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": AIMode.COLLECT,
                "mode_state": AIModeState.SEARCH,
                "mode_started_ms": 90000,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            CombatFeedback.NONE,
            ws=ws,
        )

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected a decision from the under-fire branch")
        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["behavior"]["reason_kind"] == "fuel_collect"


def test_under_fire_with_nothing_available_falls_to_the_exhausted_outcome() -> None:
    """Sustained fire with nothing to do yields the tick to hunt.

    Fully stocked (hunt-entry permitted), fuel at capacity, no larder,
    and the only fuel dot's landing viewport overlaps live coverage —
    every escape verb declines, the trapped fallback has nothing to
    fall back to, and the under-fire branch resolves through the
    exhausted outcome: ``None``, handing the tick to the hunt owner.
    """
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
    from tankpit_bot.bot.ai.context import DecideCtx
    from tests.bot.ai._support import (
        make_inventory,
        make_scanned_ai_state,
        make_world,
        seed_confirmed_incoming,
    )

    ws = WorldService()
    ws.map_fuel_dots = ((101, 101),)
    seed_confirmed_incoming(ws, 3)
    world, self_state = make_world(fuel=1200)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(dual_count=30, default_count=30),
        100000,
        None,
        CombatFeedback.NONE,
        ws=ws,
    )

    assert decide_collect_mode(ctx) is None
