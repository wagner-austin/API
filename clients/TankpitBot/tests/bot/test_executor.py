"""Tests for executor module: apply_equipment and dispatch_command."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.bot.ai.types import AIStateDict, make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.executor import (
    _format_desired_equipment,
    _is_dispatchable,
    _is_valid_move_destination,
    _is_valid_pickup,
    _is_valid_shoot,
    _is_valid_teleport,
    _tracked_combat_target,
    _tracked_container,
    _tracked_resource_target,
    _tracked_tank,
    apply_equipment,
    dispatch_command,
    execute,
)
from tankpit_bot.bot.tick_loop_types import make_tick_decision
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.sniffer.world_state import (
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_toggle
from tankpit_bot.state import (
    WorldStateDict,
    add_mine_from_radar,
    make_container_state,
    make_empty_world_state,
    make_self_state,
    make_tank_state,
)
from tests.conftest import FakeEnv
from tests.fakes import FakeCDPSession


def _make_bot(fake_env: FakeEnv) -> tuple[Bot, FakeCDPSession]:
    """Create a Bot with FakeCDPSession in IDLE state."""
    reset_world_state()
    update_world_state_from_position(100, 100)
    update_world_state_from_fuel_total(800)
    bot = Bot("https://test.tankpit.com/", headless=True)
    fake_cdp = FakeCDPSession()
    bot._cdp = fake_cdp
    bot._state_data = bot._state_data.copy()
    bot._state_data["state"] = "IDLE"
    return bot, fake_cdp


def _store_tank(
    tank_id: int,
    *,
    x: int,
    y: int,
    source: Literal["viewport", "radar", "world_state"],
) -> None:
    """Store a tracked tank directly into world state for executor tests."""
    import tankpit_bot.sniffer.world_state as ws

    new_tanks = dict(ws._world_state["tanks"])
    new_tanks[str(tank_id)] = make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=1,
        rank=0,
        damage_state=0,
        name="enemy",
        is_bot=True,
        is_self=False,
        source=source,
        timestamp_ms=1000,
    )
    ws._world_state["tanks"] = new_tanks


def _make_world() -> WorldStateDict:
    """Create a strict world-state fixture for executor helper tests."""
    world = make_empty_world_state()
    return WorldStateDict(
        **{
            **world,
            "self_state": make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=0,
                rank=1,
                fuel=800,
                leaderboard_position=1,
            ),
            "timestamp_ms": 1000,
        }
    )


class _WorldOnlyBot:
    """Minimal bot double for _is_dispatchable helper coverage."""

    def __init__(self, world: WorldStateDict) -> None:
        """Store the provided world-state snapshot."""
        self._world = world
        self._cdp = None
        self._cdp_message_buffer: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        """Return the injected world-state snapshot."""
        return self._world

    def move_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def pickup_fuel_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def pickup_equipment_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def teleport_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def shoot_at(self, x: int, y: int, target_id: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y, target_id)
        return False

    def use_radar(self) -> bool:
        """Unused BotProtocol stub."""
        return False

    def open_map(self) -> bool:
        """Unused BotProtocol stub."""
        return False

    def enable_equipment(self, slot: int) -> bool:
        """Unused BotProtocol stub."""
        _ = slot
        return False

    def disable_equipment(self, slot: int) -> bool:
        """Unused BotProtocol stub."""
        _ = slot
        return False

    def _has_equipment_stock(self, slot: int) -> bool:
        """Unused BotProtocol stub."""
        _ = slot
        return False


class TestApplyEquipment:
    """Tests for apply_equipment."""

    def test_enables_desired_slots(self, fake_env: FakeEnv) -> None:
        """Enables desired combat slots that have stock."""
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_gain

        bot, fake_cdp = _make_bot(fake_env)
        # Give slot 2 (dual) stock so _has_equipment_stock returns True
        update_inventory_from_gain([0, 5, 0, 0, 0])
        # Disable slot 2 via toggle so enable triggers a toggle command
        update_inventory_from_toggle([True, False, True, True, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not desired, enabled → disable → toggle (1 CDP)
        # slot 2: desired, disabled, has stock → enable → toggle (1 CDP)
        # slot 4: not desired, enabled → disable → toggle (1 CDP)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_disables_undesired_slots(self, fake_env: FakeEnv) -> None:
        """Disables combat slots not in desired list."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so we can test disabling
        update_inventory_from_toggle([True, True, True, True, True])
        apply_equipment(bot, [5])
        # Should disable slots 1, 2, 4 (3 CDP calls)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_skips_already_correct_state(self, fake_env: FakeEnv) -> None:
        """No toggles when equipment already matches desired state."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set inventory to match: 1=off, 2=on, 4=off, 5=on
        update_inventory_from_toggle([False, True, True, False, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not in desired, already disabled → no toggle
        # slot 2: in desired, already enabled → no toggle
        # slot 4: not in desired, already disabled → no toggle
        assert len(fake_cdp._sent_methods) == 0


class TestFormatDesiredEquipment:
    """Tests for desired-equipment log formatting."""

    def test_empty_list_formats_as_none(self) -> None:
        """Formats an empty desired-equipment list clearly."""
        assert _format_desired_equipment([]) == "none"

    def test_unknown_slot_uses_slot_fallback(self) -> None:
        """Formats unknown slots with an explicit fallback label."""
        assert _format_desired_equipment([2, 9]) == "dual,slot9"


class TestExecutorHelpers:
    """Focused coverage tests for executor helper branches."""

    def test_tracked_resource_target_returns_none_when_kind_is_empty(self) -> None:
        """Empty resource locks do not resolve to a tracked container."""
        world = _make_world()
        decision = make_tick_decision(
            command=make_move_command(100, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 100, 100, "noop"),
            updated_ai_state=AIStateDict(**{**make_initial_ai_state(), "resource_target_kind": ""}),
            desired_equipment=[],
        )

        assert _tracked_resource_target(world, decision) is None

    def test_valid_teleport_allows_search_hop_without_locked_resource(self) -> None:
        """Search teleports remain valid without a locked resource target."""
        world = _make_world()
        decision = make_tick_decision(
            command=make_teleport_command(120, 120),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 120, 120, "search_fuel_local"),
            updated_ai_state=AIStateDict(**{**make_initial_ai_state(), "resource_target_kind": ""}),
            desired_equipment=[],
        )

        assert _is_valid_teleport(world, decision) is True


class TestDispatchCommand:
    """Tests for dispatch_command."""

    def test_dispatch_move(self, fake_env: FakeEnv) -> None:
        """Dispatches move command via bot.move_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_move_command(150, 160))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_fuel(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_fuel command via bot.pickup_fuel_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_fuel_command(80, 90))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_equipment(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_equipment command via bot.pickup_equipment_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_equipment_command(80, 90))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_shoot(self, fake_env: FakeEnv) -> None:
        """Dispatches shoot command via bot.shoot_at."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_shoot_command(105, 103))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_radar(self, fake_env: FakeEnv) -> None:
        """Dispatches radar command via bot.use_radar."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_radar_command())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_map_open(self, fake_env: FakeEnv) -> None:
        """Dispatches map_open command via bot.open_map."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_map_open_command())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport(self, fake_env: FakeEnv) -> None:
        """Dispatches teleport command via bot.teleport_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_teleport_command(200, 200))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods


class TestExecutorValidationHelpers:
    """Focused tests for executor-side validation helpers."""

    def test_tracked_tank_ignores_non_positive_id(self) -> None:
        """Tracked-tank lookup returns None for invalid ids."""
        assert _tracked_tank(_make_world(), 0) is None

    def test_tracked_container_returns_matching_container(self) -> None:
        """Tracked-container lookup returns the stored container."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )

        container = _tracked_container(world, 104, 100)

        if container is None:
            raise AssertionError("expected tracked container at (104,100)")
        assert container["volume"] == 500

    def test_valid_shoot_rejects_moved_target(self) -> None:
        """Shoot validation rejects targets whose tracked coordinates drifted."""
        world = _make_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=107,
            y=104,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
            source="viewport",
            timestamp_ms=1000,
        )

        result = _is_valid_shoot(world, make_shoot_command(105, 103, 10))

        assert result is False

    def test_valid_shoot_rejects_missing_target(self) -> None:
        """Shoot validation rejects unknown target ids."""
        result = _is_valid_shoot(_make_world(), make_shoot_command(105, 103, 10))

        assert result is False

    def test_valid_shoot_accepts_viewport_target(self) -> None:
        """Shoot validation accepts viewport-fresh tracked targets."""
        world = _make_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=103,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
            source="viewport",
            timestamp_ms=1000,
        )

        result = _is_valid_shoot(world, make_shoot_command(105, 103, 10))

        assert result is True

    def test_valid_shoot_accepts_world_state_target_in_visible_viewport(self) -> None:
        """Shoot validation accepts a world-state target already inside view."""
        world = _make_world()
        world["viewport"]["left"] = 100
        world["viewport"]["top"] = 96
        world["viewport"]["width"] = 16
        world["viewport"]["height"] = 16
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=103,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
            source="world_state",
            timestamp_ms=1000,
        )

        result = _is_valid_shoot(world, make_shoot_command(105, 103, 10))

        assert result is True

    def test_valid_pickup_rejects_wrong_fuel_kind(self) -> None:
        """Fuel pickup validation rejects equipment on the same tile."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=False,
            volume=0,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )

        result = _is_valid_pickup(world, make_pickup_fuel_command(104, 100))

        assert result is False

    def test_valid_pickup_accepts_equipment_target(self) -> None:
        """Equipment pickup validation accepts matching tracked equipment."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=False,
            volume=0,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )

        result = _is_valid_pickup(world, make_pickup_equipment_command(104, 100))

        assert result is True

    def test_valid_pickup_rejects_missing_equipment_target(self) -> None:
        """Equipment pickup validation rejects missing containers."""
        result = _is_valid_pickup(_make_world(), make_pickup_equipment_command(104, 100))

        assert result is False

    def test_valid_pickup_rejects_equipment_on_fuel_tile(self) -> None:
        """Equipment pickup validation rejects fuel on the same tile."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )

        result = _is_valid_pickup(world, make_pickup_equipment_command(104, 100))

        assert result is False

    def test_valid_move_destination_accepts_safe_move(self) -> None:
        """Move validation accepts destinations that are not mined."""
        result = _is_valid_move_destination(_make_world(), make_move_command(104, 100))

        assert result is True

    def test_tracked_combat_target_returns_none_for_mismatch(self) -> None:
        """Combat target lookup rejects stale AI coordinates."""
        world = _make_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=107,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
            source="world_state",
            timestamp_ms=1000,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=make_behavior_score("HUNT", 800, 105, 100, "teleport enemy", target_id=10),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _tracked_combat_target(world, decision) is None

    def test_tracked_combat_target_returns_none_when_tank_missing(self) -> None:
        """Combat target lookup returns None when the tank is absent."""
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=make_behavior_score("HUNT", 800, 105, 100, "teleport enemy", target_id=10),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _tracked_combat_target(_make_world(), decision) is None

    def test_tracked_combat_target_returns_none_without_locked_target(self) -> None:
        """Combat target lookup returns None when no combat target is locked."""
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=make_behavior_score("HUNT", 800, 105, 100, "teleport enemy"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        assert _tracked_combat_target(_make_world(), decision) is None

    def test_tracked_resource_target_rejects_kind_mismatch(self) -> None:
        """Resource target lookup rejects containers with the wrong kind."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=False,
            volume=0,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(103, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 104, 100, "fuel=500"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _tracked_resource_target(world, decision) is None

    def test_tracked_resource_target_rejects_equipment_kind_mismatch(self) -> None:
        """Resource target lookup rejects equipment locks on fuel containers."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "resource_target_kind": "equipment",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(103, 100),
            behavior=make_behavior_score("COLLECT_EQUIPMENT", 900, 104, 100, "equipment_low"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _tracked_resource_target(world, decision) is None

    def test_valid_teleport_accepts_combat_target(self) -> None:
        """Teleport validation accepts tracked combat targets with valid sources."""
        world = _make_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=106,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
            source="world_state",
            timestamp_ms=1000,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=make_behavior_score("HUNT", 800, 105, 100, "teleport enemy", target_id=10),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _is_valid_teleport(world, decision) is True

    def test_valid_teleport_rejects_stale_combat_target(self) -> None:
        """Teleport validation rejects missing locked combat targets."""
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=make_behavior_score("HUNT", 800, 105, 100, "teleport enemy", target_id=10),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _is_valid_teleport(_make_world(), decision) is False

    def test_valid_teleport_allows_hunt_teleport_without_locked_target(self) -> None:
        """HUNT teleports without a locked combat target are allowed through unchanged."""
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=make_behavior_score("HUNT", 800, 105, 100, "teleport enemy"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        assert _is_valid_teleport(_make_world(), decision) is True

    def test_valid_teleport_rejects_stale_resource_target(self) -> None:
        """Teleport validation rejects stale locked resource targets."""
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(103, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 104, 100, "fuel=500"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _is_valid_teleport(_make_world(), decision) is False

    def test_valid_teleport_rejects_invalid_resource_source(self) -> None:
        """Teleport validation rejects resource targets with invalid sources."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="world_state",
            timestamp_ms=1000,
            failed_pickups=0,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(103, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 104, 100, "fuel=500"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _is_valid_teleport(world, decision) is False

    def test_valid_teleport_accepts_resource_target(self) -> None:
        """Teleport validation accepts locked resource targets from radar."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(103, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 104, 100, "fuel=500"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _is_valid_teleport(world, decision) is True

    def test_valid_teleport_ignores_stale_combat_lock_for_fuel_recovery(self) -> None:
        """Fuel recovery teleports validate against resource lock, not stale combat state."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
                "resource_target_kind": "fuel",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_teleport_command(103, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 104, 100, "fuel=500"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        assert _is_valid_teleport(world, decision) is True

    def test_dispatchable_accepts_valid_pickup_decision(self) -> None:
        """Dispatchability accepts fully valid pickup decisions."""
        world = _make_world()
        world["containers"]["104,100"] = make_container_state(
            x=104,
            y=100,
            is_fuel=True,
            volume=500,
            source="radar",
            timestamp_ms=1000,
            failed_pickups=0,
        )
        decision = make_tick_decision(
            command=make_pickup_fuel_command(104, 100),
            behavior=make_behavior_score("COLLECT_FUEL", 900, 104, 100, "fuel=500"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        assert _is_dispatchable(_WorldOnlyBot(world), decision) is True


class TestExecute:
    """Tests for execute (apply_equipment + dispatch_command)."""

    def test_execute_applies_equipment_then_dispatches(self, fake_env: FakeEnv) -> None:
        """Execute applies equipment changes before dispatching command."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so execute needs to disable 1, 2, 4
        update_inventory_from_toggle([True, True, True, True, True])
        behavior = make_behavior_score("HUNT", 50, 100, 200, "patrol_waypoint")
        decision = make_tick_decision(
            command=make_move_command(100, 200),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[5],
        )
        execute(bot, decision)
        # Equipment toggles (disable 1, 2, 4) + move command
        assert len(fake_cdp._sent_methods) == 4

    def test_execute_rejects_stale_non_viewport_shoot(self, fake_env: FakeEnv) -> None:
        """Execute drops shoot commands for non-viewport targets."""
        bot, fake_cdp = _make_bot(fake_env)
        _store_tank(10, x=105, y=103, source="world_state")
        behavior = make_behavior_score("HUNT", 800, 105, 103, "shoot enemy", target_id=10)
        decision = make_tick_decision(
            command=make_shoot_command(105, 103, 10),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision)

        assert fake_cdp._sent_methods == []

    def test_execute_rejects_missing_pickup_target(self, fake_env: FakeEnv) -> None:
        """Execute drops pickup commands when the container no longer exists."""
        bot, fake_cdp = _make_bot(fake_env)
        behavior = make_behavior_score("COLLECT_FUEL", 900, 80, 90, "fuel=500")
        decision = make_tick_decision(
            command=make_pickup_fuel_command(80, 90),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision)

        assert fake_cdp._sent_methods == []

    def test_execute_rejects_invalid_combat_teleport_source(self, fake_env: FakeEnv) -> None:
        """Execute drops combat teleports when the target source is invalid."""
        bot, fake_cdp = _make_bot(fake_env)
        _store_tank(10, x=106, y=100, source="radar")
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
            }
        )
        behavior = make_behavior_score("HUNT", 800, 105, 100, "teleport enemy", target_id=10)
        decision = make_tick_decision(
            command=make_teleport_command(105, 100),
            behavior=behavior,
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        execute(bot, decision)

        assert fake_cdp._sent_methods == []

    def test_execute_rejects_teleport_to_known_mine(self, fake_env: FakeEnv) -> None:
        """Execute drops teleports whose landing tile is a known mine."""
        bot, fake_cdp = _make_bot(fake_env)
        import tankpit_bot.sniffer.world_state as ws

        ws._world_state = add_mine_from_radar(
            ws._world_state,
            x=200,
            y=200,
            team=1,
            timestamp_ms=1000,
        )
        behavior = make_behavior_score("COLLECT_EQUIPMENT", 800, 200, 200, "search_equipment_local")
        decision = make_tick_decision(
            command=make_teleport_command(200, 200),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision)

        assert fake_cdp._sent_methods == []
