"""Tests for executor module: apply_equipment and dispatch_command."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
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
    make_hold_command,
    make_map_open_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.sniffer.world_state import (
    get_world_service,
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


def _make_snapshot(*, map_visible: bool = False) -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot for executor tests.

    Defaults to ``map_visible=False`` so the ``map_open`` and ``teleport``
    dispatch branches exercise their normal "open the map first" path.
    Tests covering the short-circuit pass ``map_visible=True``.
    """
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=True,
        map_visible=map_visible,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=100,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _make_bot(fake_env: FakeEnv) -> tuple[Bot, FakeCDPSession]:
    """Create a Bot with FakeCDPSession in IDLE state."""
    reset_world_state()
    update_world_state_from_position(100, 100)
    update_world_state_from_fuel_total(get_world_service(), 800)
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

    new_tanks = dict(get_world_service().world_state["tanks"])
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
    get_world_service().world_state["tanks"] = new_tanks


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

    def close_map(self) -> bool:
        """Unused BotProtocol stub."""
        return False

    def captured_message_count(self) -> int:
        """Unused BotProtocol stub."""
        return 0

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
        update_inventory_from_gain(get_world_service(), [0, 5, 0, 0, 0])
        # Disable slot 2 via toggle so enable triggers a toggle command
        update_inventory_from_toggle(get_world_service(), [True, False, True, True, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not desired, enabled -> disable -> toggle (1 CDP)
        # slot 2: desired, disabled, has stock -> enable -> toggle (1 CDP)
        # slot 4: not desired, enabled -> disable -> toggle (1 CDP)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_disables_undesired_slots(self, fake_env: FakeEnv) -> None:
        """Disables combat slots not in desired list."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so we can test disabling
        update_inventory_from_toggle(get_world_service(), [True, True, True, True, True])
        apply_equipment(bot, [5])
        # Should disable slots 1, 2, 4 (3 CDP calls)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_skips_already_correct_state(self, fake_env: FakeEnv) -> None:
        """No toggles when equipment already matches desired state."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set inventory to match: 1=off, 2=on, 4=off, 5=on
        update_inventory_from_toggle(get_world_service(), [False, True, True, False, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not in desired, already disabled -> no toggle
        # slot 2: in desired, already enabled -> no toggle
        # slot 4: not in desired, already disabled -> no toggle
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
            behavior=make_behavior_score("COLLECT", 900, 100, 100, "noop"),
            updated_ai_state=AIStateDict(**{**make_initial_ai_state(), "resource_target_kind": ""}),
            desired_equipment=[],
        )

        assert _tracked_resource_target(world, decision) is None

    def test_valid_teleport_allows_search_hop_without_locked_resource(self) -> None:
        """Search teleports remain valid without a locked resource target."""
        world = _make_world()
        decision = make_tick_decision(
            command=make_teleport_command(120, 120),
            behavior=make_behavior_score("COLLECT", 900, 120, 120, "search_collect_local"),
            updated_ai_state=AIStateDict(**{**make_initial_ai_state(), "resource_target_kind": ""}),
            desired_equipment=[],
        )

        assert _is_valid_teleport(world, decision) is True


class TestDispatchCommand:
    """Tests for dispatch_command."""

    def test_dispatch_move(self, fake_env: FakeEnv) -> None:
        """Dispatches move command via bot.move_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_move_command(150, 160), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_fuel(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_fuel command via bot.pickup_fuel_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_fuel_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_equipment(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_equipment command via bot.pickup_equipment_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_equipment_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_shoot(self, fake_env: FakeEnv) -> None:
        """Dispatches shoot command via bot.shoot_at."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_shoot_command(105, 103), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_radar(self, fake_env: FakeEnv) -> None:
        """Dispatches radar command via bot.use_radar."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_radar_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_map_open(self, fake_env: FakeEnv) -> None:
        """Dispatches map_open command via bot.open_map."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_map_open_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_hold_sends_nothing(self, fake_env: FakeEnv) -> None:
        """Hold command returns True and does not touch the wire.

        The SPA-driven idle tick must not dispatch any CDP command;
        the fake CDP session confirms no ``Runtime.evaluate`` reached
        it while ``dispatch_command`` still reports success (the
        desired effect — do nothing — was achieved).
        """
        bot, fake_cdp = _make_bot(fake_env)
        assert "Runtime.evaluate" not in fake_cdp._sent_methods
        result = dispatch_command(bot, make_hold_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" not in fake_cdp._sent_methods

    def test_dispatch_map_open_sends_wire_only_when_already_visible(
        self, fake_env: FakeEnv
    ) -> None:
        """A visible map dispatches CMD_MAP_OPEN with no client-side keypress.

        Regression guard for capture 20260620-183916: CMD_MAP_OPEN is
        idempotent on the server -- every wire dispatch produces a fresh
        MAP_DATA payload regardless of overlay visibility. The previous
        close-then-reopen hack added a synthetic 'm' keypress before
        every redundant intel refresh; the wire dispatch alone is what
        the server actually needs.
        """
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(
            bot,
            make_map_open_command(),
            _make_snapshot(map_visible=True),
        )
        assert result is True
        # No synthetic 'm' keypress dispatched -- only the wire command.
        key_events = [m for m in fake_cdp._sent_methods if m == "Input.dispatchKeyEvent"]
        assert key_events == []
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport(self, fake_env: FakeEnv) -> None:
        """Dispatches teleport command via bot.teleport_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_teleport_command(200, 200), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport_records_no_attempt_when_send_fails(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A failed teleport send leaves no pending attempt to mislabel later."""
        from tankpit_bot.diagnostics.teleport_attempts import emit_teleport_attempt_outcome

        world = _make_world()
        result = dispatch_command(
            _WorldOnlyBot(world),
            make_teleport_command(200, 200),
            _make_snapshot(map_visible=True),
        )

        assert result is False
        assert emit_teleport_attempt_outcome(status="landed_exact", messages=[]) is False

    def test_dispatch_teleport_skips_open_map_when_map_already_visible(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Teleport skips the precondition map_open when the map is already open."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(
            bot,
            make_teleport_command(200, 200),
            _make_snapshot(map_visible=True),
        )
        assert result is True
        sent_methods = fake_cdp._sent_methods
        runtime_calls = [m for m in sent_methods if m == "Runtime.evaluate"]
        assert len(runtime_calls) == 1


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

    def test_valid_shoot_accepts_clamped_aim_drift(self) -> None:
        """Aim tile is a viewport-legal hint; target_id is the truth channel.

        Under ``_clamp_aim_into_viewport`` the aim tile is deliberately
        different from the target tank's current position: the server
        picks homing from ``target_id`` (off-adjacent aim + valid target
        id) and the seeker tracks the true target wherever it is. The
        executor accepts the drift; rejecting it silently blocked every
        clamped homing shot in the 2026-07-06 20:47:31 live-run deadlock
        (26 s of client-side self-rejections). Only the tank-existence
        race is guarded.
        """
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

        assert result is True

    def test_valid_shoot_rejects_missing_target(self) -> None:
        """Shoot validation rejects unknown target ids."""
        result = _is_valid_shoot(_make_world(), make_shoot_command(105, 103, 10))

        assert result is False

    def test_valid_shoot_accepts_tracked_in_position_target(self) -> None:
        """Shoot validation accepts a tracked target at the commanded tile."""
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

    def test_valid_shoot_accepts_target_regardless_of_source(self) -> None:
        """Source no longer gates the shot: presence is the AI gate's job.

        A world-state (map-sourced) target that is not inside the visible
        viewport was rejected by the old viewport-fresh proxy; it is now
        accepted structurally, because the wire-presence kill gate -- not
        the executor -- decides whether the tank is a live target.
        """
        world = _make_world()
        world["viewport"]["left"] = 0
        world["viewport"]["top"] = 0
        world["viewport"]["width"] = 4
        world["viewport"]["height"] = 4
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "fuel=500"),
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "equipment_low"),
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "fuel=500"),
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "fuel=500"),
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "fuel=500"),
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "fuel=500"),
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
            behavior=make_behavior_score("COLLECT", 900, 104, 100, "fuel=500"),
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
        update_inventory_from_toggle(get_world_service(), [True, True, True, True, True])
        behavior = make_behavior_score("HUNT", 50, 100, 200, "patrol_waypoint")
        decision = make_tick_decision(
            command=make_move_command(100, 200),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[5],
        )
        execute(bot, decision, _make_snapshot())
        # Equipment toggles (disable 1, 2, 4) + move command
        assert len(fake_cdp._sent_methods) == 4

    def test_execute_dispatches_structurally_valid_shoot(self, fake_env: FakeEnv) -> None:
        """Execute dispatches a shoot at a tracked, in-position target.

        Combat presence (live tank vs map-only afterimage) is gated
        upstream by the HUNT owner's wire-presence kill gate; the executor
        only re-checks that the target still exists at the commanded tile,
        so ``source`` no longer affects dispatch.
        """
        bot, fake_cdp = _make_bot(fake_env)
        _store_tank(10, x=105, y=103, source="world_state")
        behavior = make_behavior_score("HUNT", 800, 105, 103, "shoot enemy", target_id=10)
        decision = make_tick_decision(
            command=make_shoot_command(105, 103, 10),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision, _make_snapshot())

        assert fake_cdp._sent_methods == ["Runtime.evaluate"]

    def test_execute_rejects_missing_pickup_target(self, fake_env: FakeEnv) -> None:
        """Execute drops pickup commands when the container no longer exists."""
        bot, fake_cdp = _make_bot(fake_env)
        behavior = make_behavior_score("COLLECT", 900, 80, 90, "fuel=500")
        decision = make_tick_decision(
            command=make_pickup_fuel_command(80, 90),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision, _make_snapshot())

        assert fake_cdp._sent_methods == []

    def test_execute_rejects_stale_combat_teleport(self, fake_env: FakeEnv) -> None:
        """Execute drops combat teleports whose locked target has drifted.

        The tracked tank no longer sits on the locked combat-target tile,
        so the target reads as stale and the teleport is rejected before
        any CDP command is sent.
        """
        bot, fake_cdp = _make_bot(fake_env)
        _store_tank(10, x=108, y=100, source="viewport")
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

        execute(bot, decision, _make_snapshot())

        assert fake_cdp._sent_methods == []

    def test_execute_rejects_teleport_to_known_mine(self, fake_env: FakeEnv) -> None:
        """Execute drops teleports whose landing tile is a known mine."""
        bot, fake_cdp = _make_bot(fake_env)

        get_world_service().world_state = add_mine_from_radar(
            get_world_service().world_state,
            x=200,
            y=200,
            team=1,
            timestamp_ms=1000,
        )
        behavior = make_behavior_score("COLLECT", 800, 200, 200, "search_collect_local")
        decision = make_tick_decision(
            command=make_teleport_command(200, 200),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision, _make_snapshot())

        assert fake_cdp._sent_methods == []


class TestSecondaryCommandDispatch:
    """Tests for multi-command tick dispatch via secondary_command."""

    def setup_method(self) -> None:
        reset_world_state()

    def teardown_method(self) -> None:
        reset_world_state()

    def test_secondary_dispatched_after_primary(self, fake_env: FakeEnv) -> None:
        """When primary succeeds, secondary is also dispatched."""
        bot, _cdp = _make_bot(fake_env)
        ws = get_world_service()
        ws.world_state = WorldStateDict(
            self_state=ws.world_state["self_state"],
            tanks={
                "50": make_tank_state(
                    tank_id=50,
                    x=101,
                    y=100,
                    team=1,
                    rank=0,
                    damage_state=0,
                    name="Enemy",
                    is_bot=False,
                    is_self=False,
                    source="viewport",
                    timestamp_ms=1000,
                    last_wire_seen_ms=1000,
                    last_position_update_ms=1000,
                    last_viewport_observation_ms=1000,
                ),
            },
            containers={
                "99,100": make_container_state(
                    x=99,
                    y=100,
                    is_fuel=True,
                    volume=300,
                    timestamp_ms=1000,
                ),
            },
            mines=ws.world_state["mines"],
            terrain=ws.world_state["terrain"],
            viewport=ws.world_state["viewport"],
            scanned_tiles=ws.world_state["scanned_tiles"],
            timestamp_ms=1000,
        )

        behavior = make_behavior_score("HUNT", 900, 101, 100, "engage_shoot")
        decision = make_tick_decision(
            command=make_shoot_command(101, 100, 50),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[2],
            secondary_command=make_pickup_fuel_command(99, 100),
        )

        result = execute(bot, decision, _make_snapshot())
        assert result is True

    def test_no_secondary_when_primary_fails_validation(self) -> None:
        """When primary fails validation, secondary is not dispatched."""
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        behavior = make_behavior_score("HUNT", 900, 101, 100, "engage_shoot")
        decision = make_tick_decision(
            command=make_shoot_command(101, 100, 999),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[2],
            secondary_command=make_pickup_fuel_command(99, 100),
        )

        result = execute(bot, decision, _make_snapshot())
        assert result is False


class TestTickDecisionCodecs:
    """Tests for encode/decode with secondary_command."""

    def test_encode_decode_with_secondary(self) -> None:
        """Round-trip encode/decode preserves secondary_command."""
        from tankpit_bot.bot.tick_loop_types import (
            decode_tick_decision,
            encode_tick_decision,
        )

        decision = make_tick_decision(
            command=make_shoot_command(50, 60, 99),
            behavior=make_behavior_score("HUNT", 900, 50, 60, "engage"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[1, 2],
            secondary_command=make_pickup_fuel_command(49, 60),
        )

        encoded = encode_tick_decision(decision)
        decoded = decode_tick_decision(encoded)
        if decoded["secondary_command"] is None:
            raise AssertionError("secondary_command should not be None")
        assert decoded["secondary_command"]["cmd_type"] == "pickup_fuel"

    def test_encode_decode_without_secondary(self) -> None:
        """Round-trip encode/decode with no secondary_command."""
        from tankpit_bot.bot.tick_loop_types import (
            decode_tick_decision,
            encode_tick_decision,
        )

        decision = make_tick_decision(
            command=make_radar_command(),
            behavior=make_behavior_score("HUNT", 500, 0, 0, "scan"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        encoded = encode_tick_decision(decision)
        decoded = decode_tick_decision(encoded)
        assert decoded["secondary_command"] is None
