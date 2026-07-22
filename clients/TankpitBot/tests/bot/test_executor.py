"""Tests for executor module: apply_equipment and dispatch_command."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.bot.ai.types import make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.executor import (
    _format_desired_equipment,
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
        from tankpit_bot.ledger.outcome.teleport import emit_teleport_landed

        world = _make_world()
        result = dispatch_command(
            _WorldOnlyBot(world),
            make_teleport_command(200, 200),
            _make_snapshot(map_visible=True),
        )

        assert result is False
        landed = emit_teleport_landed(
            duration_ms=0, target_x=200, target_y=200, landed_x=200, landed_y=200, messages=[]
        )
        assert landed["detail"]["sent_window"] == "(none)"

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


class TestExecute:
    """Tests for execute (apply_equipment + dispatch_command)."""

    def test_execute_applies_equipment_then_dispatches(self, fake_env: FakeEnv) -> None:
        """Execute applies equipment changes before dispatching command."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so execute needs to disable 1, 2, 4
        update_inventory_from_toggle(get_world_service(), [True, True, True, True, True])
        behavior = make_behavior_score("HUNT", 50, 100, 200, "search_collect_local")
        decision = make_tick_decision(
            command=make_move_command(100, 200),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[5],
        )
        execute(bot, decision, _make_snapshot())
        # Equipment toggles (disable 1, 2, 4) + move command
        assert len(fake_cdp._sent_methods) == 4

    def test_execute_hold_records_no_decision(self, fake_env: FakeEnv) -> None:
        """A hold decision produces no ledger decision record."""
        from tankpit_bot.ledger.decision import latest_decision_event_id

        bot, _fake_cdp = _make_bot(fake_env)
        decision = make_tick_decision(
            command=make_hold_command(),
            behavior=make_behavior_score("HUNT", 0, 0, 0, "manual_hold"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )
        assert execute(bot, decision, _make_snapshot()) is True
        assert latest_decision_event_id() == 0

    def test_execute_records_decision_for_dispatchable_command(self, fake_env: FakeEnv) -> None:
        """A dispatchable decision is recorded before validation."""
        from tankpit_bot.ledger.decision import decision_record, latest_decision_event_id

        bot, _fake_cdp = _make_bot(fake_env)
        decision = make_tick_decision(
            command=make_move_command(100, 200),
            behavior=make_behavior_score("COLLECT", 500, 100, 200, "search_collect_local"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )
        execute(bot, decision, _make_snapshot())
        recorded_id = latest_decision_event_id()
        assert decision_record(recorded_id) == {
            "event_id": recorded_id,
            "action_kind": "move",
            "cmd_type": "move",
            "mode": "COLLECT",
            "score": 500,
            "reason_kind": "search_collect_local",
            "reason_context": {},
            "target_x": 100,
            "target_y": 200,
            "target_id": 0,
        }

    def test_execute_dispatches_structurally_valid_shoot(self, fake_env: FakeEnv) -> None:
        """Execute dispatches a shoot at a tracked, in-position target.

        Combat presence (live tank vs map-only afterimage) is gated
        upstream by the HUNT owner's wire-presence kill gate; the executor
        only re-checks that the target still exists at the commanded tile,
        so ``source`` no longer affects dispatch.
        """
        bot, fake_cdp = _make_bot(fake_env)
        _store_tank(10, x=105, y=103, source="world_state")
        behavior = make_behavior_score("HUNT", 800, 105, 103, "shoot_target", target_id=10)
        decision = make_tick_decision(
            command=make_shoot_command(105, 103, 10),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        execute(bot, decision, _make_snapshot())

        assert fake_cdp._sent_methods == ["Runtime.evaluate"]


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

        behavior = make_behavior_score("HUNT", 900, 101, 100, "shoot_target")
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
        behavior = make_behavior_score("HUNT", 900, 101, 100, "shoot_target")
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
            behavior=make_behavior_score("HUNT", 900, 50, 60, "shoot_target"),
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
            behavior=make_behavior_score("HUNT", 500, 0, 0, "scan_on_landing"),
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[],
        )

        encoded = encode_tick_decision(decision)
        decoded = decode_tick_decision(encoded)
        assert decoded["secondary_command"] is None


class TestFuelBookEntries:
    """Dispatch-side fuel-book entries (Phase 3 live divergence)."""

    def test_failed_radar_dispatch_records_no_entry(self) -> None:
        """A radar the page refused must not enter the fuel book."""
        reset_world_state()
        bot = _WorldOnlyBot(make_empty_world_state())
        result = dispatch_command(bot, make_radar_command(), _make_snapshot())
        assert result is False
        assert get_world_service().fuel_book["entries"] == []

    def test_teleport_entry_needs_a_self_position(self) -> None:
        """Without a self fix the teleport cost cannot be priced."""
        from tankpit_bot.bot.executor import _record_teleport_fuel_entry

        reset_world_state()
        _record_teleport_fuel_entry(10, 20)
        assert get_world_service().fuel_book["entries"] == []
