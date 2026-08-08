"""Tests for executor module: apply_equipment and execute."""

from __future__ import annotations

from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import make_initial_ai_state
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
    make_move_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_toggle
from tankpit_bot.state import (
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_tank_state,
)
from tests.bot._executor_support import (
    _make_bot,
    _make_snapshot,
    _store_tank,
    _WorldOnlyBot,
)
from tests.conftest import FakeEnv


class TestApplyEquipment:
    """Tests for apply_equipment."""

    def test_enables_desired_slots(self, fake_env: FakeEnv) -> None:
        """Enables desired combat slots that have stock."""
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_gain

        bot, fake_cdp = _make_bot(fake_env)
        # Give slot 2 (dual) stock so _has_equipment_stock returns True
        update_inventory_from_gain(bot.world, [0, 5, 0, 0, 0])
        # Disable slot 2 via toggle so enable triggers a toggle command
        update_inventory_from_toggle(bot.world, [True, False, True, True, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not desired, enabled -> disable -> toggle (1 CDP)
        # slot 2: desired, disabled, has stock -> enable -> toggle (1 CDP)
        # slot 4: not desired, enabled -> disable -> toggle (1 CDP)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_disables_undesired_slots(self, fake_env: FakeEnv) -> None:
        """Disables combat slots not in desired list."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so we can test disabling
        update_inventory_from_toggle(bot.world, [True, True, True, True, True])
        apply_equipment(bot, [5])
        # Should disable slots 1, 2, 4 (3 CDP calls)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_skips_already_correct_state(self, fake_env: FakeEnv) -> None:
        """No toggles when equipment already matches desired state."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set inventory to match: 1=off, 2=on, 4=off, 5=on
        update_inventory_from_toggle(bot.world, [False, True, True, False, True])
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


class TestExecutorValidationHelpers:
    """Focused tests for executor-side validation helpers."""


class TestExecute:
    """Tests for execute (apply_equipment + dispatch_command)."""

    def test_execute_applies_equipment_then_dispatches(self, fake_env: FakeEnv) -> None:
        """Execute applies equipment changes before dispatching command."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so execute needs to disable 1, 2, 4
        update_inventory_from_toggle(bot.world, [True, True, True, True, True])
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
        assert latest_decision_event_id(bot.world.ledger) == 0

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
        recorded_id = latest_decision_event_id(bot.world.ledger)
        assert decision_record(bot.world.ledger, recorded_id) == {
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
        _store_tank(bot.world, 10, x=105, y=103, source="world_state")
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

    def test_secondary_dispatched_after_primary(self, fake_env: FakeEnv) -> None:
        """When primary succeeds, secondary is also dispatched."""
        bot, _cdp = _make_bot(fake_env)
        bot.world.world_state = WorldStateDict(
            self_state=bot.world.world_state["self_state"],
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
            mines=bot.world.world_state["mines"],
            terrain=bot.world.world_state["terrain"],
            viewport=bot.world.world_state["viewport"],
            scanned_tiles=bot.world.world_state["scanned_tiles"],
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
        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
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
        bot = _WorldOnlyBot(make_empty_world_state())
        result = dispatch_command(bot, make_radar_command(), _make_snapshot())
        assert result is False
        assert bot.world.fuel_book["entries"] == []

    def test_teleport_entry_needs_a_self_position(self) -> None:
        """Without a self fix the teleport cost cannot be priced."""
        from tankpit_bot.bot.executor import _record_teleport_fuel_entry

        ws = WorldService()
        _record_teleport_fuel_entry(ws, 10, 20)
        assert ws.fuel_book["entries"] == []
