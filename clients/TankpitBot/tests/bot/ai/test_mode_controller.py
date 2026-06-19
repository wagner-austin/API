"""Tests for durable AI mode controller helpers."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.mode_controller import (
    apply_mode_to_decision,
    clear_ai_mode,
    clear_mode_on_decision,
    derive_hunt_mode_state,
    derive_recover_equipment_mode_state,
    derive_recover_fuel_mode_state,
    set_ai_mode,
    should_enter_hunt,
    should_enter_recover_equipment,
    should_enter_recover_fuel,
    should_exit_hunt,
    should_exit_recover_equipment,
    should_exit_recover_fuel,
)
from tankpit_bot.bot.ai.types import AIStateDict, make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.tick_loop_types import make_tick_decision
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_shoot_command,
    make_teleport_command,
)
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _make_ctx(*, fuel: int = 800, dual_count: int = 30, radar_count: int = 30) -> DecideCtx:
    """Create a focused DecideCtx for durable mode tests.

    Args:
        fuel: Current fuel amount.
        dual_count: Dual-shot count.
        radar_count: Extra-radar count.

    Returns:
        Decision context for testing mode predicates.
    """
    world, self_state = make_world(fuel=fuel)
    ai_state = make_scanned_ai_state()
    inventory = make_inventory(default_count=30, dual_count=dual_count)
    inventory["extra_radars"]["count"] = radar_count
    return DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")


def test_clear_ai_mode_resets_durable_fields() -> None:
    """Clearing durable ownership resets mode fields only."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 12345,
        }
    )

    cleared = clear_ai_mode(state)

    assert cleared["mode"] == "UNSET"
    assert cleared["mode_state"] == ""
    assert cleared["mode_started_ms"] == 0
    assert cleared["last_scan_ms"] == state["last_scan_ms"]


def test_set_ai_mode_starts_new_mode_at_current_timestamp() -> None:
    """Entering a new durable mode records the entry timestamp."""
    state = make_initial_ai_state()

    updated = set_ai_mode(state, "HUNT", "ACQUIRE", 2000)

    assert updated["mode"] == "HUNT"
    assert updated["mode_state"] == "ACQUIRE"
    assert updated["mode_started_ms"] == 2000


def test_set_ai_mode_preserves_started_timestamp_when_mode_continues() -> None:
    """Rewriting substate within the same mode keeps the original entry time."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 2000,
        }
    )

    updated = set_ai_mode(state, "HUNT", "ENGAGE", 5000)

    assert updated["mode_state"] == "ENGAGE"
    assert updated["mode_started_ms"] == 2000


def test_set_ai_mode_rejects_invalid_pair() -> None:
    """Invalid durable mode/state pairs fail immediately."""
    with pytest.raises(ValueError, match="Invalid AI mode/state pair"):
        set_ai_mode(make_initial_ai_state(), "HUNT", "SEARCH", 1000)


def test_set_ai_mode_allows_unset_without_started_timestamp() -> None:
    """Explicitly setting UNSET keeps the entry timestamp cleared."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 1000,
        }
    )

    updated = set_ai_mode(state, "UNSET", "", 5000)

    assert updated["mode"] == "UNSET"
    assert updated["mode_state"] == ""
    assert updated["mode_started_ms"] == 0


def test_clear_mode_on_decision_clears_updated_ai_state_mode() -> None:
    """Decision rewriting to UNSET keeps command and behavior intact."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 1000,
        }
    )
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 0, 0, 0, "find_enemies"),
        updated_ai_state=state,
        desired_equipment=[1, 2],
    )

    cleared = clear_mode_on_decision(decision)

    assert cleared["command"] == decision["command"]
    assert cleared["behavior"] == decision["behavior"]
    assert cleared["desired_equipment"] == [1, 2]
    assert cleared["updated_ai_state"]["mode"] == "UNSET"


def test_apply_mode_to_decision_sets_durable_mode() -> None:
    """Decision rewriting can attach durable mode ownership."""
    decision = make_tick_decision(
        command=make_move_command(110, 100),
        behavior=make_behavior_score("HUNT", 800, 110, 100, "teleport enemy"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    updated = apply_mode_to_decision(decision, "HUNT", "CLOSE", 9000)

    assert updated["updated_ai_state"]["mode"] == "HUNT"
    assert updated["updated_ai_state"]["mode_state"] == "CLOSE"
    assert updated["updated_ai_state"]["mode_started_ms"] == 9000


def test_should_enter_recover_fuel_uses_low_threshold() -> None:
    """Fuel recovery entry uses the configured low threshold."""
    assert should_enter_recover_fuel(_make_ctx(fuel=250)) is True
    assert should_enter_recover_fuel(_make_ctx(fuel=300)) is True
    assert should_enter_recover_fuel(_make_ctx(fuel=500)) is False


def test_should_exit_recover_fuel_uses_full_threshold() -> None:
    """Fuel recovery exit uses the configured full threshold."""
    assert should_exit_recover_fuel(_make_ctx(fuel=1100)) is True
    assert should_exit_recover_fuel(_make_ctx(fuel=800)) is False


def test_should_enter_recover_equipment_uses_break_threshold() -> None:
    """Equipment recovery entry uses the configured break threshold."""
    assert should_enter_recover_equipment(_make_ctx(dual_count=5, radar_count=5)) is True
    assert should_enter_recover_equipment(_make_ctx(dual_count=30, radar_count=30)) is False


def test_should_exit_recover_equipment_uses_resume_threshold() -> None:
    """Equipment recovery exit uses the configured resume threshold."""
    assert should_exit_recover_equipment(_make_ctx(dual_count=25, radar_count=25)) is True
    assert should_exit_recover_equipment(_make_ctx(dual_count=5, radar_count=5)) is False


def test_radar_at_break_enters_recover_equipment_to_restock() -> None:
    """Radars at the break threshold enter restock even with full weapons.

    Radars find enemies and equipment, so the bot rebuilds the kit
    before hunting blind. The grid-sweep forager makes this safe at
    zero extras (it spends none), reversing the conservative exclusion
    that left the bot looping 0->3->2->1 (live run 20260613-011044).
    """
    assert should_enter_recover_equipment(_make_ctx(dual_count=30, radar_count=5)) is True


def test_radars_between_break_and_resume_do_not_re_enter() -> None:
    """Radars above the break but below resume do not start a fresh restock.

    The break/resume gap is hysteresis: a fresh entry needs the low
    break, so a bot fighting with a partial stock is not yanked back
    into restock at every spent radar.
    """
    assert should_enter_recover_equipment(_make_ctx(dual_count=30, radar_count=6)) is False


def test_exit_recover_equipment_requires_radar_resume() -> None:
    """Restored weapons do NOT release recovery while radars stay low.

    The mode holds until radars are rebuilt to the resume buffer, so
    the bot reaches a healthy stock before returning to the hunt
    instead of leaving at the first radar it scrapes together.
    """
    assert should_exit_recover_equipment(_make_ctx(dual_count=25, radar_count=5)) is False
    assert should_exit_recover_equipment(_make_ctx(dual_count=25, radar_count=19)) is False
    assert should_exit_recover_equipment(_make_ctx(dual_count=25, radar_count=20)) is True


def test_should_enter_hunt_when_no_recovery_mode_has_priority() -> None:
    """HUNT owns the tick only when no recovery mode has stronger entry rules."""
    assert should_enter_hunt(_make_ctx(fuel=500, dual_count=30, radar_count=30)) is True
    assert should_enter_hunt(_make_ctx(fuel=250, dual_count=30, radar_count=30)) is False


def test_should_exit_hunt_when_recovery_takes_priority() -> None:
    """HUNT exits when recovery conditions become active."""
    assert should_exit_hunt(_make_ctx(fuel=500, dual_count=30, radar_count=30)) is False
    assert should_exit_hunt(_make_ctx(fuel=250, dual_count=30, radar_count=30)) is True


def test_derive_hunt_mode_state_uses_command_shape_for_close_and_engage() -> None:
    """HUNT substates are derived from concrete combat commands."""
    closing = make_tick_decision(
        command=make_teleport_command(110, 100),
        behavior=make_behavior_score("HUNT", 800, 110, 100, "teleport enemy"),
        updated_ai_state=AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 42,
            }
        ),
        desired_equipment=[],
    )
    engaging = make_tick_decision(
        command=make_shoot_command(110, 100, 42),
        behavior=make_behavior_score("HUNT", 800, 110, 100, "shoot enemy"),
        updated_ai_state=AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 42,
            }
        ),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(closing) == "CLOSE"
    assert derive_hunt_mode_state(engaging) == "ENGAGE"


def test_derive_hunt_mode_state_keeps_search_teleport_in_acquire() -> None:
    """Enemy-search teleports do not masquerade as close-combat transitions."""
    decision = make_tick_decision(
        command=make_teleport_command(110, 100),
        behavior=make_behavior_score("HUNT", 0, 110, 100, "edge_for_enemies"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_maps_locked_walk_to_close() -> None:
    """A combat walk toward a locked target is a CLOSE transition."""
    decision = make_tick_decision(
        command=make_move_command(103, 100),
        behavior=make_behavior_score("HUNT", 800, 103, 100, "walk to Enemy"),
        updated_ai_state=AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 42,
            }
        ),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "CLOSE"


def test_derive_hunt_mode_state_uses_acquire_for_delegated_fuel_pickup() -> None:
    """A refuel-for-hunt delegation's pickup stays in HUNT acquire.

    When every engagement is unaffordable the hunt owner delegates the
    tick to the fuel planner; its pickup command has no HUNT-specific
    shape and lands in the acquire substate.
    """
    decision = make_tick_decision(
        command=make_pickup_fuel_command(110, 100),
        behavior=make_behavior_score("COLLECT_FUEL", 900, 110, 100, "fuel=500"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_uses_refresh_for_map_refresh() -> None:
    """Map refresh with a locked target maps to REFRESH."""
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "find target"),
        updated_ai_state=AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": 42,
            }
        ),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "REFRESH"


def test_derive_hunt_mode_state_uses_acquire_for_generic_enemy_search() -> None:
    """Generic enemy search remains HUNT acquire, not refresh."""
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 0, 0, 0, "find_enemies"),
        updated_ai_state=AIStateDict(
            **{
                **make_initial_ai_state(),
                "combat_target_id": -1,
            }
        ),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_maps_confirm_kill_reason() -> None:
    """Confirm-kill behavior reasons map to the explicit HUNT confirmation state."""
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "confirm_kill"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "CONFIRM_KILL"


def test_derive_recover_equipment_mode_state_maps_sense_search_and_pickup() -> None:
    """Equipment recovery substates are derived from concrete command intent."""
    sense = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("COLLECT_EQUIPMENT", 925, 0, 0, "radar_for_equipment"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    search = make_tick_decision(
        command=make_move_command(130, 100),
        behavior=make_behavior_score(
            "COLLECT_EQUIPMENT",
            925,
            130,
            100,
            "search_equipment_local",
        ),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    pickup = make_tick_decision(
        command=make_pickup_equipment_command(102, 101),
        behavior=make_behavior_score("COLLECT_EQUIPMENT", 925, 102, 101, "equipment_critical"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_recover_equipment_mode_state(sense) == "SENSE"
    assert derive_recover_equipment_mode_state(search) == "SEARCH"
    assert derive_recover_equipment_mode_state(pickup) == "PICKUP"


def test_derive_recover_equipment_mode_state_uses_approach_for_nonpickup_targeting() -> None:
    """Equipment movement toward a known target maps to APPROACH."""
    decision = make_tick_decision(
        command=make_move_command(108, 107),
        behavior=make_behavior_score("COLLECT_EQUIPMENT", 925, 108, 107, "equipment_restock"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_recover_equipment_mode_state(decision) == "APPROACH"


def test_derive_recover_fuel_mode_state_maps_sense_search_pickup_and_approach() -> None:
    """Fuel recovery substates are derived from concrete command intent."""
    sense = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("COLLECT_FUEL", 900, 0, 0, "radar_for_fuel"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    search = make_tick_decision(
        command=make_move_command(130, 100),
        behavior=make_behavior_score("COLLECT_FUEL", 900, 130, 100, "search_fuel_local"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    reposition = make_tick_decision(
        command=make_move_command(92, 92),
        behavior=make_behavior_score("COLLECT_FUEL", 900, 92, 92, "edge_for_fuel"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    pickup = make_tick_decision(
        command=make_pickup_fuel_command(102, 101),
        behavior=make_behavior_score("COLLECT_FUEL", 900, 102, 101, "fuel=700"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    approach = make_tick_decision(
        command=make_move_command(102, 101),
        behavior=make_behavior_score("COLLECT_FUEL", 900, 102, 101, "fuel=700"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_recover_fuel_mode_state(sense) == "SENSE"
    assert derive_recover_fuel_mode_state(search) == "SEARCH"
    assert derive_recover_fuel_mode_state(reposition) == "SEARCH"
    assert derive_recover_fuel_mode_state(pickup) == "PICKUP"
    assert derive_recover_fuel_mode_state(approach) == "APPROACH"
