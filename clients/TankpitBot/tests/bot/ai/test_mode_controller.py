"""Tests for :mod:`tankpit_bot.bot.ai.mode_controller`.

Mode set/clear, dispatch counters, the hold decision, and substate
derivation. ``test_mode_controller.py`` was 791 lines; the gates are
now a sibling, mirroring the source split.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.mode_controller import (
    apply_dispatch_counters,
    apply_mode_to_decision,
    clear_ai_mode,
    clear_mode_on_decision,
    derive_collect_mode_state,
    derive_hunt_mode_state,
    make_hold_decision,
    resolve_owner_from_manual,
    set_ai_mode,
)
from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.tick_loop_types import (
    make_tick_decision,
)
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tests.bot.ai._mode_fixtures import (
    _make_decision,
    _make_hold_inventory,
)


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
        behavior=make_behavior_score("HUNT", 800, 110, 100, "teleport_target"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    updated = apply_mode_to_decision(decision, "HUNT", "CLOSE", 9000)

    assert updated["updated_ai_state"]["mode"] == "HUNT"
    assert updated["updated_ai_state"]["mode_state"] == "CLOSE"
    assert updated["updated_ai_state"]["mode_started_ms"] == 9000


def test_derive_hunt_mode_state_uses_command_shape_for_close_and_engage() -> None:
    """HUNT substates are derived from concrete combat commands."""
    closing = make_tick_decision(
        command=make_teleport_command(110, 100),
        behavior=make_behavior_score("HUNT", 800, 110, 100, "teleport_target"),
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
        behavior=make_behavior_score("HUNT", 800, 110, 100, "shoot_target"),
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


def test_derive_hunt_mode_state_keeps_non_combat_teleport_in_acquire() -> None:
    """A HUNT teleport without a locked combat target derives ACQUIRE.

    Defensive: HUNT acquire dispatches map_open as its only enemy
    search action (post-2026-06-22), so production no longer produces
    teleports without a locked target. The derive function still
    needs to land on ACQUIRE (not CLOSE) for any such input, in case
    a future HUNT path produces a teleport without first setting a
    combat target.
    """
    decision = make_tick_decision(
        command=make_teleport_command(110, 100),
        behavior=make_behavior_score("HUNT", 0, 110, 100, "search_collect_local"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_uses_acquire_for_delegated_fuel_pickup() -> None:
    """A refuel-for-hunt delegation's pickup stays in HUNT acquire.

    When every engagement is unaffordable the hunt owner delegates the
    tick to the fuel planner; its pickup command has no HUNT-specific
    shape and lands in the acquire substate.
    """
    decision = make_tick_decision(
        command=make_pickup_fuel_command(110, 100),
        behavior=make_behavior_score("COLLECT", 900, 110, 100, "fuel_collect"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_uses_refresh_for_map_refresh() -> None:
    """Map refresh with a locked target maps to REFRESH."""
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "find_target"),
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


def test_derive_collect_mode_state_maps_sense_search_and_pickup() -> None:
    """Equipment recovery substates are derived from concrete command intent."""
    sense = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("COLLECT", 925, 0, 0, "forage_radar"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    search = make_tick_decision(
        command=make_move_command(130, 100),
        behavior=make_behavior_score(
            "COLLECT",
            925,
            130,
            100,
            "search_collect_local",
        ),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    pickup = make_tick_decision(
        command=make_pickup_equipment_command(102, 101),
        behavior=make_behavior_score("COLLECT", 925, 102, 101, "equipment_restock"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_collect_mode_state(sense) == "SENSE"
    assert derive_collect_mode_state(search) == "SEARCH"
    assert derive_collect_mode_state(pickup) == "PICKUP"


def test_derive_collect_mode_state_uses_approach_for_nonpickup_targeting() -> None:
    """Equipment movement toward a known target maps to APPROACH."""
    decision = make_tick_decision(
        command=make_move_command(108, 107),
        behavior=make_behavior_score("COLLECT", 925, 108, 107, "equipment_restock"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_collect_mode_state(decision) == "APPROACH"


def test_derive_collect_mode_state_maps_sense_search_pickup_and_approach() -> None:
    """Fuel recovery substates are derived from concrete command intent."""
    sense = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("COLLECT", 900, 0, 0, "forage_radar"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    search = make_tick_decision(
        command=make_move_command(130, 100),
        behavior=make_behavior_score("COLLECT", 900, 130, 100, "search_collect_local"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    pickup = make_tick_decision(
        command=make_pickup_fuel_command(102, 101),
        behavior=make_behavior_score("COLLECT", 900, 102, 101, "fuel_locked"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    approach = make_tick_decision(
        command=make_move_command(102, 101),
        behavior=make_behavior_score("COLLECT", 900, 102, 101, "fuel_locked"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_collect_mode_state(sense) == "SENSE"
    assert derive_collect_mode_state(search) == "SEARCH"
    assert derive_collect_mode_state(pickup) == "PICKUP"
    assert derive_collect_mode_state(approach) == "APPROACH"


def test_resolve_owner_from_manual_returns_none_when_unset() -> None:
    """``manual_mode = None`` yields ``None`` (auto-arbitration)."""
    state = make_initial_ai_state()
    assert state["manual_mode"] is None
    assert resolve_owner_from_manual(state) is None


def test_resolve_owner_from_manual_pins_unset() -> None:
    """``manual_mode = "UNSET"`` short-circuits with the same literal."""
    state = AIStateDict(**{**make_initial_ai_state(), "manual_mode": "UNSET"})
    assert resolve_owner_from_manual(state) == "UNSET"


def test_resolve_owner_from_manual_pins_hunt() -> None:
    """``manual_mode = "HUNT"`` short-circuits with the same literal."""
    state = AIStateDict(**{**make_initial_ai_state(), "manual_mode": "HUNT"})
    assert resolve_owner_from_manual(state) == "HUNT"


def test_resolve_owner_from_manual_pins_collect() -> None:
    """``manual_mode = "COLLECT"`` short-circuits with the same literal."""
    state = AIStateDict(**{**make_initial_ai_state(), "manual_mode": "COLLECT"})
    assert resolve_owner_from_manual(state) == "COLLECT"


def test_make_hold_decision_produces_hold_command_and_unset_state() -> None:
    """Hold decision emits ``cmd_type = "hold"`` and clears durable ownership."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "manual_mode": "UNSET",
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 12000,
        }
    )

    decision = make_hold_decision(
        state, timestamp_ms=15000, fuel=900, inventory=_make_hold_inventory(), rank=1
    )

    assert decision["command"]["cmd_type"] == "hold"
    assert decision["updated_ai_state"]["mode"] == "UNSET"
    assert decision["updated_ai_state"]["mode_state"] == ""
    # A transition from HUNT into UNSET refreshes the started timestamp.
    assert decision["updated_ai_state"]["mode_started_ms"] == 15000
    assert decision["updated_ai_state"]["manual_mode"] == "UNSET"
    assert decision["behavior"]["reason_kind"] == "manual_hold"
    # The hold keeps the tank ARMED: dual (2) + homing (4) while
    # stocked, radar (5) always — the empty set the first idle-pin
    # implementation requested disarmed the tank visibly (user
    # report 2026-07-29) and, because toggle state persists across
    # logout, left it disarmed for the next login too.
    assert decision["desired_equipment"] == [2, 4, 5]
    assert decision["secondary_command"] is None


def test_make_hold_decision_drops_empty_weapon_stocks_from_the_loadout() -> None:
    """Depleted dual/homing stocks stay off the hold loadout (no dead toggles)."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "manual_mode": "UNSET",
        }
    )

    decision = make_hold_decision(
        state,
        timestamp_ms=15000,
        fuel=900,
        inventory=_make_hold_inventory(dual_count=0, homing_count=0),
        rank=1,
    )

    assert decision["desired_equipment"] == [5]


def test_apply_dispatch_counters_increments_radars_used_on_radar() -> None:
    """A radar primary advances ``live_radars_used`` by 1."""
    decision = _make_decision(make_radar_command())

    updated = apply_dispatch_counters(decision)

    assert updated["updated_ai_state"]["live_radars_used"] == 1
    assert updated["updated_ai_state"]["live_teleports"] == 0


def test_apply_dispatch_counters_increments_teleports_on_teleport() -> None:
    """A teleport primary advances ``live_teleports`` by 1."""
    decision = _make_decision(make_teleport_command(50, 60))

    updated = apply_dispatch_counters(decision)

    assert updated["updated_ai_state"]["live_teleports"] == 1
    assert updated["updated_ai_state"]["live_radars_used"] == 0


def test_apply_dispatch_counters_leaves_counters_untouched_on_other_commands() -> None:
    """A shoot / move / pickup / map_open primary leaves counters alone."""
    for command in (
        make_shoot_command(1, 2, 3),
        make_move_command(10, 20),
        make_pickup_fuel_command(30, 40),
        make_pickup_equipment_command(50, 60),
        make_map_open_command(),
    ):
        decision = _make_decision(command)
        updated = apply_dispatch_counters(decision)
        assert updated["updated_ai_state"]["live_radars_used"] == 0, command["cmd_type"]
        assert updated["updated_ai_state"]["live_teleports"] == 0, command["cmd_type"]


def test_apply_dispatch_counters_increments_secondary_radar() -> None:
    """A radar as ``secondary_command`` also contributes to the counter."""
    decision = _make_decision(
        make_shoot_command(1, 2, 3),
        secondary_command=make_radar_command(),
    )

    updated = apply_dispatch_counters(decision)

    assert updated["updated_ai_state"]["live_radars_used"] == 1
    assert updated["updated_ai_state"]["live_teleports"] == 0


def test_apply_dispatch_counters_increments_secondary_teleport() -> None:
    """A teleport as ``secondary_command`` also contributes to the counter."""
    decision = _make_decision(
        make_shoot_command(1, 2, 3),
        secondary_command=make_teleport_command(70, 80),
    )

    updated = apply_dispatch_counters(decision)

    assert updated["updated_ai_state"]["live_teleports"] == 1
    assert updated["updated_ai_state"]["live_radars_used"] == 0


def test_apply_dispatch_counters_accumulates_from_prior_state() -> None:
    """Counter increments stack on the existing state values."""
    state = AIStateDict(**{**make_initial_ai_state(), "live_radars_used": 3, "live_teleports": 7})
    decision = _make_decision(make_radar_command(), state)

    updated = apply_dispatch_counters(decision)

    assert updated["updated_ai_state"]["live_radars_used"] == 4
    assert updated["updated_ai_state"]["live_teleports"] == 7
