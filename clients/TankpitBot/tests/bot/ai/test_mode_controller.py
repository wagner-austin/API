"""Tests for durable AI mode controller helpers."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.mode_controller import (
    apply_dispatch_counters,
    apply_mode_to_decision,
    clear_ai_mode,
    clear_mode_on_decision,
    derive_collect_mode_state,
    derive_hunt_mode_state,
    hunt_fuel_floor,
    make_hold_decision,
    resolve_owner_from_manual,
    set_ai_mode,
    should_enter_collect,
    should_enter_hunt,
    should_exit_collect,
    should_exit_hunt,
)
from tankpit_bot.bot.ai.types import AIStateDict, make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import (
    BotCommand,
    make_map_open_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _make_ctx(*, fuel: int = 1200, dual_count: int = 30, radar_count: int = 30) -> DecideCtx:
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
        behavior=make_behavior_score("HUNT", 800, 110, 100, "teleport_target"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    updated = apply_mode_to_decision(decision, "HUNT", "CLOSE", 9000)

    assert updated["updated_ai_state"]["mode"] == "HUNT"
    assert updated["updated_ai_state"]["mode_state"] == "CLOSE"
    assert updated["updated_ai_state"]["mode_started_ms"] == 9000


def test_should_enter_collect_fires_below_full_between_kills() -> None:
    """With no combat lock, anything short of a full tank collects.

    User contract 2026-07-25: hunting is a privilege of a full tank,
    so between kills the entry bar is the rank capacity (1200 at
    rank 2) -- the old ``fuel_low + engagement_budget`` floor (650)
    is subsumed. The low threshold itself still fires regardless.
    """
    assert should_enter_collect(_make_ctx(fuel=150)) is True
    assert should_enter_collect(_make_ctx(fuel=200)) is True
    assert should_enter_collect(_make_ctx(fuel=650)) is True
    assert should_enter_collect(_make_ctx(fuel=1199)) is True
    assert should_enter_collect(_make_ctx(fuel=1200)) is False


def test_should_exit_collect_requires_rank_capacity_fuel() -> None:
    """Fuel recovery exit demands the rank's actual full tank.

    User ruling 2026-07-25: "just determine max fuel based on the
    tank rank" -- at rank 2 the capacity is 1200, so 1100 no longer
    releases the mode.
    """
    assert should_exit_collect(_make_ctx(fuel=1200)) is True
    assert should_exit_collect(_make_ctx(fuel=1100)) is False
    assert should_exit_collect(_make_ctx(fuel=800)) is False


def test_should_enter_collect_uses_break_threshold() -> None:
    """Equipment recovery entry uses the configured break threshold."""
    assert should_enter_collect(_make_ctx(dual_count=5, radar_count=5)) is True
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=30)) is False


def test_should_exit_collect_requires_a_full_stock() -> None:
    """COLLECT releases only at a genuinely full stock.

    User contract (2026-07-25): "never hunt if it is not full on
    everything except -5 max radar." At rank 2 the cap is 30, so
    duals below 30 hold the mode even though the old resume
    threshold (25) is satisfied.
    """
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=30)) is True
    assert should_exit_collect(_make_ctx(dual_count=25, radar_count=25)) is False
    assert should_exit_collect(_make_ctx(dual_count=5, radar_count=5)) is False


def test_radar_at_break_enters_recover_equipment_to_restock() -> None:
    """Radars at the break threshold enter restock even with full weapons.

    Radars find enemies and equipment, so the bot rebuilds the kit
    before hunting blind. The grid-sweep forager makes this safe at
    zero extras (it spends none), reversing the conservative exclusion
    that left the bot looping 0->3->2->1 (live run 20260613-011044).
    """
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=5)) is True


def test_radars_below_the_cap_floor_trigger_restock_between_kills() -> None:
    """Radar counts below cap-5 re-enter recovery between kills.

    User contract 2026-07-25: the between-kills bar is the rank cap
    (30 at rank 2, radar floor 25) -- the old fixed resume threshold
    (20) under-restocked high ranks. The bot rebuilds a genuinely
    full kit before every engagement cycle.
    """
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=6)) is True
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=24)) is True
    assert should_enter_collect(_make_ctx(dual_count=30, radar_count=25)) is False


def test_exit_recover_equipment_requires_radars_within_five_of_cap() -> None:
    """Full weapons do NOT release recovery while radars stay low.

    The mode holds until extra radars are within 5 of the rank cap
    (cap 30 at rank 2, so the floor is 25), so the bot reaches a
    genuinely full kit before returning to the hunt instead of
    leaving at the first radar it scrapes together.
    """
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=5)) is False
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=24)) is False
    assert should_exit_collect(_make_ctx(dual_count=30, radar_count=25)) is True


def test_hunt_fuel_floor_is_the_rank_fuel_capacity() -> None:
    """The full-fuel floor is exactly what the rank's tank holds.

    User ruling 2026-07-25: "just determine max fuel based on the
    tank rank". A recruit is hunt-ready at their genuine full tank
    of 1000; rank 2 needs its full 1200. An unreachable fixed floor
    would trap low ranks in COLLECT forever.
    """
    recruit_ctx = _make_ctx(fuel=1000)
    recruit_ctx.self_state["rank"] = 0
    assert hunt_fuel_floor(recruit_ctx) == 1000
    assert should_enter_hunt(recruit_ctx) is True
    assert hunt_fuel_floor(_make_ctx(fuel=1200)) == 1200


def test_should_enter_hunt_requires_full_fuel_and_full_stock() -> None:
    """HUNT entry is a privilege of a full tank (contract 2026-07-25).

    Fuel below the rank's capacity (1200 at rank 2) refuses entry
    even with a perfect inventory; a full tank with weapons below
    cap refuses too.
    """
    assert should_enter_hunt(_make_ctx(fuel=1200, dual_count=30, radar_count=30)) is True
    assert should_enter_hunt(_make_ctx(fuel=700, dual_count=30, radar_count=30)) is False
    assert should_enter_hunt(_make_ctx(fuel=1200, dual_count=25, radar_count=30)) is False
    assert should_enter_hunt(_make_ctx(fuel=150, dual_count=30, radar_count=30)) is False


def test_should_exit_hunt_when_recovery_takes_priority() -> None:
    """HUNT exits when a COLLECT trigger fires.

    Between kills (no lock) a non-full tank releases the hunt for a
    restock; a full tank holds it.
    """
    assert should_exit_hunt(_make_ctx(fuel=1200, dual_count=30, radar_count=30)) is False
    assert should_exit_hunt(_make_ctx(fuel=700, dual_count=30, radar_count=30)) is True
    assert should_exit_hunt(_make_ctx(fuel=150, dual_count=30, radar_count=30)) is True


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


def test_derive_hunt_mode_state_map_open_without_lock_acquires() -> None:
    """A non-find_enemies map open with no locked target derives ACQUIRE."""
    acquiring = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "dot_relay"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )
    assert derive_hunt_mode_state(acquiring) == "ACQUIRE"


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


def test_derive_hunt_mode_state_maps_unlocked_map_open_to_acquire() -> None:
    """A map_open without a locked target and a non-search reason derives ACQUIRE.

    Defensive: production map_opens during HUNT carry either the
    ``find_enemies`` reason (acquire search) or a locked target
    (REFRESH). A map_open with neither must still land in ACQUIRE.
    """
    decision = make_tick_decision(
        command=make_map_open_command(),
        behavior=make_behavior_score("HUNT", 800, 0, 0, "find_enemies"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )

    assert derive_hunt_mode_state(decision) == "ACQUIRE"


def test_derive_hunt_mode_state_maps_locked_walk_to_close() -> None:
    """A combat walk toward a locked target is a CLOSE transition."""
    decision = make_tick_decision(
        command=make_move_command(103, 100),
        behavior=make_behavior_score("HUNT", 800, 103, 100, "find_target"),
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


# =============================================================================
# resolve_owner_from_manual
# =============================================================================


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


# =============================================================================
# make_hold_decision
# =============================================================================


def _make_hold_inventory(
    dual_count: int = 25,
    homing_count: int = 25,
) -> InventoryState:
    """Build an inventory for the hold-decision equipment checks."""
    item = InventoryItem(count=25, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=InventoryItem(count=dual_count, enabled=True),
        missile_shots=item,
        homing_shots=InventoryItem(count=homing_count, enabled=True),
        extra_radars=item,
    )


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
        state, timestamp_ms=15000, fuel=900, inventory=_make_hold_inventory()
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
    )

    assert decision["desired_equipment"] == [5]


def test_make_hold_decision_preserves_started_ms_when_already_unset() -> None:
    """A UNSET → UNSET transition keeps the earlier ``mode_started_ms``."""
    state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "manual_mode": "UNSET",
            "mode": "UNSET",
            "mode_state": "",
            "mode_started_ms": 8000,
        }
    )

    decision = make_hold_decision(
        state, timestamp_ms=15000, fuel=900, inventory=_make_hold_inventory()
    )

    assert decision["updated_ai_state"]["mode_started_ms"] == 8000


# =============================================================================
# apply_dispatch_counters
# =============================================================================


def _make_decision(
    command: BotCommand,
    ai_state: AIStateDict | None = None,
    *,
    secondary_command: BotCommand | None = None,
) -> TickDecisionDict:
    """Build a minimal :class:`TickDecisionDict` for counter tests.

    Args:
        command: Primary command for the decision.
        ai_state: Optional AI state override; defaults to
            :func:`make_initial_ai_state`.
        secondary_command: Optional secondary command.

    Returns:
        A :class:`TickDecisionDict` suitable for exercising
        :func:`apply_dispatch_counters`.
    """
    state = ai_state if ai_state is not None else make_initial_ai_state()
    return make_tick_decision(
        command=command,
        behavior=make_behavior_score("HUNT", 100, 0, 0, "manual_hold"),
        updated_ai_state=state,
        desired_equipment=[],
        secondary_command=secondary_command,
    )


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


def test_apply_dispatch_counters_preserves_untouched_fields() -> None:
    """Applying counters does not clobber unrelated decision fields."""
    decision = _make_decision(
        make_teleport_command(100, 100),
        secondary_command=make_radar_command(),
    )

    updated = apply_dispatch_counters(decision)

    assert updated["command"] == decision["command"]
    assert updated["behavior"] == decision["behavior"]
    assert updated["desired_equipment"] == decision["desired_equipment"]
    assert updated["secondary_command"] == decision["secondary_command"]
