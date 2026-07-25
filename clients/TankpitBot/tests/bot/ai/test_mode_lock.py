"""Integration tests for durable mode-lock behavior in ai_strategy."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.state.types import ContainerStateDict, make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def test_unset_mode_enters_hunt_after_hunt_decision() -> None:
    """A legacy HUNT decision seeds durable HUNT ownership for later ticks."""
    world, self_state = make_world(fuel=1200)
    ai_state = make_scanned_ai_state()
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_hunt_mode_owns_tick_without_running_recovery_chain() -> None:
    """Active HUNT mode persists across ticks and keeps its original start time."""
    world, self_state = make_world(fuel=1200)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode_started_ms"] == 90000


def test_hunt_mode_switches_to_recover_fuel_when_recovery_takes_priority() -> None:
    """HUNT yields directly to fuel recovery when recovery takes priority."""
    containers: dict[str, ContainerStateDict] = {
        "95,95": make_container_state(
            x=95,
            y=95,
            is_fuel=True,
            volume=700,
            timestamp_ms=100000,
            failed_pickups=0,
        )
    }
    world, self_state = make_world(fuel=150, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "PICKUP"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_invalid_mode_state_is_ignored_and_reselected() -> None:
    """Invalid durable mode state does not crash; owner selection re-evaluates cleanly."""
    world, self_state = make_world(fuel=1200)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode_state"] == "ACQUIRE"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_invalid_mode_state_reselects_recover_fuel_when_low_fuel_demands_it() -> None:
    """Invalid mode state still reselects fuel recovery when fuel is low."""
    containers: dict[str, ContainerStateDict] = {
        "102,101": make_container_state(
            x=102,
            y=101,
            is_fuel=True,
            volume=700,
            timestamp_ms=100000,
            failed_pickups=0,
        )
    }
    world, self_state = make_world(fuel=150, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "PICKUP"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_invalid_mode_state_reselects_recover_equipment_when_reserves_are_broken() -> None:
    """Invalid mode state still reselects equipment recovery when reserves are broken."""
    world, self_state = make_world(fuel=800, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 5

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "SENSE"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_invalid_mode_state_with_locked_combat_target_migrates_into_hunt() -> None:
    """Invalid mode state with a combat lock is migrated into durable HUNT.

    The lock's target is nowhere in the registry -- genuinely gone --
    so the resume path (2026-07-25 contract: pursue a surviving
    off-viewport lock, never abandon it) releases this one and
    searches fresh.
    """
    world, self_state = make_world(fuel=800)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 120,
            "combat_target_y": 100,
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_unset_mode_enters_recover_equipment_after_equipment_decision() -> None:
    """A legacy equipment recovery decision seeds durable recovery ownership."""
    world, self_state = make_world(fuel=800, scanned=False)
    ai_state = make_scanned_ai_state(landing_scan_viewport="")
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 5

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "SENSE"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_hunt_mode_switches_to_recover_equipment_when_reserves_break() -> None:
    """Active HUNT yields directly to equipment recovery when reserves break."""
    world, self_state = make_world(fuel=800, scanned=False)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 5

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "SENSE"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_unset_mode_enters_recover_fuel_after_fuel_decision() -> None:
    """A legacy fuel recovery decision seeds durable fuel ownership."""
    containers: dict[str, ContainerStateDict] = {
        "102,101": make_container_state(
            x=102,
            y=101,
            is_fuel=True,
            volume=700,
            timestamp_ms=100000,
            failed_pickups=0,
        )
    }
    world, self_state = make_world(fuel=150, containers=containers)
    ai_state = make_scanned_ai_state()
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "PICKUP"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_collect_mode_owns_tick_below_full_threshold() -> None:
    """Fuel recovery persists until the configured full threshold is reached.

    Container volume kept small so ``_pickup_not_worth_walk`` does not refuse
    the pickup: corporal cap 1200, fuel 800, walk 3 tiles, volume 200
    --> 800 + 3 + 200 = 1003 <= 1200. Overflow-refusal is covered by
    the ``_pickup_not_worth_walk`` tests in test_collect_mode_fuel.py.
    """
    containers: dict[str, ContainerStateDict] = {
        "102,101": make_container_state(
            x=102,
            y=101,
            is_fuel=True,
            volume=200,
            timestamp_ms=100000,
            failed_pickups=0,
        )
    }
    world, self_state = make_world(fuel=800, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "last_scan_ms": 1,
            "last_landing_scan_viewport": "92,92",
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "PICKUP"
    assert decision["updated_ai_state"]["mode_started_ms"] == 90000


def test_collect_mode_switches_to_hunt_after_full_recovery() -> None:
    """Fuel recovery hands control directly to HUNT after full recovery."""
    world, self_state = make_world(fuel=1200)
    ai_state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_scan_ms": 1,
        }
    )
    inventory = make_inventory()

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode_state"] == "ACQUIRE"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000


def test_collect_mode_owns_tick_below_resume_threshold() -> None:
    """Equipment recovery persists after break has cleared but resume has not."""
    containers: dict[str, ContainerStateDict] = {
        "102,101": make_container_state(
            x=102,
            y=101,
            is_fuel=False,
            volume=30,
            timestamp_ms=100000,
            failed_pickups=0,
        )
    }
    world, self_state = make_world(fuel=800, containers=containers)
    ai_state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "last_scan_ms": 1,
            "last_landing_scan_viewport": "92,92",
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 15

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["updated_ai_state"]["mode"] == "COLLECT"
    assert decision["updated_ai_state"]["mode_state"] == "PICKUP"
    assert decision["updated_ai_state"]["mode_started_ms"] == 90000


def test_collect_mode_switches_to_hunt_after_full_restock() -> None:
    """COLLECT hands control to HUNT once the stock is genuinely full.

    Contract 2026-07-25: weapons at the rank cap (30 at rank 2) and
    radars within 5 of cap -- the old resume thresholds (25) no
    longer release the mode.
    """
    world, self_state = make_world(fuel=1200)
    ai_state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "last_scan_ms": 1,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 30
    inventory["homing_shots"]["count"] = 30
    inventory["extra_radars"]["count"] = 25

    decision = decide(world, self_state, ai_state, inventory, 100000, None)

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["mode_state"] == "ACQUIRE"
    assert decision["updated_ai_state"]["mode_started_ms"] == 100000
