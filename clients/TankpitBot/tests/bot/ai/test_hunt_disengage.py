"""Tests for hunt disengagement: ammo exhaustion, disabled weapons, and
rejected shots.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
)
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import (
    TankStateDict,
    make_container_state,
    make_tank_state,
)
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


class TestHuntDisengage:
    """Tests for hunt disengagement: ammo exhaustion, disabled weapons, and"""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_ammo_exhaustion_mid_fight_disengages_to_collect_keeping_the_lock(self) -> None:
        """An ammo-exhausted tick disengages to COLLECT without blocking.

        Historical Bug 0.6 (2026-07-06 22:39/22:40): the cardinal-shot
        override kept an ammo-exhausted bot in HUNT and the
        stationary-miss classifier wrongfully blacklisted a live
        target. Under the 2026-07-25 contract the weapon break yanks
        the tick to COLLECT at the mode selector itself (the override
        is deleted), so no under-armed shot is classified at all. The
        combat lock is RETAINED through the restock cycle -- damage
        persists, and the bot returns to the same target at full
        stock -- and the live target is never blacklisted. The fuel
        container at (100,101) gives the cascade a legal recovery
        action.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="red-1",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        containers = {
            "100,101": make_container_state(
                x=100,
                y=101,
                is_fuel=True,
                volume=100,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks, containers=containers)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        # Ammo exhausted: dual + homing both at 0 -- the weapon break
        # takes the tick regardless of the enemy one tile away.
        exhausted_slot = InventoryItem(count=0, enabled=True)
        stocked_slot = InventoryItem(count=30, enabled=True)
        inventory = InventoryState(
            armor_shields=stocked_slot,
            dual_shots=exhausted_slot,
            missile_shots=stocked_slot,
            homing_shots=exhausted_slot,
            extra_radars=stocked_slot,
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] != "shoot"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert "50" not in decision["updated_ai_state"]["blocked_combat_targets"]

    def test_disabled_weapons_miss_disengages_without_blocking(self) -> None:
        """A weapon=0 miss with stocked-but-DISABLED weapons refuels, no block.

        The count-based break thresholds cannot see disabled slots, so
        HUNT keeps the tick; the stationary-miss classifier must still
        recognise the ``ammo_exhaustion_miss`` (no dual or homing can
        fire) and route to fuel recovery instead of blacklisting a
        live target.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="red-1",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        containers = {
            "100,101": make_container_state(
                x=100,
                y=101,
                is_fuel=True,
                volume=100,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks, containers=containers)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        disabled_slot = InventoryItem(count=30, enabled=False)
        stocked_slot = InventoryItem(count=30, enabled=True)
        inventory = InventoryState(
            armor_shields=stocked_slot,
            dual_shots=disabled_slot,
            missile_shots=stocked_slot,
            homing_shots=disabled_slot,
            extra_radars=stocked_slot,
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] != "shoot"
        assert "50" not in decision["updated_ai_state"]["blocked_combat_targets"]

    def test_rejected_shot_blocks_target_and_replans(self) -> None:
        """A server-rejected shot blocks the target instead of redispatching.

        The 0x52 rejection ("You can't do this") means the server
        refused the dispatch outright -- no ShootEvent, no ammo delta.
        With the viewport-clamped aim every dispatch is legal, so a
        residual rejection means the server refuses this engagement
        geometry for a reason the bot cannot see; repeating the
        identical shot cannot change the answer (live run 2026-07-03
        20:34: five identical redispatches at 4 s of dead wait each).
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=110,
                y=110,
                team=2,
                rank=1,
                name="red-1",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 110,
                "combat_target_y": 110,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "rejected")

        assert decision["command"]["cmd_type"] != "shoot"
        assert decision["updated_ai_state"]["combat_target_id"] == -1
        assert "50" in decision["updated_ai_state"]["blocked_combat_targets"]

    def test_expired_kills_removed(self) -> None:
        """Expired kill cooldown entries are removed from the updated AI state."""
        world, self_state = make_world(fuel=1200)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "killed_tank_ids": {"50": 50000},
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert "50" not in decision["updated_ai_state"]["killed_tank_ids"]
