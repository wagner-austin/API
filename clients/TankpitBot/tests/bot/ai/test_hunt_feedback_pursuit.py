"""Tests for hunt pursuit feedback and lock release."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    TankStateDict,
    make_tank_state,
)
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


class TestDepartedTargetFollowUp:
    """Orange-2 follow-up: released escapees are re-acquired from fresh maps."""

    def test_escapee_is_chased_lock_held_to_its_new_position(self) -> None:
        """The full chase cycle: miss -> map open (lock held) -> teleport back.

        The orange-9 sequence end-to-end (user ruling 2026-07-26): the
        post-reroute-window miss opens the map with the lock HELD; the
        snapshot shows the escapee at its new position; the pursuit
        path teleports back to the SAME locked target -- no fresh
        acquisition, no distance lottery. (Run 194658 lost orange-9 at
        13 banked hits to the distance lottery under the old release.)
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=110,
                y=110,
                team=2,
                rank=1,
                name="red-9",
                is_self=False,
                is_bot=False,
                damage_state=1,
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
                "last_shot_target_name": "red-9",
            }
        )
        inventory = make_inventory()

        chase = decide(world, self_state, ai_state, inventory, 100000, None, "miss", ws=ws)
        assert chase["command"]["cmd_type"] == "map_open"
        assert chase["updated_ai_state"]["combat_target_id"] == 50
        assert chase["updated_ai_state"]["blocked_combat_targets"] == {}

        # Fresh map snapshot: the escapee reappears 20 tiles away with
        # a map-fresh timestamp (off-viewport, so the held lock goes
        # through the pursuit path).
        followed = dict(tanks)
        followed["50"] = make_tank_state(
            tank_id=50,
            x=130,
            y=100,
            team=2,
            rank=1,
            name="red-9",
            is_self=False,
            is_bot=False,
            damage_state=1,
            timestamp_ms=102000,
            last_wire_seen_ms=100000,
            last_position_update_ms=102000,
            last_viewport_observation_ms=0,
        )
        world2, self_state2 = make_world(fuel=900, tanks=followed)
        chase_state = AIStateDict(
            **{
                **chase["updated_ai_state"],
                "mode": "HUNT",
                "mode_state": "ACQUIRE",
                "mode_started_ms": 100000,
                "last_map_open_ms": 102000,
            }
        )

        decision = decide(world2, self_state2, chase_state, inventory, 102500, None, "", ws=ws)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
