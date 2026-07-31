"""Feedback and cooldown integration tests for HUNT routing."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import TankStateDict, make_container_state, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


class TestDecideCombatFeedback:
    """Tests for combat feedback handling in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_hit_feedback_after_kill_sees_no_enemy(self) -> None:
        """After a kill, deactivated enemies no longer participate in threat selection.

        The death tile is preserved on the tank state -- it's the
        ``liveness="deactivated"`` filter in ``analyze_threats`` that
        keeps the corpse from re-acquiring as a target. Pre-2026-06-20
        this test used ``x=0, y=0`` as the dead-sentinel; that hack is
        replaced by the explicit liveness state machine.
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
                liveness="deactivated",
            ),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = make_scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state_with_shot, inventory, 100000, None, "hit")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_enemies"
        assert decision["updated_ai_state"]["last_shot_target_id"] == -1

    def test_miss_with_no_target_in_world_opens_map(self) -> None:
        """Miss feedback with no target state falls through to reacquisition."""
        world, self_state = make_world(fuel=1200)
        ai_state = make_scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state_with_shot, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_enemies"

    def test_hit_feedback_continues_normally(self) -> None:
        """Hit feedback preserves normal combat routing when the target remains visible.

        The enemy carries a fresh wire-sourced position inside the
        viewport, so normal combat routing engages it from the current
        tile (in-view law) -- neither ``kill_confirmed`` nor
        ``miss_relocate`` short-circuits the decision.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=105,
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
        ai_state = make_scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
                "last_map_open_ms": 94000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state_with_shot, inventory, 100000, None, "hit")

        assert decision["behavior"]["reason_kind"] != "confirm_kill"
        assert decision["behavior"]["reason_kind"] != "find_enemies"
        assert decision["command"]["cmd_type"] == "shoot"

    def test_no_feedback_when_no_shot_pending(self) -> None:
        """Empty combat feedback leaves normal planning unchanged."""
        world, self_state = make_world(fuel=1200)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_enemies"


class TestDecideShotTracking:
    """Tests for shot target tracking in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_shoot_command_records_target(self) -> None:
        """Shoot decisions record the target for next-tick feedback handling."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
                y=103,
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
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        if decision["command"]["cmd_type"] == "shoot":
            assert decision["updated_ai_state"]["last_shot_target_id"] == 50
            assert decision["updated_ai_state"]["last_shot_target_name"] == "red-1"


class TestDecideKillCooldown:
    """Tests for kill cooldown filtering in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_killed_tanks_filtered_from_world(self) -> None:
        """Killed tanks do not remain eligible HUNT targets."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="red-19",
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
                "killed_tank_ids": {"50": 90000},
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        if decision["behavior"]["mode"] == "HUNT":
            assert (
                decision["behavior"]["target_x"],
                decision["behavior"]["target_y"],
            ) != (105, 105)

    def test_miss_on_stationary_far_target_chases_with_lock_held(self) -> None:
        """A consumption-miss on a stationary distant target starts a chase.

        Consumption = hit (user contract 2026-07-02): pursuit homings
        that land arrive as ``weapon>0`` hits and keep the engagement
        alive through the hit path. A genuine miss (weapon=0, nothing
        spent) at a registry position that has not moved means the
        reroute window closed on an escaped target. User ruling
        2026-07-26: live targets are NEVER dropped -- the lock is
        HELD, the map is opened, and HUNT/ACQUIRE's pursuit machinery
        chases the refreshed position; only a dead/vanished registry
        entry releases (the original orange-2 case, 2026-07-20).
        Repeat-shot loops stay impossible because the decision is a
        map open, not a shot (the 2026-07-02 01:23 loop re-fired).
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

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert decision["updated_ai_state"]["blocked_combat_targets"] == {}

    def test_miss_on_adjacent_stationary_target_chases_not_refires(self) -> None:
        """A consumption-miss on an adjacent stationary target opens the map.

        The original same-tile re-engage loop (run 20260611-103244:
        12 shots at a frozen tile) fired again from the same state;
        the chase decision is a map open with the lock held (user
        ruling 2026-07-26), so the loop stays impossible while the
        departed tank is followed up instead of abandoned.
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
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert decision["updated_ai_state"]["blocked_combat_targets"] == {}

    def test_miss_on_moved_target_reaims_and_keeps_lock(self) -> None:
        """A miss on a target that moved since the shot re-aims, not blocks.

        The one ambiguous miss case: a live enemy may have stepped off
        the tile as the shot resolved. The registry shows the new
        position, so the bot re-aims there and keeps the lock.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
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
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_x"] == 103
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_closing_keeps_firing_when_engaged_at_diagonal(self) -> None:
        """Diagonal landing on an engaged target keeps firing, not chasing.

        User contract (2026-06-26): once engaged, the bot stays put and
        fires homing at any non-adjacent position. A diagonal landing
        (distance 2) on an engaged target dispatches another shoot
        rather than spending fuel on a re-close teleport.
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
        world, self_state = make_world(self_x=100, self_y=99, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "red-1",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "")

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["behavior"]["reason_kind"] == "shoot_target"

    def test_closing_shoots_when_cardinally_adjacent(self) -> None:
        """Closing combat engages once the landed position is cardinally usable."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=99,
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
        world, self_state = make_world(self_x=100, self_y=99, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 99,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "")

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["updated_ai_state"]["last_shot_target_id"] == 50

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

        chase = decide(world, self_state, ai_state, inventory, 100000, None, "miss")
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

        decision = decide(world2, self_state2, chase_state, inventory, 102500, None, "")

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
