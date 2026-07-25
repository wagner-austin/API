"""Close-phase integration tests for HUNT routing through decide()."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import combat_landing_tile as _combat_landing_tile
from tankpit_bot.bot.ai.context import DecideCtx, filter_killed_tanks
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestDecideTeleportToFarTarget:
    """Tests for HUNT close/teleport integration through decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_in_range_target_is_shot_on_acquire(self) -> None:
        """A visible in-range target is engaged directly, never approached.

        Shots resolve server-side and never miss in range; requiring
        adjacency (or detouring through map intel) made run
        20260611-083908 chase a moving orange-3 through 30 teleport
        hops without firing.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="NearEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["behavior"]["target_x"] == 101
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_target_position_stale_but_still_in_viewport_opens_map_then_locks(self) -> None:
        """A target with stale position but a fresh viewport observation opens map first.

        ``target_position_is_fresh`` reads the ``timestamp_ms`` of
        the most recent observation by any source. A target whose
        viewport observation is fresh (so it passes the new
        ``analyze_threats`` gate) but whose last position-bearing
        update is 6000 ms old refreshes via map_open before
        committing fuel to a teleport at coordinates the enemy
        may have left. The locked id is set so the next tick
        teleports to the refreshed coords.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=130,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=94000,
                last_wire_seen_ms=100000,
                last_position_update_ms=94000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 94000})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_fresh_target_position_teleports_without_reopening_map(self) -> None:
        """Fresh wire-sourced position allows direct teleport to enemy coordinates.

        The trust signal is per-target ``last_position_update_ms`` --
        when the wire is keeping the target's (x, y) current the bot
        does not need a map_open round-trip before teleport.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=130,
                y=100,
                team=2,
                rank=1,
                name="MappedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=99000,
                last_viewport_observation_ms=99000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 99000})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["target_x"] == 130
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_locked_phase_one_target_teleports_to_existing_enemy(self) -> None:
        """Locked targets teleport directly to the enemy's coordinates.

        The locked enemy sits beyond combat_range (distance 30 > 20),
        so closing teleports instead of shooting; an in-range enemy is
        shot directly (see the in-range test below).
        """
        tanks: dict[str, TankStateDict] = {
            "60": make_tank_state(
                tank_id=60,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="CloserEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
            "50": make_tank_state(
                tank_id=50,
                x=130,
                y=100,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 130,
                "combat_target_y": 100,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["target_x"] == 130
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_locked_target_within_combat_range_is_shot(self) -> None:
        """An in-range locked target is shot at its current position.

        Shots resolve server-side and never miss in range; requiring
        Manhattan adjacency instead made run 20260611-083908 chase a
        moving orange-3 through 30 teleport hops without firing.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["behavior"]["target_x"] == 101
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_locked_phase_one_target_teleports_directly_to_enemy(self) -> None:
        """Close teleport goes directly to the enemy's coordinates; server displaces."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=197,
                y=86,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=180, self_y=80, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 197,
                "combat_target_y": 86,
            }
        )
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(
            terrain_data={
                (197, 86): "W",
                (198, 86): "W",
                (197, 87): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 197
        assert decision["command"]["target_y"] == 86

    def test_locked_phase_one_target_surrounded_by_terrain_is_still_teleported(self) -> None:
        """Even when all adjacent tiles are blocked, teleport goes to target directly."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=197,
                y=86,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=180, self_y=80, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 197,
                "combat_target_y": 86,
            }
        )
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(
            terrain_data={
                (198, 86): "W",
                (196, 86): "W",
                (197, 87): "#",
                (197, 85): "#",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 197
        assert decision["command"]["target_y"] == 86
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_combat_landing_tile_returns_target_coords_at_map_edge(self) -> None:
        """Landing selection returns target coords directly; server handles edge placement."""
        world, self_state = make_world(self_x=10, self_y=10, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        target = EnemyThreatDict(
            tank_id=50,
            x=0,
            y=0,
            distance=20,
            damage_state=0,
            rank=1,
            team=2,
            name="EdgeEnemy",
            is_bot=False,
            timestamp_ms=0,
            last_wire_seen_ms=0,
            last_position_update_ms=0,
            last_aim_x=-1,
            last_aim_y=-1,
            last_aim_weapon=-1,
            last_aim_ms=0,
        )

        landing_x, landing_y = _combat_landing_tile(ctx, target)

        assert (landing_x, landing_y) == (0, 0)

    def test_missing_locked_target_confirms_before_reacquiring_new_enemy(self) -> None:
        """Missing locked targets clear stale combat state before reacquiring."""
        tanks: dict[str, TankStateDict] = {
            "60": make_tank_state(
                tank_id=60,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="NewEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 94000,
                "combat_target_id": 50,
                "combat_target_x": 110,
                "combat_target_y": 100,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "confirm_kill"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_stale_killed_target_is_not_reacquired_from_old_sighting(self) -> None:
        """Older sightings stay suppressed while kill cooldown is active."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="KilledEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=90000,
            ),
        }
        world, _ = make_world(fuel=800, tanks=tanks)

        filtered = filter_killed_tanks(world, {"50": 95000})

        assert "50" not in filtered["tanks"]

    def test_killed_target_can_return_after_newer_sighting(self) -> None:
        """Newer post-kill sightings are allowed back into the threat set."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="RespawnedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=96000,
            ),
        }
        world, _ = make_world(fuel=800, tanks=tanks)

        filtered = filter_killed_tanks(world, {"50": 95000})

        assert "50" in filtered["tanks"]

    def test_engaged_target_at_distance_shoots_instead_of_teleporting(self) -> None:
        """An engaged locked target at distance > 1 stays put and shoots.

        User-contract gameplay loop (2026-06-26): the bot teleports
        cardinally adjacent once on first acquire, fires dual shots
        until the target teleports away, then stays in place and fires
        homing toward the target's last wire position until the kill.
        The server picks ``homing`` when not adjacent and homing tracks,
        so chasing with another teleport burns fuel without changing
        the firing geometry.

        Concretely: enemy moved 5 tiles away (still on the bot's
        viewport) after the dual-shot phase. ``last_shot_target_id ==
        combat_target_id`` proves the bot already engaged this lock,
        so the planner must dispatch ``shoot`` rather than
        ``teleport`` even at distance 5.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=100,
                team=2,
                rank=1,
                name="EngagedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 105,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "EngagedEnemy",
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["behavior"]["target_x"] == 105
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_fresh_acquire_at_distance_teleports_to_close(self) -> None:
        """A never-engaged locked target at distance > 1 teleports to close.

        Companion to the engaged-stay-put case: a fresh acquire
        (``last_shot_target_id`` does not match ``combat_target_id``)
        is the one-time initial close that the engagement contract
        allows. The bot teleports cardinally adjacent so the next
        tick's dual shot resolves at point-blank.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=100,
                team=2,
                rank=1,
                name="FreshEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 105,
                "combat_target_y": 100,
                "last_shot_target_id": -1,
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 105
        assert decision["command"]["target_y"] == 100

    def test_no_teleport_when_fuel_too_low(self) -> None:
        """Teleport close is skipped when fuel cannot satisfy the guard."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=50, tanks=tanks, scanned=False)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 99500})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] != "teleport"
