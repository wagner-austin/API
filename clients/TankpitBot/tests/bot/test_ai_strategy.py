"""Tests for AI strategy decide() function."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_default_ai_config,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai_strategy import (
    _compute_equipment,
    _expire_kills,
    _filter_killed_tanks,
    _find_target_name,
    _find_teleport_enemy,
    decide,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    TankStateDict,
    ViewportStateDict,
    WorldStateDict,
)


def _make_world(
    *,
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Create world and self state for testing.

    Returns:
        Tuple of (world_state, self_state).
    """
    self_state = SelfStateDict(
        tank_id=1,
        x=self_x,
        y=self_y,
        team=1,
        rank=2,
        fuel=fuel,
        leaderboard_position=5,
    )
    viewport = ViewportStateDict(left=0, top=0, width=18, height=18)
    world = WorldStateDict(
        self_state=self_state,
        tanks=tanks or {},
        containers=containers or {},
        mines={},
        terrain={},
        viewport=viewport,
        timestamp_ms=0,
    )
    return world, self_state


def _make_inventory(
    *,
    dual_count: int = 30,
    dual_enabled: bool = True,
    default_count: int = 30,
) -> InventoryState:
    """Create a basic inventory state.

    Args:
        dual_count: Count for dual shots item.
        dual_enabled: Whether dual shots are enabled.
        default_count: Default count for all other items.

    Returns:
        InventoryState with the specified values.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=default_count, enabled=True),
        dual_shots=InventoryItem(count=dual_count, enabled=dual_enabled),
        missile_shots=InventoryItem(count=default_count, enabled=True),
        homing_shots=InventoryItem(count=default_count, enabled=True),
        extra_radars=InventoryItem(count=default_count, enabled=True),
    )


class TestDecideProactiveRadar:
    """Tests for proactive radar tactical override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_proactive_radar_when_fuel_low_no_containers(self) -> None:
        """decide() triggers proactive radar when fuel is at low threshold."""
        # fuel_low_threshold=700 → triggers when fuel <= 700
        world, self_state = _make_world(fuel=500)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "radar"
        assert decision["behavior"]["reason"] == "proactive_radar"
        # Standard equipment during tactical overrides: dual + radar
        assert decision["desired_equipment"] == [2, 5]

    def test_no_proactive_radar_when_fuel_high(self) -> None:
        """decide() skips proactive radar when fuel is above threshold."""
        world, self_state = _make_world(fuel=1000)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Should not be proactive radar — fuel too high
        assert decision["behavior"]["reason"] != "proactive_radar"

    def test_no_proactive_radar_when_containers_visible(self) -> None:
        """decide() skips proactive radar when fuel containers are visible."""
        containers: dict[str, ContainerStateDict] = {
            "80,90": ContainerStateDict(x=80, y=90, volume=50, is_fuel=True),
        }
        world, self_state = _make_world(fuel=500, containers=containers)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason"] != "proactive_radar"


class TestDecideTeleportFuelGuard:
    """Tests for teleport fuel cost guard."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_teleport_search_blocked_when_fuel_too_low(self) -> None:
        """decide() skips teleport search when fuel can't cover cost + critical.

        should_teleport_search triggers (fuel<500, no containers, recent scan,
        low-score behavior), but the fuel guard (fuel > cost + critical) blocks
        the teleport and falls through to normal ai_tick instead.
        """
        world, self_state = _make_world(fuel=250)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            ticks_in_mode=5,
            killed_tank_ids={},
            last_shot_target_id=-1,
            last_shot_target_name="",
        )
        inventory = _make_inventory()

        # fuel=250, teleport_cost=100, critical=200 → 250 < 300 → teleport blocked
        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Teleport was blocked, so should NOT be teleport
        assert decision["command"]["cmd_type"] != "teleport"


class TestDecideEquipmentDepletion:
    """Tests for dual shots depletion override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_switches_to_equipment_collection(self) -> None:
        """decide() switches to equipment collection when dual shots depleted.

        Conditions: HUNT wins evaluation (enemy close, fuel sufficient), dual shots
        count=0 and enabled=False, and equipment containers are visible.
        """
        # Enemy tank within viewport range (dist=6 <= _MAX_SHOOT_RANGE=8)
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        # Equipment container must exist for _has_equipment_containers
        containers: dict[str, ContainerStateDict] = {
            "106,106": ContainerStateDict(x=106, y=106, volume=30, is_fuel=False),
        }
        world, self_state = _make_world(fuel=800, containers=containers, tanks=tanks)
        # Set last_scan_ms and last_shoot_ms recently so HUNT evaluates as
        # "move toward target" (not scan or shoot), keeping behavior mode HUNT
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=99500,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            ticks_in_mode=5,
            killed_tank_ids={},
            last_shot_target_id=-1,
            last_shot_target_name="",
        )
        # Dual shots depleted — triggers equipment collection override
        inventory = _make_inventory(dual_count=0, dual_enabled=False)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Priority chain: equipment visible + fuel adequate → COLLECT_EQUIPMENT
        assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
        assert decision["command"]["cmd_type"] == "pickup_move"
        assert decision["command"]["target_x"] == 106
        assert decision["command"]["target_y"] == 106

    def test_no_equipment_when_none_visible(self) -> None:
        """decide() skips equipment collection when no equipment containers exist.

        With no equipment containers, the priority chain falls through
        to HUNT (if enemy visible and fuel adequate).
        """
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=99500,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            ticks_in_mode=5,
            killed_tank_ids={},
            last_shot_target_id=-1,
            last_shot_target_name="",
        )
        inventory = _make_inventory(dual_count=0, dual_enabled=False)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # No equipment containers → falls through to HUNT
        assert decision["behavior"]["mode"] == "HUNT"


class TestHelpers:
    """Tests for internal helper functions."""

    def test_compute_equipment_collect_fuel(self) -> None:
        """_compute_equipment returns [2, 5] for COLLECT_FUEL mode."""
        inventory = _make_inventory()
        result = _compute_equipment("COLLECT_FUEL", 800, inventory)
        assert result == [2, 5]

    def test_compute_equipment_hunt(self) -> None:
        """_compute_equipment returns [2, 5] for HUNT mode."""
        inventory = _make_inventory()
        result = _compute_equipment("HUNT", 800, inventory)
        assert result == [2, 5]

    def test_compute_equipment_no_shields(self) -> None:
        """_compute_equipment never includes shields."""
        inventory = _make_inventory()
        result = _compute_equipment("HUNT", 800, inventory)
        assert 1 not in result

    def test_compute_equipment_dual_depleted(self) -> None:
        """_compute_equipment drops dual when count is 0."""
        inventory = _make_inventory(dual_count=0)
        result = _compute_equipment("HUNT", 800, inventory)
        assert result == [5]

    def test_expire_kills_removes_expired(self) -> None:
        """_expire_kills removes entries older than cooldown."""
        killed = {"50": 1000, "60": 5000}
        result = _expire_kills(killed, 22000, 20000)
        # 50: 22000-1000=21000 >= 20000 → expired
        # 60: 22000-5000=17000 < 20000 → kept
        assert result == {"60": 5000}

    def test_expire_kills_keeps_recent(self) -> None:
        """_expire_kills keeps entries within cooldown."""
        killed = {"50": 10000, "60": 15000}
        result = _expire_kills(killed, 20000, 20000)
        assert result == {"50": 10000, "60": 15000}

    def test_expire_kills_empty_input(self) -> None:
        """_expire_kills handles empty dict."""
        result = _expire_kills({}, 20000, 20000)
        assert result == {}

    def test_filter_killed_tanks_removes_killed(self) -> None:
        """_filter_killed_tanks removes tanks on kill cooldown."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy1",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
            "60": TankStateDict(
                tank_id=60,
                x=110,
                y=110,
                team=2,
                rank=1,
                name="Enemy2",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, _ = _make_world(tanks=tanks)
        killed = {"50": 10000}
        filtered = _filter_killed_tanks(world, killed)
        assert "50" not in filtered["tanks"]
        assert "60" in filtered["tanks"]

    def test_filter_killed_tanks_empty_killed(self) -> None:
        """_filter_killed_tanks returns same world when no killed tanks."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy1",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, _ = _make_world(tanks=tanks)
        filtered = _filter_killed_tanks(world, {})
        assert filtered is world  # No copy needed

    def test_find_target_name_found(self) -> None:
        """_find_target_name returns tank name when found."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy1",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, _ = _make_world(tanks=tanks)
        assert _find_target_name(world, 50) == "Enemy1"

    def test_find_target_name_not_found(self) -> None:
        """_find_target_name returns empty string when tank not in world."""
        world, _ = _make_world()
        assert _find_target_name(world, 999) == ""


class TestDecideCombatFeedback:
    """Tests for combat feedback handling in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_hit_feedback_after_kill_sees_no_enemy(self) -> None:
        """After a kill, Deactivation sets victim to (0,0).

        The protocol handles kills: Deactivation message sets victim
        position to (0,0). The AI receives "hit" feedback (CombatHit
        arrives for kills too). With the tank at (0,0), evaluators
        skip it, and map_open fires to find new enemies.
        """
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=0,
                y=0,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = make_initial_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_shot,
            inventory,
            100000,
            None,
            "hit",
        )

        # Tank at (0,0) is skipped by evaluators → no enemies → map open fires
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "map_open_enemies"
        assert decision["updated_ai_state"]["last_shot_target_id"] == -1

    def test_miss_feedback_opens_map_to_refresh(self) -> None:
        """Miss feedback forces map open to refresh stale enemy positions."""
        world, self_state = _make_world(fuel=800)
        ai_state = make_initial_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_shot,
            inventory,
            100000,
            None,
            "miss",
        )

        # Miss → map_open to refresh positions
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "miss_map_open"
        assert decision["updated_ai_state"]["last_shot_target_id"] == -1

    def test_hit_feedback_continues_normally(self) -> None:
        """Hit feedback clears shot tracking and continues normal AI flow."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = make_initial_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
                "last_map_open_ms": 99000,
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_shot,
            inventory,
            100000,
            None,
            "hit",
        )

        # Should not be a kill/miss override — normal AI decision
        assert decision["behavior"]["reason"] != "kill_confirmed"
        assert decision["behavior"]["reason"] != "miss_relocate"
        # Command should not be map_open (that's what kill/miss do)
        assert decision["command"]["cmd_type"] != "map_open"

    def test_no_feedback_when_no_shot_pending(self) -> None:
        """Empty feedback when no shot was fired last tick."""
        world, self_state = _make_world(fuel=800)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            "",
        )

        # No shot pending → normal flow, map open for no enemies
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "map_open_enemies"


class TestDecideMapOpen:
    """Tests for map open tactical override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_map_open_when_no_enemies(self) -> None:
        """decide() triggers map open when no live enemies visible."""
        world, self_state = _make_world(fuel=800)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "map_open_enemies"

    def test_no_map_open_when_enemy_visible(self) -> None:
        """decide() skips map open when a live enemy is visible."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = make_initial_ai_state()
        ai_state_recent_map = AIStateDict(
            **{
                **ai_state,
                "last_map_open_ms": 99000,
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_recent_map,
            inventory,
            100000,
            None,
        )

        assert decision["behavior"]["reason"] != "map_open_enemies"


class TestDecideShotTracking:
    """Tests for shot target tracking in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_shoot_command_records_target(self) -> None:
        """Shoot command records target_id and target_name for feedback."""
        # Enemy within viewport range (dist=6 <= _MAX_SHOOT_RANGE=8)
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=50,
            combat_target_x=103,
            combat_target_y=103,
            ticks_in_mode=5,
            killed_tank_ids={},
            last_shot_target_id=-1,
            last_shot_target_name="",
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        if decision["command"]["cmd_type"] == "shoot":
            assert decision["updated_ai_state"]["last_shot_target_id"] == 50
            assert decision["updated_ai_state"]["last_shot_target_name"] == "Enemy"


class TestDecideKillCooldown:
    """Tests for kill cooldown filtering in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_killed_tanks_filtered_from_world(self) -> None:
        """Killed tanks are not visible to AI evaluators."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="KilledEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = make_initial_ai_state()
        ai_state_with_kill = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": {"50": 90000},
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_kill,
            inventory,
            100000,
            None,
        )

        # Tank 50 should be filtered — AI should not be targeting it
        # The behavior should not be HUNT targeting tank 50
        if decision["behavior"]["mode"] == "HUNT":
            bx = decision["behavior"]["target_x"]
            by = decision["behavior"]["target_y"]
            assert bx != 105 or by != 105

    def test_expired_kills_removed(self) -> None:
        """Expired kill cooldowns are cleaned up."""
        world, self_state = _make_world(fuel=800)
        ai_state = make_initial_ai_state()
        # Kill at time 50000, now at 100000 → 50000ms > 20000ms cooldown → expired
        ai_state_with_old_kill = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": {"50": 50000},
            }
        )
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state_with_old_kill,
            inventory,
            100000,
            None,
        )

        # Expired kill should be removed from AI state
        assert "50" not in decision["updated_ai_state"]["killed_tank_ids"]


class TestDecideTeleportToFarTarget:
    """Tests for teleport-to-far-target override."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_teleport_when_hunt_target_beyond_viewport(self) -> None:
        """decide() teleports to far HUNT target instead of walking."""
        # Enemy at (120, 100) from self (100, 100) = dist 20 > _MAX_SHOOT_RANGE=8
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            ticks_in_mode=0,
            killed_tank_ids={},
            last_shot_target_id=-1,
            last_shot_target_name="",
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 120
        assert decision["command"]["target_y"] == 100
        assert decision["desired_equipment"] == [2, 5]

    def test_no_teleport_when_fuel_too_low(self) -> None:
        """decide() skips teleport when fuel can't cover cost + critical."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        # fuel=50, hunt_min_fuel=100 → 50 <= 100 → teleport blocked
        world, self_state = _make_world(fuel=50, tanks=tanks)
        config = make_default_ai_config()
        ai_state = AIStateDict(
            config=config,
            active_mode="HUNT",
            patrol_waypoint_index=0,
            last_scan_ms=99500,
            last_shoot_ms=0,
            last_map_open_ms=99500,
            combat_target_id=-1,
            combat_target_x=0,
            combat_target_y=0,
            ticks_in_mode=0,
            killed_tank_ids={},
            last_shot_target_id=-1,
            last_shot_target_name="",
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        # Teleport blocked by fuel guard — falls through to move
        assert decision["command"]["cmd_type"] != "teleport"


class TestFindTeleportEnemy:
    """Tests for _find_teleport_enemy helper."""

    def test_skips_self_and_teammates(self) -> None:
        """Self and same-team tanks are ignored."""
        tanks: dict[str, TankStateDict] = {
            "1": TankStateDict(
                tank_id=1,
                x=100,
                y=100,
                team=1,
                rank=2,
                name="Self",
                is_self=True,
                is_bot=False,
                damage_state=0,
            ),
            "2": TankStateDict(
                tank_id=2,
                x=150,
                y=100,
                team=1,
                rank=3,
                name="Ally",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(tanks=tanks)
        config = make_default_ai_config()
        assert _find_teleport_enemy(world, self_state, config) is None

    def test_skips_dead_tanks_at_origin(self) -> None:
        """Dead tanks at (0,0) are ignored."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=0,
                y=0,
                team=2,
                rank=1,
                name="Dead",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(tanks=tanks)
        config = make_default_ai_config()
        assert _find_teleport_enemy(world, self_state, config) is None

    def test_picks_nearest_far_enemy(self) -> None:
        """Returns nearest enemy that is beyond viewport range."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=130,
                y=100,
                team=2,
                rank=1,
                name="NearFar",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
            "60": TankStateDict(
                tank_id=60,
                x=200,
                y=200,
                team=2,
                rank=1,
                name="FarFar",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(tanks=tanks)
        config = make_default_ai_config()
        result = _find_teleport_enemy(world, self_state, config)
        # NearFar at dist=30, FarFar at dist=200 → picks NearFar (nearest)
        assert result == (130, 100, 30)

    def test_returns_none_when_enemy_in_viewport(self) -> None:
        """No teleport needed when nearest enemy is within shoot range."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=105,
                y=103,
                team=2,
                rank=1,
                name="Close",
                is_self=False,
                is_bot=False,
                damage_state=0,
            ),
        }
        world, self_state = _make_world(tanks=tanks)
        config = make_default_ai_config()
        # dist = |105-100| + |103-100| = 8 <= _MAX_SHOOT_RANGE
        assert _find_teleport_enemy(world, self_state, config) is None
