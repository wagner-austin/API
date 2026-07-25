"""Tests for ai_strategy coverage gaps — locked targets, radar search, exploration."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import reset_world_state, update_world_state_from_position
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_self_state,
    make_viewport_state,
)


def _make_world(
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
    scanned: bool = True,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Build a world state for testing."""
    self_state = make_self_state(
        tank_id=1,
        x=self_x,
        y=self_y,
        team=1,
        rank=0,
        fuel=fuel,
        leaderboard_position=0,
    )
    vp_left = self_x - 8
    vp_top = self_y - 8
    scanned_tiles: dict[str, int] = (
        {
            f"{x},{y}": 100000
            for y in range(vp_top, vp_top + 16)
            for x in range(vp_left, vp_left + 16)
        }
        if scanned
        else {}
    )
    return (
        WorldStateDict(
            self_state=self_state,
            tanks=tanks or {},
            containers=containers or {},
            mines={},
            terrain={},
            viewport=make_viewport_state(left=vp_left, top=vp_top, width=16, height=16),
            scanned_tiles=scanned_tiles,
            timestamp_ms=100000,
        ),
        self_state,
    )


def _c(x: int, y: int, volume: int, is_fuel: bool) -> ContainerStateDict:
    """Create a container state."""
    from tankpit_bot.state.types import make_container_state

    return make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        timestamp_ms=100000,
        failed_pickups=0,
    )


def _make_inventory(
    dual_count: int = 30,
    default_count: int = 30,
    radar_count: int = 30,
) -> InventoryState:
    """Build an inventory."""
    item = InventoryItem(count=default_count, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=InventoryItem(count=dual_count, enabled=True),
        missile_shots=item,
        homing_shots=item,
        extra_radars=InventoryItem(count=radar_count, enabled=True),
    )


def _scanned_ai_state() -> AIStateDict:
    """Build a scanned AI state.

    ``last_landing_scan_viewport`` matches the (92,92) viewport every
    ``_make_world`` builds around position (100,100), so COLLECT's
    unconditional scan-on-landing latch reads as already satisfied and
    tests exercise the downstream cascade steps.
    """
    return AIStateDict(
        **{
            **make_initial_ai_state(),
            "last_landing_scan_viewport": "92,92",
        }
    )


class TestLockedEquipmentTarget:
    """Tests for locked equipment target continuation."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_continues_locked_equipment_target(self) -> None:
        """Locked equipment target is continued when still actionable."""
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        # default_count=15: below low (20) but above break (12) → _try_collect_equipment path
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "equipment_locked"

    def test_locked_equipment_target_on_water_releases_lock(self) -> None:
        """A water-locked equipment target is released by the lock-continuation.

        User contract (2026-06-26): ``walk_or_teleport`` returns
        ``None`` for non-walk-reachable pickup targets. The lock
        continuation path treats that as "no longer executable"
        and releases the lock, dropping the bot through to forage.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers, fuel=800)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["reason_kind"] != "equipment_locked"
        assert decision["command"]["cmd_type"] != "pickup_equipment"

    def test_locked_equipment_target_clears_when_teleport_unaffordable(self) -> None:
        """Locked equipment target is cleared when teleport is unaffordable."""
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # Far target outside viewport — all viewport tiles water so no walkable approach,
        # and the direct teleport is unaffordable.
        containers = {"200,200": _c(200, 200, 0, False)}
        world, self_state = _make_world(containers=containers, fuel=550)
        terrain_data: dict[tuple[int, int], str] = {
            (x, y): "W" for x in range(92, 108) for y in range(92, 108)
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "equipment",
                "resource_target_x": 200,
                "resource_target_y": 200,
            },
        )
        # default_count=15: below low but above break
        inventory = _make_inventory(default_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        # Locked target cleared — teleport to (200,200) costs 1200 fuel, only have 550
        assert decision["behavior"]["reason_kind"] != "equipment_locked"


class TestLockedFuelTarget:
    """Tests for locked fuel target continuation."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_continues_locked_fuel_target(self) -> None:
        """Locked fuel target is continued when still actionable."""
        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=150, containers=containers)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "fuel_locked"
        assert decision["behavior"]["reason_context"]["volume"] == 700


class TestRadarForEquipment:
    """Tests for radar-for-equipment in unscanned viewport."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_forage_radar_in_unscanned_viewport(self) -> None:
        """Equipment recovery forages with a radar in an unscanned viewport.

        The forager owns the scan path regardless of extras count: it
        fires the radar when any viewport tile is unscanned and the
        radar fuel cost is payable.
        """
        world, self_state = _make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
                "last_landing_scan_viewport": "",
            }
        )
        # default_count=15: below low (20) but above break (12) → _try_collect_equipment
        # radar_count=15: above break (12) so _try_search_critical doesn't fire
        inventory = _make_inventory(default_count=15, radar_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"


class TestExplorationSkipsTeleportLowFuel:
    """Tests for exploration rejecting teleport when fuel too low."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_exploration_skips_teleport_when_cant_afford(self) -> None:
        """Exploration skips teleport candidates when fuel reserve too low."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.movement import select_exploration_command
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # All edge tiles are water — only teleport is possible, but fuel is too low
        terrain_data: dict[tuple[int, int], str] = {}
        for x in range(92, 108):
            for y in range(92, 108):
                terrain_data[(x, y)] = "W"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        # Fuel below the short-hop cost so even cheap exploration teleports
        # are unaffordable (reserve gate dropped 2026-06-24; the new floor
        # is raw teleport cost).
        world, self_state = _make_world(fuel=30)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        inventory = _make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

        result = select_exploration_command(ctx)

        # All candidates blocked or teleport-unaffordable → None
        assert result is None


class TestEquipmentSearchHopFallback:
    """Tests for equipment search hop when no target and no radar."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_equipment_search_hops_to_nearest_dot_when_viewport_scanned(self) -> None:
        """Equipment search dot-hops to fresh ground when viewport scanned and no radar."""
        world, self_state = _make_world(fuel=800, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15: below low but above break; radar=0 so no scan
        # radar_count=13: above break (12) so critical path doesn't fire; radar stock
        # doesn't matter for this test since viewport is already scanned
        inventory = _make_inventory(default_count=15, radar_count=13)

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((150, 100),),
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "search_collect_local"
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 150
        assert decision["command"]["target_y"] == 100

    def test_equipment_search_exits_when_no_dot_affordable_at_low_fuel(self) -> None:
        """The COLLECT owner exits ``out_of_fuel`` when marooned at low fuel.

        The only atlas dot is 150 tiles away (teleport cost 900 vs
        fuel 150), fuel is at or below ``fuel_low_threshold`` (200),
        and lock / pickup / sense / hop all decline, so the session
        ends instead of gambling on a blind hop (user contract
        2026-07-02/03).
        """
        import pytest

        world, self_state = _make_world(fuel=150, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        # default_count=15, radar_count=13: above break, viewport scanned → search hop path
        inventory = _make_inventory(default_count=15, radar_count=13)

        with pytest.raises(SessionExitError) as exc_info:
            decide(
                world,
                self_state,
                ai_state,
                inventory,
                100000,
                None,
                map_fuel_dots=((250, 100),),
            )
        assert exc_info.value.reason == "out_of_fuel"

    def test_exhausted_collect_yields_to_hunt_at_healthy_fuel(self) -> None:
        """An exhausted COLLECT cascade hands the tick to HUNT when combat-ready.

        Same marooned setup but fuel 550 > ``fuel_low_threshold``
        (200) AND inventory at combat-ready (Bug 0.4: recruit needs
        duals/homings at ``inventory_capacity(0) = 20`` and radars at
        least ``combat_radar_min(0) = 15``): the tank is stocked, so
        instead of a bogus ``out_of_fuel`` exit (live run 2026-07-06
        exited at fuel 1100 this way) the tick falls through to the
        hunt owner.
        """
        world, self_state = _make_world(fuel=550, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        inventory = _make_inventory(default_count=20, radar_count=15)

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((250, 100),),
        )

        assert decision["behavior"]["mode"] == "HUNT"

    def test_exhausted_collect_under_armed_raises_no_productive_collect(self) -> None:
        """An exhausted COLLECT cascade with under-armed inventory ends the session.

        User contract (Bug 0.4 / Bug 0.7, 2026-07-06): the yield-to-hunt
        gesture requires combat-ready inventory. When fuel is healthy
        but duals/homings/radars sit below their rank-derived caps AND
        no tracked equipment container is affordably teleport-reachable
        (``world.containers`` empty here), COLLECT refuses to hand the
        tick to HUNT and the session exits with
        ``no_productive_collect`` instead of engaging under-armed.
        """
        world, self_state = _make_world(fuel=550, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        import pytest

        inventory = _make_inventory(default_count=15, radar_count=13)

        with pytest.raises(SessionExitError, match="no_productive_collect"):
            decide(
                world,
                self_state,
                ai_state,
                inventory,
                100000,
                None,
                map_fuel_dots=((250, 100),),
            )


class TestCriticalEquipmentLockedTarget:
    """Tests for critical equipment locked target paths."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_continues_locked_critical_equipment_target(self) -> None:
        """Critical locked equipment target is continued when executable."""
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory(default_count=3)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "equipment_locked"

    def test_locked_critical_equipment_target_drives_recovery_owner(self) -> None:
        """A locked critical equipment target stays under COLLECT.

        Earlier the test asserted the lock would clear when the
        underlying tile was water-locked, but the actual behaviour
        under the 2026-06-22 resume-threshold mode-entry rule is
        that COLLECT owns the tick, dispatches a pickup
        attempt at the locked tile, and lets the server's reject
        (e.g. ``Empty container`` / ``You can't go there!``) clear
        the lock via the `_clear_command_error` path. The decision
        on this tick is therefore a normal recovery action, not a
        crash, and the mode label is the contract being asserted.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(containers=containers)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "equipment",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        # default_count=5: every counter below resume → mode-entry triggers.
        inventory = _make_inventory(default_count=5)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT"


class TestFuelSearchFallbacks:
    """Tests for fuel search fallback paths."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_locked_fuel_on_water_releases_lock(self) -> None:
        """A water-locked fuel target is released by the lock-continuation.

        User contract (2026-06-26): no pickup dispatched at a target
        the bot cannot walk to. The lock-continuation releases the
        lock and falls through to forage / search-hop.
        """
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        containers = {"105,105": _c(105, 105, 700, True)}
        world, self_state = _make_world(fuel=150, containers=containers)
        terrain_data: dict[tuple[int, int], str] = {
            (105, 105): "W",
            (104, 105): "W",
            (106, 105): "W",
            (105, 104): "W",
            (105, 106): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert "fuel=700" not in decision["behavior"]["reason_kind"]
        assert decision["command"]["cmd_type"] != "pickup_fuel"

    def test_fuel_search_hop_when_scanned_no_visible_fuel(self) -> None:
        """Fuel search hops to fresh sector when viewport tiles fully swept."""
        world, self_state = _make_world(fuel=150, scanned=True)
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((120, 100),),
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "search_collect_local"
        assert decision["command"]["cmd_type"] == "teleport"

    def test_fuel_raises_when_no_hop_affordable_and_no_atlas_dot(self) -> None:
        """Fuel recovery raises when no productive action remains.

        The viewport-edge walk fallback was removed 2026-06-22 (per-tile
        fuel cost for no visibility gain) and the map_intel terminal
        was removed the same day. With no atlas-known dot, no
        affordable search hop, and a scanned viewport, the bot has
        nothing legal to do; raising surfaces the wedged state loudly.
        """
        import pytest

        # Fuel below the short-hop cost (8 * 6 = 48). The
        # ``hunt_min_fuel`` reserve drop (2026-06-24) means stranding
        # now requires fuel < raw teleport cost.
        world, self_state = _make_world(fuel=30, scanned=True)
        # Recent map open: the dot atlas is empty and a re-open inside
        # the cooldown teaches nothing, so the hop declines.
        ai_state = AIStateDict(**{**_scanned_ai_state(), "last_map_open_ms": 96000})
        inventory = _make_inventory()

        with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
            decide(world, self_state, ai_state, inventory, 100000, None)

    def test_fuel_recovery_raises_when_all_paths_are_blocked(self) -> None:
        """Durable fuel recovery raises when boxed in.

        With every viewport tile already scanned (bot-side coverage),
        every viewport map tile water, no affordable hop, and no fuel
        dots, the bot is genuinely wedged. The map_intel terminal
        was removed 2026-06-22; the owner now raises so the wedged
        state can't be missed in production logs.
        """
        import pytest

        from tests.in_memory_terrain_map import InMemoryTerrainMap

        terrain_data: dict[tuple[int, int], str] = {}
        for x in range(92, 108):
            for y in range(92, 108):
                terrain_data[(x, y)] = "W"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)
        # Fuel below the short-hop cost so no teleport is affordable.
        world, self_state = _make_world(fuel=30, scanned=True)
        # Recent map open: the dot atlas is empty and a re-open inside
        # the cooldown teaches nothing, so the hop declines.
        ai_state = AIStateDict(**{**_scanned_ai_state(), "last_map_open_ms": 96000})
        inventory = _make_inventory()

        with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
            decide(world, self_state, ai_state, inventory, 100000, terrain)


class TestHuntOnlyWhenFull:
    """Tests for the hunt-only-when-full mode-selector contract.

    User contract (2026-07-25): "it should never hunt when its low on
    fuel or equipment. it should never hunt if it is not full on
    everything except -5 max radar." The 2026-07-13 cardinal-adjacent
    override that outranked every reserve check is deleted -- it
    produced the practice-room fight-to-death (bot traded down from
    384 fuel to 0 against a gang-up because an enemy was always one
    tile away). Ignoring an adjacent enemy while collecting is safe:
    bots never initiate, they only return fire.
    """

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()
        update_world_state_from_position(100, 100)

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def _make_cardinal_enemy(self) -> dict[str, TankStateDict]:
        """Return a viewport-fresh live enemy at (101,100), Manhattan 1 from (100,100)."""
        from tankpit_bot.state.types import make_tank_state

        return {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="CardinalEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }

    def _make_distant_enemy(self) -> dict[str, TankStateDict]:
        """Return a viewport-fresh live enemy at (104,100), Manhattan 4 from (100,100)."""
        from tankpit_bot.state.types import make_tank_state

        return {
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="DistantEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }

    def test_cardinal_enemy_cannot_divert_an_understocked_collect_tick(self) -> None:
        """An adjacent enemy never flips an under-stocked COLLECT tick.

        Bot durably in COLLECT with a fuel target lock on (105,105),
        fuel 800 (below the 1100 full threshold), and an enemy one
        tile away at (101,100). Under the deleted 2026-07-13 override
        this tick became HUNT and the bot opened a fight it could not
        fund; under the 2026-07-25 contract COLLECT keeps ownership
        and dispatches the fuel pickup. The adjacent bot is no danger:
        bots never initiate, they only return fire.
        """
        containers = {"105,105": _c(105, 105, 400, True)}
        world, self_state = _make_world(
            fuel=800, containers=containers, tanks=self._make_cardinal_enemy()
        )
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "pickup_fuel"

    def test_full_stock_with_adjacent_enemy_hunts(self) -> None:
        """A fully stocked bot (fuel full, weapons at cap) hunts the
        adjacent enemy through the ordinary selector path -- no
        override needed once readiness is genuine."""
        world, self_state = _make_world(fuel=1100, tanks=self._make_cardinal_enemy())
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"

    def test_understocked_weapons_never_hunt_even_at_full_fuel(self) -> None:
        """Full fuel with duals below cap stays COLLECT -- "never hunt
        if it is not full on everything except -5 max radar"."""
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(
            fuel=1100, containers=containers, tanks=self._make_cardinal_enemy()
        )
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=3)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "COLLECT"
