"""Tests for state mutation functions."""

from tankpit_bot.state import (
    TEAM_BLUE,
    TERRAIN_BLOCK_BRIDGE,
    TERRAIN_FERRY,
    TERRAIN_GROUND,
    WorldStateDict,
    add_mine,
    add_mine_from_radar,
    coord_key,
    deactivate_tank,
    make_container_state,
    make_empty_world_state,
    pickup_container,
    remove_container,
    remove_mine,
    remove_tank,
    set_self_fuel,
    set_self_rank,
    update_container_from_radar,
    update_self_from_movement_response,
    update_terrain_from_viewport,
)
from tankpit_bot.state.types import make_self_state
from tests.world_state.helpers import get_self_state


class TestUpdateSelfFromMovementResponse:
    """Tests for update_self_from_movement_response."""

    def test_creates_self_state(self) -> None:
        """Creates self state from movement response."""
        state = make_empty_world_state()
        updated = update_self_from_movement_response(
            state,
            tank_id=1,
            x=100,
            y=150,
            team=TEAM_BLUE,
            rank=3,
            leaderboard_position=5,
            timestamp_ms=1000,
        )

        self_state = get_self_state(updated)
        assert self_state["tank_id"] == 1
        assert self_state["x"] == 100
        assert self_state["y"] == 150
        assert self_state["team"] == TEAM_BLUE
        assert self_state["rank"] == 3
        assert self_state["leaderboard_position"] == 5
        assert updated["timestamp_ms"] == 1000

    def test_preserves_fuel(self) -> None:
        """Preserves existing fuel value."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=100, y=100, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        # Manually set fuel
        state = WorldStateDict(
            self_state=make_self_state(
                tank_id=1, x=100, y=100, team=0, rank=0, fuel=750, leaderboard_position=1
            ),
            tanks=state["tanks"],
            containers=state["containers"],
            mines=state["mines"],
            terrain=state["terrain"],
            viewport=state["viewport"],
            scanned_tiles=state["scanned_tiles"],
            timestamp_ms=state["timestamp_ms"],
        )

        updated = update_self_from_movement_response(
            state,
            tank_id=1,
            x=110,
            y=110,
            team=0,
            rank=0,
            leaderboard_position=2,
            timestamp_ms=1000,
        )

        self_state = get_self_state(updated)
        assert self_state["fuel"] == 750

    def test_default_fuel_for_new_self(self) -> None:
        """Uses default fuel of 0 for new self state (real fuel comes from TankStatusSync)."""
        state = make_empty_world_state()
        updated = update_self_from_movement_response(
            state,
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=1000,
        )

        self_state = get_self_state(updated)
        assert self_state["fuel"] == 0


# TestUpdateTankFromRegistry, TestWirePresenceFunnel, and
# TestUpdateTankDamage were deleted 2026-06-19 with the freshness-model
# refactor. The deleted mutators are replaced by apply_tank_observation;
# its contract -- including the freshness-funnel rules these classes
# pinned -- lives in tests/world_state/test_tank_observation.py.


class TestUpdateContainerFromRadar:
    """Tests for update_container_from_radar."""

    def test_adds_fuel_container(self) -> None:
        """Adds fuel container from radar."""
        state = make_empty_world_state()
        updated = update_container_from_radar(state, x=100, y=150, volume=500, timestamp_ms=1000)

        key = "100,150"
        assert key in updated["containers"]
        container = updated["containers"][key]
        assert container["x"] == 100
        assert container["y"] == 150
        assert container["is_fuel"] is True
        assert container["volume"] == 500

    def test_adds_equipment_container(self) -> None:
        """Treats -1 volume as equipment."""
        state = make_empty_world_state()
        updated = update_container_from_radar(state, x=50, y=75, volume=-1, timestamp_ms=1000)

        container = updated["containers"]["50,75"]
        assert container["is_fuel"] is False
        assert container["volume"] == 0

    def test_skips_empty_fuel_container(self) -> None:
        """Skips empty fuel containers (volume=0) since they have no contents."""
        state = make_empty_world_state()
        updated = update_container_from_radar(state, x=50, y=75, volume=0, timestamp_ms=1000)

        # Empty fuel containers are skipped, not added
        assert "50,75" not in updated["containers"]

    def test_removes_existing_fuel_container_when_radar_reports_empty(self) -> None:
        """Radar volume=0 clears an already-known fuel container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=75, volume=500, timestamp_ms=500)

        updated = update_container_from_radar(state, x=50, y=75, volume=0, timestamp_ms=1000)

        assert "50,75" not in updated["containers"]

    def test_preserves_failed_pickups_when_refreshing_existing_container(self) -> None:
        """Refreshing a known container keeps failed pickup memory intact."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=150, volume=500, timestamp_ms=500)
        container = state["containers"]["100,150"]
        state["containers"]["100,150"] = make_container_state(
            x=container["x"],
            y=container["y"],
            is_fuel=container["is_fuel"],
            volume=container["volume"],
            timestamp_ms=container["timestamp_ms"],
            failed_pickups=2,
        )

        updated = update_container_from_radar(state, x=100, y=150, volume=500, timestamp_ms=1000)

        assert updated["containers"]["100,150"]["failed_pickups"] == 2


class TestRemoveContainer:
    """Tests for remove_container."""

    def test_removes_existing_container(self) -> None:
        """Removes container from state."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=100, volume=500, timestamp_ms=500)
        updated = remove_container(state, x=100, y=100, timestamp_ms=1000)

        assert "100,100" not in updated["containers"]

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if container doesn't exist."""
        state = make_empty_world_state()
        updated = remove_container(state, x=100, y=100, timestamp_ms=1000)

        assert updated is state  # Same reference


class TestIncrementContainerFailedPickups:
    """Tests for increment_container_failed_pickups (central state-layer mutator)."""

    def test_advances_failed_pickups_by_one(self) -> None:
        """An existing container's failed_pickups counter advances by 1."""
        from tankpit_bot.state import increment_container_failed_pickups

        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=100, volume=500, timestamp_ms=500)
        before = state["containers"]["100,100"]["failed_pickups"]

        updated = increment_container_failed_pickups(state, x=100, y=100)

        assert updated["containers"]["100,100"]["failed_pickups"] == before + 1

    def test_preserves_container_timestamp(self) -> None:
        """The bump must NOT advance the container's freshness timestamp.

        ``failed_pickups`` is a planner-deprioritization counter, not a
        freshness signal. If this test ever flips, planning logic that
        relies on container freshness will silently regress.
        """
        from tankpit_bot.state import increment_container_failed_pickups

        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=100, volume=500, timestamp_ms=500)
        container_ts_before = state["containers"]["100,100"]["timestamp_ms"]

        updated = increment_container_failed_pickups(state, x=100, y=100)

        assert updated["containers"]["100,100"]["timestamp_ms"] == container_ts_before

    def test_returns_unchanged_for_missing_container(self) -> None:
        """No container at ``(x, y)`` -> returned state IS the input state."""
        from tankpit_bot.state import increment_container_failed_pickups

        state = make_empty_world_state()
        updated = increment_container_failed_pickups(state, x=200, y=300)

        assert updated is state


class TestAddMine:
    """Tests for add_mine."""

    def test_adds_mine(self) -> None:
        """Adds mine to state."""
        state = make_empty_world_state()
        updated = add_mine(state, x=75, y=125, mine_type=1, tank_id=42, team=0, timestamp_ms=1000)

        key = "75,125"
        assert key in updated["mines"]
        mine = updated["mines"][key]
        assert mine["x"] == 75
        assert mine["y"] == 125
        assert mine["mine_type"] == 1
        assert mine["tank_id"] == 42
        assert mine["team"] == 0


class TestAddMineFromRadar:
    """Tests for add_mine_from_radar."""

    def test_adds_radar_mine(self) -> None:
        """Adds mine discovered via radar."""
        state = make_empty_world_state()
        updated = add_mine_from_radar(state, x=45, y=203, team=0, timestamp_ms=1000)

        key = "45,203"
        assert key in updated["mines"]
        mine = updated["mines"][key]
        assert mine["x"] == 45
        assert mine["y"] == 203
        assert mine["team"] == 0
        assert mine["mine_type"] == 0  # Unknown from radar
        assert mine["tank_id"] == -1  # Unknown from radar

    def test_adds_multiple_radar_mines(self) -> None:
        """Adds multiple radar-discovered mines."""
        state = make_empty_world_state()
        state = add_mine_from_radar(state, x=45, y=203, team=0, timestamp_ms=1000)
        state = add_mine_from_radar(state, x=46, y=203, team=0, timestamp_ms=1000)
        state = add_mine_from_radar(state, x=47, y=203, team=0, timestamp_ms=1000)

        assert len(state["mines"]) == 3
        assert "45,203" in state["mines"]
        assert "46,203" in state["mines"]
        assert "47,203" in state["mines"]

    def test_adds_mines_from_different_teams(self) -> None:
        """Adds mines from different teams."""
        state = make_empty_world_state()
        state = add_mine_from_radar(state, x=10, y=10, team=0, timestamp_ms=1000)  # red
        state = add_mine_from_radar(state, x=20, y=20, team=1, timestamp_ms=1000)  # purple
        state = add_mine_from_radar(state, x=30, y=30, team=2, timestamp_ms=1000)  # blue
        state = add_mine_from_radar(state, x=40, y=40, team=3, timestamp_ms=1000)  # orange

        assert state["mines"]["10,10"]["team"] == 0
        assert state["mines"]["20,20"]["team"] == 1
        assert state["mines"]["30,30"]["team"] == 2
        assert state["mines"]["40,40"]["team"] == 3


class TestAddMineFromRadarPreservesWireFields:
    """Radar refresh must NOT clobber wire-known mine_type / tank_id.

    Radar 3-byte mine entries (per V.O / 0x4F tunneled, see
    wiki/pages/v-table-complete.md) carry only x, y, team. Wire
    MinePlacement (V.K / 0x4B per Dg.h) carries mine_type and tank_id.
    A radar refresh of a wire-placed mine MUST preserve the
    wire-richer fields. Locked here as a contract.
    """

    def test_radar_refresh_preserves_wire_mine_type(self) -> None:
        """A radar refresh keeps the wire-placed mine_type intact."""
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["mine_type"] == 2

    def test_radar_refresh_preserves_wire_tank_id(self) -> None:
        """A radar refresh keeps the wire-placed placer tank_id intact."""
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["tank_id"] == 42

    def test_radar_refresh_preserves_wire_source_label(self) -> None:
        """A wire-known mine refreshed by radar stays marked as viewport-sourced.

        The mine_type and tank_id remain wire-richer values, so the
        source label that flags them as wire-richer must stick.
        """
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["source"] == "viewport"

    def test_radar_refresh_advances_timestamp(self) -> None:
        """A radar refresh still advances the mine's timestamp."""
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["timestamp_ms"] == 1500

    def test_radar_refresh_updates_team_when_radar_disagrees(self) -> None:
        """When radar reports a different team for an existing mine,
        the radar value wins. Team cannot legally change for an
        undetonated mine, so a discrepancy indicates the wire team
        field went stale and is re-synced from radar.
        """
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=0, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=3, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["team"] == 3

    def test_radar_refresh_of_radar_mine_preserves_radar_source(self) -> None:
        """Radar-then-radar refresh keeps source='radar' (no wire history)."""
        state = make_empty_world_state()
        state = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["source"] == "radar"
        assert refreshed["mines"]["50,50"]["mine_type"] == 0
        assert refreshed["mines"]["50,50"]["tank_id"] == -1


class TestRemoveMine:
    """Tests for remove_mine."""

    def test_removes_existing_mine(self) -> None:
        """Removes mine from state."""
        state = make_empty_world_state()
        state = add_mine(state, x=75, y=125, mine_type=1, tank_id=42, team=0, timestamp_ms=500)
        updated = remove_mine(state, x=75, y=125, timestamp_ms=1000)

        assert "75,125" not in updated["mines"]

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if mine doesn't exist."""
        state = make_empty_world_state()
        updated = remove_mine(state, x=75, y=125, timestamp_ms=1000)

        assert updated is state  # Same reference


class TestUpdateTerrainFromViewport:
    """Tests for update_terrain_from_viewport."""

    def test_updates_terrain_tiles(self) -> None:
        """Updates terrain from viewport entities."""
        state = make_empty_world_state()
        entities = [
            (0, 0, TERRAIN_GROUND, 0, 255),
            (1, 0, TERRAIN_BLOCK_BRIDGE, -1, 3),
            (2, 0, TERRAIN_FERRY, 25, 255),
        ]
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=entities, timestamp_ms=1000
        )

        assert "99,49" in updated["terrain"]
        assert "100,49" in updated["terrain"]
        assert "101,49" in updated["terrain"]

        assert updated["terrain"]["99,49"]["terrain_type"] == TERRAIN_GROUND
        assert updated["terrain"]["100,49"]["terrain_type"] == TERRAIN_BLOCK_BRIDGE
        assert updated["terrain"]["101,49"]["terrain_type"] == TERRAIN_FERRY

    def test_updates_viewport_position(self) -> None:
        """Updates viewport position."""
        state = make_empty_world_state()
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["viewport"]["left"] == 100
        assert updated["viewport"]["top"] == 50

    def test_does_not_touch_scan_coverage(self) -> None:
        """0x5A viewport patches carry terrain only -- never scan coverage.

        Only the wire-side radar handler writes ``scanned_tiles`` -- a
        viewport patch confirms terrain but says nothing about whether
        containers / mines on those tiles were revealed.
        """
        state = make_empty_world_state()

        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["scanned_tiles"] == {}


class TestRemoveTank:
    """Tests for remove_tank (the 0x58 TankRemove handler)."""

    def test_keeps_tank_in_registry(self) -> None:
        """0x58 leaves the registry entry intact (changed 2026-06-22).

        Earlier behaviour deleted the tank, which caused the bot to
        abandon pursuit of locked targets that merely teleported out
        of viewport (live capture 2026-06-22). 0x58 is benign tracking
        churn: orange-5 got 5 TankRemove events across 2 actual kills
        (ghost_observe capture 2026-06-20). Only 0x41 Deactivation is
        an authoritative death signal; keeping 0x58 a no-op lets the
        freshness / liveness gates do the work.
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=42,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(50, 50),
                team=0,
                rank=1,
                name="Test",
                is_bot=False,
            ),
        )
        assert state["tanks"]["42"]["liveness"] == "alive"

        updated = remove_tank(state, tank_id=42, timestamp_ms=1000)

        assert "42" in updated["tanks"]
        assert updated["tanks"]["42"]["liveness"] == "alive"
        assert updated is state

    def test_no_op_for_nonexistent_tank(self) -> None:
        """A 0x58 for a tank we have never heard of is also a no-op."""
        state = make_empty_world_state()
        updated = remove_tank(state, tank_id=999, timestamp_ms=1000)

        assert updated is state


class TestDeactivateTank:
    """Tests for deactivate_tank (0x41 corpse-window handler)."""

    def test_marks_tank_deactivated(self) -> None:
        """Existing tank flips to ``liveness="deactivated"`` and keeps tile.

        Replays the 2026-06-20 ghost_visual kill cycle at the
        world-state layer: TankEntry establishes orange-8 at
        (170, 174); Deactivation marks it ``deactivated`` while
        preserving the death tile (the bot still reasons about that
        tile for mines, fuel deposits, etc.).
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=534,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(170, 174),
                team=3,
                rank=2,
                name="orange-8",
                is_bot=True,
            ),
        )
        assert state["tanks"]["534"]["liveness"] == "alive"

        updated = deactivate_tank(state, tank_id=534, timestamp_ms=1000)

        tank = updated["tanks"]["534"]
        assert tank["liveness"] == "deactivated"
        assert tank["x"] == 170
        assert tank["y"] == 174
        assert tank["timestamp_ms"] == 1000

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Deactivating an unknown tank id is a no-op."""
        state = make_empty_world_state()
        updated = deactivate_tank(state, tank_id=999, timestamp_ms=1000)
        assert updated is state


# TestUpdateSelfFuel was deleted 2026-06-19 with the additive-delta
# update_self_fuel mutator. Production wire fuel messages all funnel
# through set_self_fuel (absolute value); the delta variant had no
# callers in src/.


class TestSetSelfFuel:
    """Tests for set_self_fuel mutation."""

    def test_set_self_fuel_no_self_state(self) -> None:
        """Returns unchanged state when self_state is None."""
        state = make_empty_world_state()
        result = set_self_fuel(state, 50, 1000)
        assert result["self_state"] is None
        assert result is state

    def test_set_self_fuel_sets_absolute_value(self) -> None:
        """Sets fuel to absolute value."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )

        result = set_self_fuel(state, 250, 1000)
        assert get_self_state(result)["fuel"] == 250

    def test_set_self_fuel_clamps_negative(self) -> None:
        """Clamps negative values to zero."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )

        result = set_self_fuel(state, -10, 1000)
        assert get_self_state(result)["fuel"] == 0


class TestSetSelfRank:
    """Tests for set_self_rank mutation (driven by 0x2B Rf Promotion)."""

    def test_returns_unchanged_when_no_self_state(self) -> None:
        """No-op until self has joined: rank can't precede a self_state."""
        state = make_empty_world_state()
        result = set_self_rank(state, 5, 1000)
        assert result["self_state"] is None
        assert result is state

    def test_sets_absolute_rank_and_preserves_other_fields(self) -> None:
        """Promotion lifts rank only; team/position/fuel/lb stay intact."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=42, x=10, y=20, team=1, rank=2, leaderboard_position=7, timestamp_ms=500
        )
        state = set_self_fuel(state, 175, 600)

        result = set_self_rank(state, 5, 1000)

        promoted = get_self_state(result)
        assert promoted["rank"] == 5
        assert promoted["tank_id"] == 42
        assert promoted["x"] == 10
        assert promoted["y"] == 20
        assert promoted["team"] == 1
        assert promoted["fuel"] == 175
        assert promoted["leaderboard_position"] == 7

    def test_timestamp_advances(self) -> None:
        """Mutation stamps the new world-state timestamp."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=0, y=0, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        result = set_self_rank(state, 3, 1234)
        assert result["timestamp_ms"] == 1234


class TestPickupContainer:
    """Tests for pickup_container mutation."""

    def test_pickup_container_removes_container(self) -> None:
        """Removes container from world state."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=500)
        assert coord_key(50, 60) in state["containers"]

        result = pickup_container(state, 50, 60, 1000)
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_container_does_not_modify_self_fuel(self) -> None:
        """``pickup_container`` only touches the container registry, not fuel.

        Fuel updates flow through the wire's absolute-fuel messages
        (0x44 FuelGain / 0x2E TankStatusSync / 0x64 FuelDeposit), which
        call :func:`set_self_fuel` separately. Adding ``transferred``
        fuel here on top of the wire's absolute total double-counts the
        pickup -- a 438-volume container produced a +438 ghost beyond
        the wire's already-correct 633 fuel in live run 20260623-0035
        before this branch was removed.
        """
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=600)

        result = pickup_container(state, 50, 60, 700)

        assert get_self_state(result)["fuel"] == initial_fuel
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_equipment_container_no_fuel_change(self) -> None:
        """Equipment-container pickup also leaves fuel untouched."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]
        state = update_container_from_radar(state, x=50, y=60, volume=-1, timestamp_ms=600)

        result = pickup_container(state, 50, 60, 700)

        assert get_self_state(result)["fuel"] == initial_fuel
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_nonexistent_container(self) -> None:
        """Picking up nonexistent container just returns state with no change."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]

        result = pickup_container(state, 99, 99, 700)
        assert get_self_state(result)["fuel"] == initial_fuel

    def test_pickup_container_no_self_state(self) -> None:
        """Picking up container without self_state still removes container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=500)

        result = pickup_container(state, 50, 60, 700)
        assert coord_key(50, 60) not in result["containers"]
        assert result["self_state"] is None


class TestSetTankLastAim:
    """Tests for ``set_tank_last_aim`` -- the 0x53 ShootEvent persistence path."""

    def test_unknown_tank_is_a_no_op(self) -> None:
        """A shoot event from a tank we have never seen leaves state unchanged.

        The next per-tank wire message will create the tank record;
        dropping the aim quietly is preferable to fabricating a tank
        from a shoot-event alone (no team / rank / name information).
        """
        from tankpit_bot.state.mutations import set_tank_last_aim

        state = make_empty_world_state()

        result = set_tank_last_aim(
            state,
            tank_id=999,
            aim_x=100,
            aim_y=120,
            weapon=1,
            timestamp_ms=5000,
        )

        assert result is state
