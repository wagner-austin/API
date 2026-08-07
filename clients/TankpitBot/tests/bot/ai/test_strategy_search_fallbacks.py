"""Tests for the strategy search fallbacks.

What the cascade does when the preferred equipment or fuel target is
unreachable.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.sniffer.world_state import (
    reset_world_state,
    update_world_state_from_position,
)
from tests.bot.ai._strategy_fixtures import (
    _c,
    _make_inventory,
    _make_world,
    _scanned_ai_state,
)


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

    def test_wind_down_fully_stocked_exits_session_complete(self) -> None:
        """Winding down at full bars ends the session cleanly.

        User request 2026-07-26: "run and then collect and exit
        cleanly, instead of the program killing it mid action" — the
        tick loop raises the flag in the final stretch and the mode
        selector converts fully-stocked into ``session_complete``.
        """
        import pytest

        from tankpit_bot.bot.session_exit import SessionExitError

        world, self_state = _make_world(fuel=1100, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
                "wind_down": True,
            }
        )
        inventory = _make_inventory(default_count=25, radar_count=25)

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

        assert exc_info.value.reason == "session_complete"

    def test_wind_down_breaks_a_held_hunt_and_collects(self) -> None:
        """Winding down below full bars disengages HUNT into COLLECT."""
        world, self_state = _make_world(fuel=550, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "wind_down": True,
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
            map_fuel_dots=((150, 100),),
        )

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_wind_down_finishes_a_live_kill_before_collecting(self) -> None:
        """A fight in progress completes before the wind-down collects.

        User rulings 2026-07-25/26: never abandon a target mid-fight;
        the kill boundary is the clean-exit point. The held HUNT with
        a LIVE locked target keeps the tick while the break
        thresholds stay green.
        """
        from tankpit_bot.state.types import make_tank_state

        world, self_state = _make_world(fuel=900, scanned=True)
        world["tanks"]["511"] = make_tank_state(
            tank_id=511,
            x=101,
            y=100,
            team=1,
            rank=1,
            name="WindDownTarget",
            is_self=False,
            is_bot=True,
            damage_state=0,
            timestamp_ms=99500,
            last_wire_seen_ms=99500,
            last_position_update_ms=99500,
            last_viewport_observation_ms=99500,
        )
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "combat_target_id": 511,
                "wind_down": True,
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
            map_fuel_dots=((150, 100),),
        )

        assert decision["behavior"]["mode"] == "HUNT"

    def test_wind_down_with_collect_exhausted_exits_session_complete(self) -> None:
        """Winding down with nothing collectable ends the session early-clean."""
        import pytest

        from tankpit_bot.bot.session_exit import SessionExitError

        world, self_state = _make_world(fuel=550, scanned=True)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "mode_started_ms": 90000,
                "wind_down": True,
            }
        )
        inventory = _make_inventory(default_count=20, radar_count=15)

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

        assert exc_info.value.reason == "session_complete"

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
        world, self_state = _make_world(fuel=150, scanned=True, block_scanned=False)
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
