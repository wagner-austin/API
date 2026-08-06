"""Tests for per-tile tank-body occupancy."""

from __future__ import annotations

from tankpit_bot.state.occupancy import is_tank_body_present, occupied_tank_keys
from tankpit_bot.state.types import (
    VIEWPORT_PRESENCE_TTL_MS,
    TankStateDict,
    make_empty_world_state,
    make_tank_state,
)
from tankpit_bot.state.types.constants import TankLiveness

_NOW_MS = 1_000_000


def _tank(
    *,
    tank_id: int = 7,
    x: int = 10,
    y: int = 20,
    is_self: bool = False,
    liveness: TankLiveness = "alive",
    last_viewport_observation_ms: int = _NOW_MS,
    last_position_update_ms: int = 0,
) -> TankStateDict:
    """Build a tank registry entry for occupancy tests.

    Args:
        tank_id: Unique tank identifier.
        x: Tile X coordinate.
        y: Tile Y coordinate.
        is_self: Whether the entry is the bot's own tank.
        liveness: Lifecycle state (``alive`` / ``deactivated`` /
            ``removed``).
        last_viewport_observation_ms: Timestamp of the last
            viewport-sourced observation.
        last_position_update_ms: Timestamp of the last authoritative
            position message (zero = never; the default (10, 20)
            coordinates still count as known via the non-zero check).

    Returns:
        Tank state carrying exactly the fields occupancy reads.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=0,
        rank=1,
        damage_state=3,
        name="red-1",
        is_bot=True,
        is_self=is_self,
        liveness=liveness,
        last_viewport_observation_ms=last_viewport_observation_ms,
        last_position_update_ms=last_position_update_ms,
    )


class TestIsTankBodyPresent:
    """Tests for the per-tank occupancy predicate."""

    def test_viewport_fresh_alive_tank_occupies_its_tile(self) -> None:
        """A live tank seen in the viewport just now blocks its tile."""
        assert is_tank_body_present(_tank(), _NOW_MS) is True

    def test_own_tank_never_occupies(self) -> None:
        """The bot is not blocked by its own body."""
        assert is_tank_body_present(_tank(is_self=True), _NOW_MS) is False

    def test_corpse_does_not_occupy_its_tile(self) -> None:
        """A deactivated tank never blocks -- corpses are walkable.

        Archive-disproven 2026-08-04: six 0x47 echoes of the bot
        walking ONTO fresh corpse tiles 2-10 s after its own kills,
        zero blocked crossings. Kills drop no loot -- the crossings
        are ordinary post-kill restock collection routes, so counting
        corpses would veto the bot's own restock walks.
        """
        assert is_tank_body_present(_tank(liveness="deactivated"), _NOW_MS) is False

    def test_stale_viewport_observation_does_not_occupy(self) -> None:
        """Past the presence TTL the tank may have walked away."""
        stale = _tank(last_viewport_observation_ms=_NOW_MS - VIEWPORT_PRESENCE_TTL_MS - 1)
        assert is_tank_body_present(stale, _NOW_MS) is False

    def test_observation_exactly_at_the_ttl_still_occupies(self) -> None:
        """The TTL boundary is inclusive."""
        edge = _tank(last_viewport_observation_ms=_NOW_MS - VIEWPORT_PRESENCE_TTL_MS)
        assert is_tank_body_present(edge, _NOW_MS) is True

    def test_never_viewport_confirmed_tank_does_not_occupy(self) -> None:
        """A map-only roster entry (stamp 0) is not a local body."""
        assert is_tank_body_present(_tank(last_viewport_observation_ms=0), _NOW_MS) is False

    def test_login_roster_phantom_does_not_occupy(self) -> None:
        """A 0x21-created tank is viewport-fresh but position-less.

        The login choreography sends the full-roster TankInfo dump
        first (measured 2026-08-04: 113/113 tanks 0x21-first), so
        every tank starts at the (0, 0) construction default with a
        fresh viewport stamp. Without the position gate the whole
        roster walls off the map corner for the TTL.
        """
        phantom = _tank(x=0, y=0)
        assert is_tank_body_present(phantom, _NOW_MS) is False

    def test_authoritative_zero_zero_body_occupies(self) -> None:
        """A tank an authoritative message placed on (0, 0) is a body."""
        placed = _tank(x=0, y=0, last_position_update_ms=_NOW_MS)
        assert is_tank_body_present(placed, _NOW_MS) is True


class TestOccupiedTankKeys:
    """Tests for the world-level occupancy projection."""

    def test_empty_registry_yields_no_keys(self) -> None:
        """No tanks means no occupied tiles."""
        world = make_empty_world_state()
        assert occupied_tank_keys(world, _NOW_MS) == frozenset()

    def test_present_bodies_are_keyed_by_coordinate(self) -> None:
        """Each qualifying tank contributes its own "x,y" key."""
        world = make_empty_world_state()
        world["tanks"]["7"] = _tank(tank_id=7, x=10, y=20)
        world["tanks"]["8"] = _tank(tank_id=8, x=11, y=20)
        assert occupied_tank_keys(world, _NOW_MS) == frozenset({"10,20", "11,20"})

    def test_filtered_tanks_are_excluded_from_the_projection(self) -> None:
        """Self and stale roster entries never reach the key set."""
        world = make_empty_world_state()
        world["tanks"]["1"] = _tank(tank_id=1, x=50, y=50, is_self=True)
        world["tanks"]["3"] = _tank(
            tank_id=3,
            x=52,
            y=50,
            last_viewport_observation_ms=_NOW_MS - VIEWPORT_PRESENCE_TTL_MS - 1,
        )
        world["tanks"]["4"] = _tank(tank_id=4, x=53, y=50)
        assert occupied_tank_keys(world, _NOW_MS) == frozenset({"53,50"})

    def test_two_tanks_on_one_tile_collapse_to_one_key(self) -> None:
        """The projection is a set of tiles, not a list of tanks."""
        world = make_empty_world_state()
        world["tanks"]["7"] = _tank(tank_id=7, x=10, y=20)
        world["tanks"]["8"] = _tank(tank_id=8, x=10, y=20)
        assert occupied_tank_keys(world, _NOW_MS) == frozenset({"10,20"})
