"""Tests for tank-observation merge semantics and creation.

``test_tank_observation.py`` was 871 lines; the codec and the
invariants are now siblings.
"""

from __future__ import annotations

from tankpit_bot.state.tank_mutations import apply_tank_observation
from tankpit_bot.state.types import (
    make_empty_world_state,
    make_tank_observation,
)
from tests.world_state._observation_fixtures import make_world_with_seed


class TestFieldMergeSemantics:
    """Present aspects overwrite; ``None`` aspects preserve existing values."""

    def test_position_present_overwrites(self) -> None:
        """Position field updates ``x``/``y`` on the tank."""
        state, key = make_world_with_seed(tank_id=42, x=10, y=20)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(99, 88),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["x"] == 99
        assert result["tanks"][key]["y"] == 88

    def test_position_none_preserves(self) -> None:
        """Position=None preserves existing ``(x, y)``."""
        state, key = make_world_with_seed(tank_id=42, x=10, y=20)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["x"] == 10
        assert result["tanks"][key]["y"] == 20

    def test_team_rank_damage_direction_present_overwrites(self) -> None:
        """Aspect fields write through when present."""
        state, key = make_world_with_seed(tank_id=42, team=0, rank=1, damage_state=0, direction=8)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            team=3,
            rank=5,
            damage_state=2,
            direction=33,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["team"] == 3
        assert tank["rank"] == 5
        assert tank["damage_state"] == 2
        assert tank["direction"] == 33

    def test_team_rank_damage_direction_none_preserves(self) -> None:
        """Aspect fields preserve when absent."""
        state, key = make_world_with_seed(tank_id=42, team=2, rank=4, damage_state=1, direction=12)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["team"] == 2
        assert tank["rank"] == 4
        assert tank["damage_state"] == 1
        assert tank["direction"] == 12

    def test_name_is_bot_present_overwrites(self) -> None:
        """``name`` and ``is_bot`` write through when present."""
        state, key = make_world_with_seed(tank_id=42, name="old", is_bot=False)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            name="new",
            is_bot=True,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["name"] == "new"
        assert tank["is_bot"] is True

    def test_name_is_bot_none_preserves(self) -> None:
        """``name`` and ``is_bot`` preserve when absent."""
        state, key = make_world_with_seed(tank_id=42, name="keep", is_bot=True)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["name"] == "keep"
        assert tank["is_bot"] is True


class TestTankCreationOnFirstObservation:
    """Observations create tanks that did not exist in the registry."""

    def test_first_wire_position_observation_creates_tank(self) -> None:
        """First sighting via wire with position populates the registry."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=77,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(40, 50),
            team=1,
            rank=2,
            damage_state=0,
            direction=8,
            name="Newbie",
            is_bot=False,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"]["77"]
        assert tank["x"] == 40
        assert tank["y"] == 50
        assert tank["team"] == 1
        assert tank["rank"] == 2
        assert tank["name"] == "Newbie"
        assert tank["timestamp_ms"] == 5000
        assert tank["last_wire_seen_ms"] == 5000
        assert tank["last_position_update_ms"] == 5000

    def test_first_observation_without_position_creates_tank_at_origin(self) -> None:
        """First sighting without position seeds ``(0, 0)``."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=77,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"]["77"]
        assert tank["x"] == 0
        assert tank["y"] == 0
        assert tank["damage_state"] == 2

    def test_self_tank_observation_sets_is_self(self) -> None:
        """Observations for the bot's own tank id set ``is_self`` to True."""
        state = make_empty_world_state()
        # Seed self_state so the observation can identify itself.
        from tankpit_bot.state.types.self_state import make_self_state

        state["self_state"] = make_self_state(
            tank_id=42,
            x=0,
            y=0,
            team=0,
            rank=0,
            fuel=0,
            leaderboard_position=0,
        )
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(40, 50),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["42"]["is_self"] is True


class TestStorageSourceIsRecorded:
    """``storage_source`` becomes the tank's ``source`` field."""

    def test_viewport_source_recorded(self) -> None:
        """Viewport storage source is stored on the tank."""
        state, key = make_world_with_seed(tank_id=42)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["source"] == "viewport"

    def test_radar_source_recorded(self) -> None:
        """Radar storage source is stored on the tank."""
        state, key = make_world_with_seed(tank_id=42)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="radar",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["source"] == "radar"

    def test_world_state_source_recorded(self) -> None:
        """World-state storage source is stored on the tank."""
        state, key = make_world_with_seed(tank_id=42)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["source"] == "world_state"


class TestOuterTimestampAdvances:
    """``state["timestamp_ms"]`` advances alongside the per-tank timestamp."""

    def test_outer_timestamp_advances(self) -> None:
        """The outer world-state timestamp tracks the observation."""
        state, _ = make_world_with_seed(tank_id=42, timestamp_ms=1000)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["timestamp_ms"] == 5000


class TestMapPositionDeferSentinel:
    """The defer never protects the (0,0) construction default."""

    def test_map_fix_lands_on_a_roster_sentinel_tank(self) -> None:
        """A fresh-stamped (0,0) roster entry still takes the snapshot fix.

        The login choreography seeds tanks at (0,0) with advancing
        freshness; protecting that default from the map's real
        coordinates would freeze phantom corner tanks — the exact
        class ``has_known_position`` was built against.
        """
        state, key = make_world_with_seed(tank_id=42, x=0, y=0, last_position_update_ms=4500)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert (result["tanks"][key]["x"], result["tanks"][key]["y"]) == (50, 60)
        assert result["tanks"][key]["last_position_update_ms"] == 5000
