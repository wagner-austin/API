"""Tests for self-state mutations.

Rank, fuel, and the movement-response position update.
"""

from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    set_self_fuel,
    set_self_rank,
    update_self_from_movement_response,
    update_self_rank,
)
from tankpit_bot.state.types import make_self_state
from tankpit_bot.types.constants import (
    TEAM_BLUE,
)
from tests.world_state.helpers import get_self_state


class TestUpdateSelfRank:
    """Tests for update_self_rank (the mid-session promotion mutation)."""

    def _state_with_self(self, rank: int) -> WorldStateDict:
        return update_self_from_movement_response(
            make_empty_world_state(),
            tank_id=1301,
            x=100,
            y=100,
            team=TEAM_BLUE,
            rank=rank,
            leaderboard_position=1,
            timestamp_ms=500,
        )

    def test_applies_a_promotion(self) -> None:
        """A wire rank change lands in self_state (the promoting-kill flip)."""
        state = self._state_with_self(rank=0)
        updated = update_self_rank(state, 1, 1000, "wire_0x2E_tank_status_sync")
        self_state = get_self_state(updated)
        assert self_state["rank"] == 1
        assert self_state["x"] == 100
        assert updated["timestamp_ms"] == 1000

    def test_same_rank_is_a_no_op(self) -> None:
        """An unchanged rank returns the state untouched."""
        state = self._state_with_self(rank=1)
        assert update_self_rank(state, 1, 1000, "wire_0x3D_movement") is state

    def test_no_self_state_is_a_no_op(self) -> None:
        """Rank cannot precede join."""
        state = make_empty_world_state()
        assert update_self_rank(state, 1, 1000, "wire_0x47_movement") is state


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
