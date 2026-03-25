"""Tests for AI threat analysis."""

from __future__ import annotations

from tankpit_bot.bot.ai.threats import (
    analyze_threats,
    find_closest_threat,
    manhattan_distance,
    threats_in_range,
)
from tankpit_bot.state.types import (
    SelfStateDict,
    TankStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_self_state,
    make_tank_state,
)


def _tank(
    key: str,
    x: int = 0,
    y: int = 0,
    team: int = 1,
    damage_state: int = 0,
    name: str = "",
    is_bot: bool = True,
    is_self: bool = False,
) -> TankStateDict:
    """Create a TankStateDict with defaults for testing.

    Args:
        key: Tank ID as string.
        x: X coordinate.
        y: Y coordinate.
        team: Team ID.
        damage_state: Damage state (0-3).
        name: Player name (defaults to "tank-{key}").
        is_bot: Whether this is a bot.
        is_self: Whether this is the player's tank.

    Returns:
        TankStateDict with the provided values.
    """
    return make_tank_state(
        tank_id=int(key),
        x=x,
        y=y,
        team=team,
        rank=0,
        damage_state=damage_state,
        name=name or f"tank-{key}",
        is_bot=is_bot,
        is_self=is_self,
    )


def _world(tanks: dict[str, TankStateDict]) -> WorldStateDict:
    """Build a WorldStateDict with only the given tanks.

    Args:
        tanks: Dict mapping tank_id string keys to TankStateDicts.

    Returns:
        WorldStateDict with the provided tanks.
    """
    return WorldStateDict(
        self_state=None,
        tanks=tanks,
        containers={},
        mines={},
        terrain={},
        viewport=ViewportStateDict(left=0, top=0, width=18, height=18),
        timestamp_ms=0,
    )


def _self_at(x: int = 100, y: int = 100) -> SelfStateDict:
    """Create self state at given position on team 0."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=800,
        leaderboard_position=1,
    )


# =============================================================================
# manhattan_distance
# =============================================================================


class TestManhattanDistance:
    """Tests for manhattan_distance."""

    def test_same_point(self) -> None:
        """Distance to self is zero."""
        assert manhattan_distance(10, 20, 10, 20) == 0

    def test_horizontal(self) -> None:
        """Horizontal distance only."""
        assert manhattan_distance(0, 0, 5, 0) == 5

    def test_vertical(self) -> None:
        """Vertical distance only."""
        assert manhattan_distance(0, 0, 0, 7) == 7

    def test_diagonal(self) -> None:
        """Manhattan distance is sum of x and y differences."""
        assert manhattan_distance(10, 10, 15, 20) == 15

    def test_negative_direction(self) -> None:
        """Distance is always positive regardless of direction."""
        assert manhattan_distance(20, 20, 10, 5) == 25


# =============================================================================
# analyze_threats
# =============================================================================


class TestAnalyzeThreats:
    """Tests for analyze_threats."""

    def test_empty_world(self) -> None:
        """No tanks produces empty threat list."""
        world = _world({})
        threats = analyze_threats(world, _self_at())
        assert threats == []

    def test_filters_same_team(self) -> None:
        """Tanks on same team are not threats."""
        world = _world({"10": _tank("10", x=110, y=100, team=0)})
        threats = analyze_threats(world, _self_at())
        assert threats == []

    def test_filters_self(self) -> None:
        """Player's own tank is not a threat."""
        world = _world({"1": _tank("1", x=100, y=100, team=1, is_self=True)})
        threats = analyze_threats(world, _self_at())
        assert threats == []

    def test_identifies_enemies(self) -> None:
        """Tanks on different teams are threats."""
        world = _world(
            {
                "10": _tank("10", x=110, y=100, team=1),
                "20": _tank("20", x=120, y=100, team=2),
            }
        )
        threats = analyze_threats(world, _self_at())
        assert len(threats) == 2

    def test_computes_distance(self) -> None:
        """Threats have correct Manhattan distance."""
        world = _world({"10": _tank("10", x=115, y=107, team=1)})
        threats = analyze_threats(world, _self_at())
        assert len(threats) == 1
        assert threats[0]["distance"] == 22  # |115-100| + |107-100|

    def test_sorted_by_distance_ascending(self) -> None:
        """Threats are sorted closest first."""
        world = _world(
            {
                "10": _tank("10", x=150, y=100, team=1),
                "20": _tank("20", x=105, y=100, team=2),
                "30": _tank("30", x=130, y=100, team=3),
            }
        )
        threats = analyze_threats(world, _self_at())
        assert len(threats) == 3
        assert threats[0]["tank_id"] == 20  # distance 5
        assert threats[1]["tank_id"] == 30  # distance 30
        assert threats[2]["tank_id"] == 10  # distance 50

    def test_equal_distance_damaged_first(self) -> None:
        """At equal distance, more damaged enemies sort first."""
        world = _world(
            {
                "10": _tank("10", x=110, y=100, team=1, damage_state=0),
                "20": _tank("20", x=100, y=110, team=2, damage_state=2),
            }
        )
        threats = analyze_threats(world, _self_at())
        assert len(threats) == 2
        # Both at distance 10, but tank 20 has damage_state=2 (sorted first)
        assert threats[0]["tank_id"] == 20
        assert threats[1]["tank_id"] == 10

    def test_threat_fields_populated(self) -> None:
        """All EnemyThreatDict fields are populated from tank state."""
        world = _world(
            {
                "42": _tank(
                    "42",
                    x=120,
                    y=130,
                    team=2,
                    damage_state=1,
                    name="enemy-42",
                    is_bot=False,
                ),
            }
        )
        threats = analyze_threats(world, _self_at())
        assert len(threats) == 1
        t = threats[0]
        assert t["tank_id"] == 42
        assert t["x"] == 120
        assert t["y"] == 130
        assert t["distance"] == 50
        assert t["damage_state"] == 1
        assert t["team"] == 2
        assert t["name"] == "enemy-42"
        assert t["is_bot"] is False

    def test_mixed_allies_and_enemies(self) -> None:
        """Only enemy tanks appear in results."""
        world = _world(
            {
                "10": _tank("10", x=110, y=100, team=0),  # ally
                "20": _tank("20", x=120, y=100, team=1),  # enemy
                "30": _tank("30", x=130, y=100, team=0),  # ally
                "40": _tank("40", x=140, y=100, team=2),  # enemy
            }
        )
        threats = analyze_threats(world, _self_at())
        assert len(threats) == 2
        ids = {t["tank_id"] for t in threats}
        assert ids == {20, 40}


# =============================================================================
# find_closest_threat
# =============================================================================


class TestFindClosestThreat:
    """Tests for find_closest_threat."""

    def test_empty_list(self) -> None:
        """Returns None for empty threat list."""
        assert find_closest_threat([]) is None

    def test_returns_first(self) -> None:
        """Returns first element of sorted list (closest enemy)."""
        world = _world(
            {
                "10": _tank("10", x=150, y=100, team=1),
                "20": _tank("20", x=105, y=100, team=2),
            }
        )
        threats = analyze_threats(world, _self_at())
        closest = find_closest_threat(threats)
        assert closest == threats[0]
        assert closest["tank_id"] == 20


# =============================================================================
# threats_in_range
# =============================================================================


class TestThreatsInRange:
    """Tests for threats_in_range."""

    def test_empty_list(self) -> None:
        """Returns empty list for empty threats."""
        assert threats_in_range([], 20) == []

    def test_filters_by_range(self) -> None:
        """Only threats within combat_range are returned."""
        world = _world(
            {
                "10": _tank("10", x=105, y=100, team=1),  # distance 5
                "20": _tank("20", x=115, y=100, team=2),  # distance 15
                "30": _tank("30", x=150, y=100, team=3),  # distance 50
            }
        )
        threats = analyze_threats(world, _self_at())
        in_range = threats_in_range(threats, 20)
        assert len(in_range) == 2
        assert in_range[0]["tank_id"] == 10
        assert in_range[1]["tank_id"] == 20

    def test_exact_boundary(self) -> None:
        """Threat at exactly combat_range is included."""
        world = _world(
            {
                "10": _tank("10", x=120, y=100, team=1),  # distance exactly 20
            }
        )
        threats = analyze_threats(world, _self_at())
        in_range = threats_in_range(threats, 20)
        assert len(in_range) == 1

    def test_none_in_range(self) -> None:
        """Returns empty when no threats are close enough."""
        world = _world(
            {
                "10": _tank("10", x=200, y=200, team=1),  # distance 200
            }
        )
        threats = analyze_threats(world, _self_at())
        in_range = threats_in_range(threats, 20)
        assert in_range == []
