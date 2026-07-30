"""Tests for AI threat analysis."""

from __future__ import annotations

from tankpit_bot.bot.ai.threats import (
    analyze_threats,
    find_acquisition_target,
    find_closest_threat,
    find_locked_target_pursuit,
    manhattan_distance,
    threats_in_range,
)
from tankpit_bot.state.types import (
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_self_state,
    make_tank_state,
    make_viewport_state,
)
from tankpit_bot.state.types.constants import TankLiveness
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _tank(
    key: str,
    x: int = 0,
    y: int = 0,
    team: int = 1,
    damage_state: int = 0,
    direction: int = 0,
    name: str = "",
    is_bot: bool = True,
    is_self: bool = False,
    liveness: TankLiveness = "alive",
) -> TankStateDict:
    """Create a TankStateDict with defaults for testing.

    Args:
        key: Tank ID as string.
        x: X coordinate.
        y: Y coordinate.
        team: Team ID.
        damage_state: Damage state (0-3).
        direction: Sprite direction (0-31 alive, 32-33 dead).
        name: Player name (defaults to "tank-{key}").
        is_bot: Whether this is a bot.
        is_self: Whether this is the player's tank.
        liveness: Lifecycle state. Defaults to ``"alive"``.

    Returns:
        TankStateDict with the provided values.
    """
    # rank=1 (private): the default ``tank-{key}`` names classify as
    # HUMAN under the 2026-07-28 rank-window rule, and rank-0 humans
    # are protected from targeting -- these fixtures test the
    # distance/freshness/liveness gates, so they sit inside the window.
    return make_tank_state(
        tank_id=int(key),
        x=x,
        y=y,
        team=team,
        rank=1,
        damage_state=damage_state,
        direction=direction,
        name=name or f"tank-{key}",
        is_bot=is_bot,
        is_self=is_self,
        liveness=liveness,
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
        viewport=make_viewport_state(left=0, top=0, width=18, height=18),
        scanned_tiles={},
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
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert threats == []

    def test_filters_same_team(self) -> None:
        """Tanks on same team are not threats."""
        world = _world({"10": _tank("10", x=110, y=100, team=0)})
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert threats == []

    def test_filters_self(self) -> None:
        """Player's own tank is not a threat."""
        world = _world({"1": _tank("1", x=100, y=100, team=1, is_self=True)})
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert threats == []

    def test_identifies_enemies(self) -> None:
        """Tanks on different teams are threats."""
        world = _world(
            {
                "10": _tank("10", x=110, y=100, team=1),
                "20": _tank("20", x=120, y=100, team=2),
            }
        )
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert len(threats) == 2

    def test_computes_distance(self) -> None:
        """Threats have correct Manhattan distance."""
        world = _world({"10": _tank("10", x=115, y=107, team=1)})
        threats = analyze_threats(world, _self_at(), now_ms=0)
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
        threats = analyze_threats(world, _self_at(), now_ms=0)
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
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert len(threats) == 2
        # Both at distance 10; tank 10 is tier 0 (bottom fuel quartile,
        # near death) so it is the finish-off target and sorts first.
        assert threats[0]["tank_id"] == 10
        assert threats[1]["tank_id"] == 20

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
        threats = analyze_threats(world, _self_at(), now_ms=0)
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

    def test_filters_deactivated_tanks(self) -> None:
        """Tanks in the ``deactivated`` corpse window are excluded."""
        world = _world(
            {
                "10": _tank("10", x=110, y=100, team=1, liveness="deactivated"),
                "20": _tank("20", x=120, y=100, team=2),  # alive enemy
            }
        )
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert len(threats) == 1
        assert threats[0]["tank_id"] == 20

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
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert len(threats) == 2
        ids = {t["tank_id"] for t in threats}
        assert ids == {20, 40}

    def test_filters_corpse_via_liveness_when_apply_observation_sets_it(self) -> None:
        """Corpse-direction wire observations transition liveness to
        ``deactivated`` via ``apply_tank_observation``; ``analyze_threats``
        then filters on liveness, not direction.

        The previous behaviour ran a direct ``direction >= 32`` check
        inside ``analyze_threats`` itself. That filter was removed once
        ``apply_tank_observation`` started routing corpse wire arrivals
        to ``liveness == "deactivated"`` (and the 0x41 Deactivation
        dispatcher fires the same transition). The world-state layer
        owns the corpse classification; the threat selector is a single
        liveness check.
        """
        from tankpit_bot.state.mutations import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        world = _world({"10": _tank("10", x=110, y=100, team=1)})
        # An incoming 0x3D with the corpse-direction sprite flips
        # liveness to deactivated. After apply, the tank should be
        # filtered.
        world = apply_tank_observation(
            world,
            make_tank_observation(
                tank_id=10,
                timestamp_ms=0,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(110, 100),
                team=1,
                direction=32,
            ),
        )
        assert world["tanks"]["10"]["liveness"] == "deactivated"
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert len(threats) == 0

    def test_alive_direction_passes(self) -> None:
        """Tanks with direction 0-31 pass through the corpse filter."""
        world = _world(
            {
                "10": _tank("10", x=110, y=100, team=1, direction=0),
                "20": _tank("20", x=120, y=100, team=2, direction=8),
                "30": _tank("30", x=130, y=100, team=3, direction=31),
            }
        )
        threats = analyze_threats(world, _self_at(), now_ms=0)
        assert len(threats) == 3


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
        threats = analyze_threats(world, _self_at(), now_ms=0)
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
        threats = analyze_threats(world, _self_at(), now_ms=0)
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
        threats = analyze_threats(world, _self_at(), now_ms=0)
        in_range = threats_in_range(threats, 20)
        assert len(in_range) == 1

    def test_none_in_range(self) -> None:
        """Returns empty when no threats are close enough."""
        world = _world(
            {
                "10": _tank("10", x=200, y=200, team=1),  # distance 200
            }
        )
        threats = analyze_threats(world, _self_at(), now_ms=0)
        in_range = threats_in_range(threats, 20)
        assert in_range == []


class TestFindAcquisitionTarget:
    """Tests for ``find_acquisition_target`` (loose, map-fresh acquisition)."""

    def test_picks_nearest_map_known_enemy(self) -> None:
        """Acquisition picks the closest enemy with a fresh map observation."""
        tank = _tank("10", x=105, y=100, team=1)
        # Map-fresh: timestamp_ms within map_open_cooldown_ms.
        tank["timestamp_ms"] = 100000
        world = _world({"10": tank})

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        if result is None:
            raise AssertionError("expected an acquisition target")
        assert result["tank_id"] == 10

    def test_filters_stale_timestamp(self) -> None:
        """A tank whose timestamp is older than the cooldown is filtered out.

        Covers the loose-freshness gate at ``threats.py::find_acquisition_target``
        where ``now_ms - tank["timestamp_ms"] >= map_open_cooldown_ms`` skips
        the tank. Without this, the bot would teleport at a position that may
        already be stale by minutes -- which is the gate analyze_threats was
        tightened to prevent for firing (see [[bot-behavior-contract]] §5
        Phantom firing).
        """
        tank = _tank("10", x=105, y=100, team=1)
        tank["timestamp_ms"] = 80000

        world = _world({"10": tank})

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        assert result is None

    def test_acquires_ferry_rider_with_shore_in_shot_range(self) -> None:
        """An enemy on open water is viable when shore lies within shot range.

        Live 2026-07-29 (run bot-20260729-232252): Yuppler rode a ferry
        at (128,102) -- his tile and every neighbor water -- and the old
        strictly-adjacent gate rejected him with no_passable_adjacent on
        every acquisition pass while ground sat three tiles west. The
        gate is stand-off now: the close aims at a passable shore tile
        and duals fire over water ([[weapon-selection]]).
        """
        tank = _tank("10", x=105, y=100, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"10": tank})
        terrain_data: dict[tuple[int, int], str] = {
            (105, 100): "W",
            (104, 100): "W",
            (106, 100): "W",
            (105, 99): "W",
            (105, 101): "W",
        }
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=terrain,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        if result is None:
            raise AssertionError("expected the ferry rider to be acquirable")
        assert result["tank_id"] == 10

    def test_filters_enemy_with_no_standoff_landing(self) -> None:
        """An enemy with no passable tile inside the shot-range diamond is skipped.

        Mid-ocean ferry rider: nothing to land on within
        ``SHOT_RANGE_TILES``, so there is no firing position and the
        acquisition path declines and falls through to the next
        strategy.
        """
        tank = _tank("10", x=105, y=100, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"10": tank})
        terrain = InMemoryTerrainMap.from_passable_set(set())

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=terrain,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        assert result is None

    def test_returns_none_when_no_enemies_visible(self) -> None:
        """Empty world produces no acquisition target."""
        world = _world({})

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        assert result is None

    def test_rejects_unaffordable_enemy(self) -> None:
        """An enemy whose approach teleport breaks the engagement reserve is skipped.

        User contract (2026-07-02): the bot never picks a fight it
        cannot pay for. Live run 2026-07-01 20:45: the nearest map
        enemy cost 505 fuel to reach, leaving too little to finish the
        kill -- acquisition must reject it, not commit to it. Enemy at
        distance 100 costs 600 to reach; 600 + 650 reserve > 800 fuel.
        """
        tank = _tank("10", x=200, y=100, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"10": tank})

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        assert result is None

    def test_picks_affordable_enemy_over_nearer_unaffordable(self) -> None:
        """The nearest-by-Manhattan enemy loses to an affordable one.

        Teleport cost scales with Euclidean distance while candidate
        ordering is Manhattan, so a diagonal enemy can be farther by
        Manhattan yet cheaper to reach: axial (125,100) is Manhattan 25
        but costs 150; diagonal (114,114) is Manhattan 28 but costs
        ~118. With reserve 660 and fuel 800 only the diagonal fits.
        """
        axial = _tank("10", x=125, y=100, team=1)
        axial["timestamp_ms"] = 100000
        diagonal = _tank("20", x=114, y=114, team=2)
        diagonal["timestamp_ms"] = 100000
        world = _world({"10": axial, "20": diagonal})

        result = find_acquisition_target(
            world,
            _self_at(),
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=660,
        )

        if result is None:
            raise AssertionError("expected the affordable diagonal enemy")
        assert result["tank_id"] == 20


class TestFindLockedTargetPursuit:
    """Tests for ``find_locked_target_pursuit`` (locked-target chase)."""

    def test_returns_none_when_no_lock(self) -> None:
        """``locked_target_id == -1`` means no lock to pursue."""
        world = _world({})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=-1,
            killed={},
        )

        assert result is None

    def test_returns_none_when_locked_target_killed(self) -> None:
        """A locked target on the kill cooldown is not pursued.

        Once the bot has observed a kill, the cooldown applies even
        if the registry still lists the tank. Otherwise the pursuit
        path would re-engage corpses for a few seconds.
        """
        tank = _tank("50", x=105, y=100, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={"50": 99500},
        )

        assert result is None

    def test_returns_none_when_target_not_in_registry(self) -> None:
        """A locked id that is no longer in ``world["tanks"]`` cannot be pursued."""
        world = _world({})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        assert result is None

    def test_returns_none_when_target_deactivated(self) -> None:
        """A deactivated (corpse-window) tank is not a pursuit target."""
        tank = _tank("50", x=105, y=100, team=1, liveness="deactivated")
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        assert result is None

    def test_returns_none_when_target_at_origin(self) -> None:
        """A tank with no position-bearing wire (still at (0,0)) is unfireable."""
        tank = _tank("50", x=0, y=0, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        assert result is None

    def test_returns_pursuit_threat_even_when_timestamp_is_stale(self) -> None:
        """Pursuit fires at the cached coords regardless of timestamp staleness.

        The earlier 5 s freshness gate was removed 2026-06-22 -- it
        was tripping on tanks the server stopped broadcasting 0x2E
        for (typically because they teleported far away), ending
        pursuit prematurely. Ammo only decrements on confirmed hit,
        so over-pursuing burns no resources -- the loop is bounded
        by 0x41 Deactivation or the kill cooldown, both authoritative
        death signals.
        """
        tank = _tank("50", x=105, y=100, team=1, name="prey")
        tank["timestamp_ms"] = 80000  # 20 s stale at now_ms=100000 -- old gate would have tripped

        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        if result is None:
            raise AssertionError("expected pursuit threat even with stale timestamp")
        assert result["tank_id"] == 50
        assert result["x"] == 105
        assert result["y"] == 100

    def test_returns_pursuit_threat_for_alive_locked_target(self) -> None:
        """An alive locked target returns a pursuit threat at cached coords."""
        tank = _tank("50", x=105, y=100, team=1, name="prey")
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        if result is None:
            raise AssertionError("expected a pursuit threat for alive target")
        assert result["tank_id"] == 50
        assert result["x"] == 105
        assert result["y"] == 100
