"""Tests for per-tile scan coverage primitives and the world-state mutator."""

from __future__ import annotations

from tankpit_bot.physics.capacity import free_radar_radius
from tankpit_bot.state.scan_coverage import (
    FORAGE_COVERAGE_TTL_MS,
    free_radar_revealed_tiles,
    is_tile_covered,
    is_viewport_fully_covered,
    record_scanned_tiles,
    select_best_free_radar_position,
    tile_key,
    viewport_tiles,
)
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_self_state,
    make_viewport_state,
)


class TestTileKey:
    """Tests for the dict-key helper."""

    def test_tile_key_uses_comma_separated_coordinates(self) -> None:
        """A tile key is exactly ``"x,y"`` so it round-trips by sight."""
        assert tile_key(0, 0) == "0,0"
        assert tile_key(132, 180) == "132,180"
        assert tile_key(-3, 7) == "-3,7"


class TestIsTileCovered:
    """Tests for the per-tile coverage / TTL predicate."""

    def test_uncovered_tile_reports_false(self) -> None:
        """A tile absent from the map is never covered."""
        assert is_tile_covered({}, 100, 100, 100000) is False

    def test_recently_scanned_tile_is_covered(self) -> None:
        """A live scan mark makes the tile covered."""
        coverage = {"100,100": 100000}

        assert is_tile_covered(coverage, 100, 100, 100000) is True
        assert is_tile_covered(coverage, 100, 100, 100000 + FORAGE_COVERAGE_TTL_MS) is True

    def test_expired_tile_is_not_covered(self) -> None:
        """A mark older than the coverage TTL no longer counts."""
        coverage = {"100,100": 100000}

        assert is_tile_covered(coverage, 100, 100, 100000 + FORAGE_COVERAGE_TTL_MS + 1) is False


def _make_world(scanned_tiles: dict[str, int] | None = None) -> WorldStateDict:
    """Build a minimal WorldStateDict for mutator tests."""
    self_state: SelfStateDict = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=1,
        rank=2,
        fuel=500,
        leaderboard_position=5,
    )
    return WorldStateDict(
        self_state=self_state,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=make_viewport_state(left=92, top=92, width=16, height=16),
        scanned_tiles=dict(scanned_tiles) if scanned_tiles else {},
        timestamp_ms=0,
    )


class TestRecordScannedTiles:
    """Tests for the world-state mutator that records radar coverage."""

    def test_records_each_revealed_tile(self) -> None:
        """Every tile in the reveal set lands in scanned_tiles."""
        world = _make_world()

        result = record_scanned_tiles(world, [(10, 10), (10, 11), (11, 11)], 100000)

        assert result["scanned_tiles"] == {
            "10,10": 100000,
            "10,11": 100000,
            "11,11": 100000,
        }
        assert result["timestamp_ms"] == 100000

    def test_prunes_expired_tiles_in_the_same_pass(self) -> None:
        """Marks older than harvest memory drop; younger ones survive.

        Retention is :data:`HARVEST_MEMORY_TTL_MS`, NOT the forage TTL:
        the barren-memory veto must still know "we swept this ground 4
        minutes ago" after the 180 s forage coverage aged out
        ([[flag-triage-20260729]] F2).
        """
        from tankpit_bot.state.scan_coverage import HARVEST_MEMORY_TTL_MS

        stale = {"1,1": 100000 - HARVEST_MEMORY_TTL_MS - 1}
        aged_but_remembered = {"3,3": 100000 - FORAGE_COVERAGE_TTL_MS - 1}
        fresh = {"2,2": 100000 - 1000}
        world = _make_world({**stale, **aged_but_remembered, **fresh})

        result = record_scanned_tiles(world, [(10, 10)], 100000)

        assert result["scanned_tiles"] == {
            "3,3": 100000 - FORAGE_COVERAGE_TTL_MS - 1,
            "2,2": 100000 - 1000,
            "10,10": 100000,
        }

    def test_does_not_mutate_input(self) -> None:
        """The original world state is unchanged after recording."""
        world = _make_world({"5,5": 99000})

        record_scanned_tiles(world, [(10, 10)], 100000)

        assert world["scanned_tiles"] == {"5,5": 99000}

    def test_recording_zero_tiles_still_prunes_expired(self) -> None:
        """An empty reveal set is a valid pure-prune operation."""
        from tankpit_bot.state.scan_coverage import HARVEST_MEMORY_TTL_MS

        world = _make_world({"1,1": 100000 - HARVEST_MEMORY_TTL_MS - 1})

        result = record_scanned_tiles(world, [], 100000)

        assert result["scanned_tiles"] == {}


class TestViewportTiles:
    """Tests for the full-viewport tile enumeration helper."""

    def test_enumerates_every_tile_in_inclusive_bounds(self) -> None:
        """A 3x2 viewport returns six tiles in row-major order."""
        tiles = viewport_tiles(10, 20, 12, 21)

        assert tiles == [(10, 20), (11, 20), (12, 20), (10, 21), (11, 21), (12, 21)]

    def test_single_tile_viewport_returns_one_tile(self) -> None:
        """A degenerate 1x1 viewport still produces exactly one tile."""
        assert viewport_tiles(5, 5, 5, 5) == [(5, 5)]


class TestFreeRadarRevealedTiles:
    """Tests for the rank-scaled free-radar reveal footprint."""

    def test_corporal_interior_reveals_full_5x5_block(self) -> None:
        """Rank 2 (corporal) built-in radar reveals a 5x5 = 25 tile block."""
        revealed = free_radar_revealed_tiles(100, 100, 92, 92, 107, 107, 2)

        assert len(revealed) == 25
        assert (100, 100) in revealed
        assert (98, 98) in revealed
        assert (102, 102) in revealed

    def test_sergeant_interior_reveals_full_7x7_block(self) -> None:
        """Rank 3 (sergeant) built-in radar reveals a 7x7 = 49 tile block."""
        revealed = free_radar_revealed_tiles(100, 100, 92, 92, 107, 107, 3)

        assert len(revealed) == 49
        assert (97, 97) in revealed
        assert (103, 103) in revealed
        # A rank-2 radar would only reach 98..102; sergeant's is one wider.
        assert (100, 97) in revealed
        assert (100, 103) in revealed

    def test_major_interior_reveals_full_9x9_block(self) -> None:
        """Rank 6 (major) built-in radar reveals a 9x9 = 81 tile block."""
        revealed = free_radar_revealed_tiles(100, 100, 92, 92, 107, 107, 6)

        assert len(revealed) == 81
        assert (96, 96) in revealed
        assert (104, 104) in revealed

    def test_corporal_corner_tank_reveals_clipped_block(self) -> None:
        """Rank 2 at the top-left corner is clipped: 3x3 = 9 tiles."""
        revealed = free_radar_revealed_tiles(92, 92, 92, 92, 107, 107, 2)

        for x, y in revealed:
            assert 92 <= x <= 107
            assert 92 <= y <= 107
        assert (92, 92) in revealed
        radius = free_radar_radius(2)
        assert len(revealed) == (radius + 1) ** 2

    def test_tank_off_viewport_returns_empty(self) -> None:
        """A tank whose radar box misses the viewport reveals nothing."""
        assert free_radar_revealed_tiles(50, 50, 92, 92, 107, 107, 2) == []


class TestIsViewportFullyCovered:
    """Tests for the viewport-coverage gate."""

    def test_returns_false_when_any_tile_is_uncovered(self) -> None:
        """A single hole in the viewport defeats the gate."""
        coverage = {tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)}
        del coverage["100,100"]

        assert is_viewport_fully_covered(coverage, 92, 92, 107, 107, 100000) is False

    def test_returns_true_when_every_tile_is_covered(self) -> None:
        """Every viewport tile within TTL trips the gate."""
        coverage = {tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)}

        assert is_viewport_fully_covered(coverage, 92, 92, 107, 107, 100000) is True

    def test_expired_tile_disqualifies_full_coverage(self) -> None:
        """A stale mark counts as uncovered for the gate."""
        coverage = {tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)}
        coverage["100,100"] = 100000 - FORAGE_COVERAGE_TTL_MS - 1

        assert is_viewport_fully_covered(coverage, 92, 92, 107, 107, 100000) is False


class TestSelectBestFreeRadarPosition:
    """Tests for max-coverage destination selection inside the viewport."""

    def test_returns_none_when_viewport_is_fully_covered(self) -> None:
        """Fully covered viewport yields no walk target."""
        coverage = {tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)}

        result = select_best_free_radar_position(coverage, 100, 100, 92, 92, 107, 107, 100000, 2)

        assert result is None

    def test_picks_position_whose_footprint_reveals_the_most_uncovered_tiles(self) -> None:
        """Selection maximises next-radar coverage, not nearest-unscanned distance.

        Coverage layout: every tile in the viewport scanned EXCEPT a
        4x4 cluster well clear of the tank. The optimal destination is
        the centre of that cluster -- its 5x5 footprint contains all
        16 uncovered tiles -- even though one corner of the cluster is
        slightly closer to the tank.
        """
        # Cover every viewport tile, then carve out a 4x4 unscanned block at (100..103, 100..103).
        coverage: dict[str, int] = {
            tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)
        }
        for y in range(100, 104):
            for x in range(100, 104):
                del coverage[tile_key(x, y)]

        # Tank sits at the top-left corner of the viewport, far from the unscanned cluster.
        result = select_best_free_radar_position(coverage, 92, 92, 92, 92, 107, 107, 100000, 2)

        # The centre of the 4x4 cluster is (101, 101) or (102, 102). A 5x5 footprint
        # at either covers all 16 uncovered tiles; the picker should land inside the
        # cluster (not on the corner nearest the tank).
        if result is None:
            raise AssertionError("expected a destination inside the unscanned cluster")
        rx, ry = result
        assert 100 <= rx <= 103
        assert 100 <= ry <= 103

    def test_higher_rank_footprint_widens_the_reach(self) -> None:
        """A rank 6 (9x9) tank reaches an uncovered tile a rank 2 (5x5) can't.

        Single uncovered tile at (99, 99). Tank at (95, 95). Rank 2's
        5x5 from any tank position can reach at most 2 tiles away,
        yielding a best destination of (97, 97). Rank 6's 9x9 reaches
        4 tiles away, so it can stay at the tank's tile candidates
        closer to itself -- verify that the higher-rank result comes
        strictly nearer to (95, 95) than the lower-rank one.
        """
        coverage: dict[str, int] = {
            tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)
        }
        del coverage[tile_key(99, 99)]

        low_result = select_best_free_radar_position(coverage, 95, 95, 92, 92, 107, 107, 100000, 2)
        high_result = select_best_free_radar_position(coverage, 95, 95, 92, 92, 107, 107, 100000, 6)

        if low_result is None or high_result is None:
            raise AssertionError("expected destinations at both ranks")
        low_dist = abs(low_result[0] - 95) + abs(low_result[1] - 95)
        high_dist = abs(high_result[0] - 95) + abs(high_result[1] - 95)
        assert high_dist < low_dist

    def test_skips_expired_marks(self) -> None:
        """A tile whose mark expired is again counted as uncovered."""
        coverage = {tile_key(x, y): 100000 for y in range(92, 108) for x in range(92, 108)}
        # Expire one tile so it becomes selectable.
        coverage["105,100"] = 100000 - FORAGE_COVERAGE_TTL_MS - 1

        result = select_best_free_radar_position(
            coverage,
            100,
            100,
            92,
            92,
            107,
            107,
            100000,
            2,
        )

        # The one uncovered tile is (105, 100); the optimal destination is any
        # position whose 5x5 covers it. Closest-by-distance among equal-score
        # candidates wins; (103, 100) is the closest position with (105, 100)
        # inside its 5x5 footprint.
        assert result == (103, 100)


class TestIsViewportScannedWithin:
    """The barren-memory predicate ([[flag-triage-20260729]] F2)."""

    def _full_coverage(self, ts: int) -> dict[str, int]:
        """Every tile of the (100,100)-(103,103) test viewport marked at ts."""
        return {f"{x},{y}": ts for y in range(100, 104) for x in range(100, 104)}

    def test_full_recent_coverage_is_scanned(self) -> None:
        """Every tile marked inside the window -> True."""
        from tankpit_bot.state.scan_coverage import (
            HARVEST_MEMORY_TTL_MS,
            is_viewport_scanned_within,
        )

        coverage = self._full_coverage(100000)
        assert (
            is_viewport_scanned_within(
                coverage,
                100,
                100,
                103,
                103,
                100000 + HARVEST_MEMORY_TTL_MS,
                ttl_ms=HARVEST_MEMORY_TTL_MS,
            )
            is True
        )

    def test_one_missing_tile_is_not_scanned(self) -> None:
        """A single unmarked tile means the sweep was incomplete."""
        from tankpit_bot.state.scan_coverage import (
            HARVEST_MEMORY_TTL_MS,
            is_viewport_scanned_within,
        )

        coverage = self._full_coverage(100000)
        del coverage["102,101"]
        assert (
            is_viewport_scanned_within(
                coverage, 100, 100, 103, 103, 100000, ttl_ms=HARVEST_MEMORY_TTL_MS
            )
            is False
        )

    def test_expired_coverage_is_not_scanned(self) -> None:
        """Marks older than the window no longer count as knowledge."""
        from tankpit_bot.state.scan_coverage import (
            HARVEST_MEMORY_TTL_MS,
            is_viewport_scanned_within,
        )

        coverage = self._full_coverage(100000)
        assert (
            is_viewport_scanned_within(
                coverage,
                100,
                100,
                103,
                103,
                100000 + HARVEST_MEMORY_TTL_MS + 1,
                ttl_ms=HARVEST_MEMORY_TTL_MS,
            )
            is False
        )

    def test_edge_viewport_clamps_off_map_tiles(self) -> None:
        """Off-map tiles beyond the 0..255 border cannot dirty the answer."""
        from tankpit_bot.state.scan_coverage import (
            HARVEST_MEMORY_TTL_MS,
            is_viewport_scanned_within,
        )

        coverage = {f"{x},{y}": 100000 for y in range(0, 3) for x in range(0, 3)}
        # Viewport (-2,-2)-(2,2): the on-map part is (0,0)-(2,2), fully marked.
        assert (
            is_viewport_scanned_within(coverage, -2, -2, 2, 2, 100000, ttl_ms=HARVEST_MEMORY_TTL_MS)
            is True
        )

    def test_fully_off_map_viewport_is_not_scanned(self) -> None:
        """Bounds with no on-map tiles carry no knowledge at all."""
        from tankpit_bot.state.scan_coverage import (
            HARVEST_MEMORY_TTL_MS,
            is_viewport_scanned_within,
        )

        assert (
            is_viewport_scanned_within({}, 260, 260, 270, 270, 100000, ttl_ms=HARVEST_MEMORY_TTL_MS)
            is False
        )
