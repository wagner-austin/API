"""Tests for the fresh-viewport hop planner."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.resource_search import (
    is_recently_attempted,
    make_resource_search_hop,
    record_attempt_mark,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _fully_cover_viewport_origins(
    origins: list[tuple[int, int]],
    *,
    timestamp_ms: int = 100000,
    width: int = 16,
    height: int = 16,
) -> dict[str, int]:
    """Return a tile-coverage dict that fully covers every supplied 16x16 viewport."""
    covered: dict[str, int] = {}
    for left, top in origins:
        for y in range(top, top + height):
            for x in range(left, left + width):
                covered[f"{x},{y}"] = timestamp_ms
    return covered


def _ctx(
    *,
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    terrain: InMemoryTerrainMap | None = None,
    ai_state: AIStateDict | None = None,
    scanned_viewport_origins: list[tuple[int, int]] | None = None,
) -> DecideCtx:
    """Build a DecideCtx with a clean default world and optional overrides."""
    world, self_state = make_world(self_x=self_x, self_y=self_y, fuel=fuel)
    if scanned_viewport_origins:
        world["scanned_tiles"].update(_fully_cover_viewport_origins(scanned_viewport_origins))
    return DecideCtx(
        world,
        self_state,
        ai_state if ai_state is not None else make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )


class TestMakeResourceSearchHop:
    """Behavior tests for the single-method fresh-viewport hop planner."""

    def test_picks_east_cardinal_first(self) -> None:
        """From open ground, east (the first cardinal) is taken."""
        decision = make_resource_search_hop(
            _ctx(), mode="COLLECT", score=900, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("expected a cardinal hop decision from open ground")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 116
        assert command["target_y"] == 100

    def test_skips_scanned_destination_takes_next_cardinal(self) -> None:
        """When east lands in a scanned viewport, west (second cardinal) wins."""
        decision = make_resource_search_hop(
            _ctx(scanned_viewport_origins=[(108, 92)]),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected west cardinal when east landing is scanned")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 84
        assert command["target_y"] == 100

    def test_skips_impassable_cardinal(self) -> None:
        """Cardinal whose landing tile is water is skipped."""
        terrain = InMemoryTerrainMap(terrain_data={(116, 100): "W"})
        decision = make_resource_search_hop(
            _ctx(terrain=terrain),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a non-east cardinal when east is impassable")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] != 116

    def test_skips_clamped_edge_cardinal(self) -> None:
        """Near the map edge, the clamped direction is skipped, west wins.

        At (250,250) east and south clamp to displacement < 16; only
        west and north qualify. West (the first viable cardinal in the
        iteration order) wins.
        """
        decision = make_resource_search_hop(
            _ctx(self_x=250, self_y=250),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected west cardinal when east and south clamp")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 234
        assert command["target_y"] == 250

    def test_skips_unaffordable_destination(self) -> None:
        """Fuel below cardinal cost (96) returns None.

        At fuel=80 every cardinal is unaffordable (cardinal cost = 96)
        AND every diagonal is unaffordable (diagonal cost = 135). No
        hop is taken.
        """
        decision = make_resource_search_hop(
            _ctx(fuel=80),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        assert decision is None

    def test_falls_through_to_diagonal_when_all_cardinals_blocked(self) -> None:
        """All four cardinals scanned -> a diagonal is taken instead."""
        cardinal_viewport_origins = [(108, 92), (76, 92), (92, 108), (92, 76)]
        decision = make_resource_search_hop(
            _ctx(scanned_viewport_origins=cardinal_viewport_origins),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a diagonal hop when cardinals are blocked")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert abs(command["target_x"] - 100) == 16
        assert abs(command["target_y"] - 100) == 16

    def test_cardinal_preferred_over_diagonal_when_both_fresh(self) -> None:
        """A fresh cardinal beats a fresh diagonal; cheaper hop wins."""
        decision = make_resource_search_hop(
            _ctx(),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a hop from open ground")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        target_x = command["target_x"]
        target_y = command["target_y"]
        # Cardinal hops change exactly one axis by 16; diagonals change both.
        assert (abs(target_x - 100) == 16) != (abs(target_y - 100) == 16)

    def test_returns_none_when_both_rings_blocked(self) -> None:
        """Every candidate in both rings scanned -> None, caller raises."""
        directions = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        blocked = [
            (100 + dx * step - 8, 100 + dy * step - 8) for step in (16, 32) for dx, dy in directions
        ]
        decision = make_resource_search_hop(
            _ctx(scanned_viewport_origins=blocked),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        assert decision is None

    def test_reaches_second_ring_when_first_ring_scanned(self) -> None:
        """All eight one-width candidates scanned -> a two-width hop is taken."""
        directions = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        ring_one = [(100 + dx * 16 - 8, 100 + dy * 16 - 8) for dx, dy in directions]
        decision = make_resource_search_hop(
            _ctx(fuel=2000, scanned_viewport_origins=ring_one),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a second-ring hop when ring one is scanned")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 132
        assert command["target_y"] == 100

    def test_prefers_most_walkable_viewport(self) -> None:
        """A mostly-water nearer viewport loses to an all-ground one.

        The east landing viewport is water except the landing tile
        itself (so east still qualifies via the landing-passability
        gate), while every other viewport is clean ground. The picker
        must skip past the qualifying-but-dirty east candidate and take
        the clean west one -- the recorded human policy of restocking
        in mostly-"." viewports.
        """
        east_viewport_water = {
            (x, y): InMemoryTerrainMap.WATER
            for y in range(92, 108)
            for x in range(108, 124)
            if (x, y) != (116, 100)
        }
        terrain = InMemoryTerrainMap(terrain_data=east_viewport_water)
        decision = make_resource_search_hop(
            _ctx(terrain=terrain),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a hop away from the water-heavy viewport")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 84
        assert command["target_y"] == 100

    def test_offmap_clipping_penalizes_edge_viewport(self) -> None:
        """A viewport clipped by the field border scores below a full one.

        At (100, 238) with east and west scanned, the south candidate's
        viewport hangs 6 rows past the border (score 0.625 on all-ground
        terrain) while north is fully in-map (score 1.0). North must win
        even though south is evaluated first.
        """
        decision = make_resource_search_hop(
            _ctx(
                self_y=238,
                terrain=InMemoryTerrainMap(),
                scanned_viewport_origins=[(108, 230), (76, 230)],
            ),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the fully in-map north hop")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 100
        assert command["target_y"] == 222

    def test_returns_none_when_diagonal_unaffordable_and_cardinals_blocked(self) -> None:
        """Cardinals scanned + fuel < diagonal cost (135) -> None."""
        cardinal_viewport_origins = [(108, 92), (76, 92), (92, 108), (92, 76)]
        decision = make_resource_search_hop(
            _ctx(fuel=120, scanned_viewport_origins=cardinal_viewport_origins),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        assert decision is None

    def test_clears_resource_target_on_success(self) -> None:
        """A successful hop clears any previously locked resource target."""
        base = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "resource_target_kind": "fuel",
                "resource_target_x": 200,
                "resource_target_y": 150,
            }
        )
        decision = make_resource_search_hop(
            _ctx(),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
            ai_state=base,
        )

        if decision is None:
            raise AssertionError("expected a successful hop to clear the lock")
        assert decision["updated_ai_state"]["resource_target_kind"] == ""
        assert decision["updated_ai_state"]["resource_target_x"] == 0
        assert decision["updated_ai_state"]["resource_target_y"] == 0

    def test_no_terrain_treats_every_tile_as_passable(self) -> None:
        """Without a terrain map, the passability gate is skipped."""
        decision = make_resource_search_hop(
            _ctx(terrain=None),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a hop when terrain is absent")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"

    def test_uses_ctx_base_state_when_ai_state_not_provided(self) -> None:
        """Omitting ``ai_state`` uses the context's base state for clearing."""
        decision = make_resource_search_hop(
            _ctx(), mode="COLLECT", score=900, reason="search_collect_local"
        )

        if decision is None:
            raise AssertionError("expected a hop using ctx.base when ai_state omitted")
        assert decision["updated_ai_state"]["resource_target_kind"] == ""


class TestAttemptMarks:
    """Tests for the failed-pickup attempt-tracking helpers."""

    def test_is_recently_attempted_returns_true_within_ttl(self) -> None:
        """A coordinate marked inside the TTL is reported as attempted."""
        marks = {"100,100": 95000}

        assert is_recently_attempted(marks, 100, 100, 100000, ttl_ms=10000) is True

    def test_is_recently_attempted_returns_false_outside_ttl(self) -> None:
        """A coordinate marked outside the TTL is reported as not attempted."""
        marks = {"100,100": 80000}

        assert is_recently_attempted(marks, 100, 100, 100000, ttl_ms=10000) is False

    def test_is_recently_attempted_returns_false_for_unknown_coord(self) -> None:
        """An unmarked coordinate is reported as not attempted."""
        assert is_recently_attempted({}, 100, 100, 100000, ttl_ms=10000) is False

    def test_record_attempt_mark_adds_new_coordinate(self) -> None:
        """A fresh attempt is recorded with the dispatch timestamp."""
        result = record_attempt_mark({}, 100, 100, 100000, ttl_ms=10000)

        assert result == {"100,100": 100000}

    def test_record_attempt_mark_prunes_expired_entries(self) -> None:
        """Expired marks are dropped while the new mark is added."""
        marks = {"50,50": 80000, "60,60": 95000}

        result = record_attempt_mark(marks, 100, 100, 100000, ttl_ms=10000)

        assert "50,50" not in result
        assert result["60,60"] == 95000
        assert result["100,100"] == 100000
