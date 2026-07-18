"""Tests for the fuel-dot hop planner."""

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
    map_fuel_dots: tuple[tuple[int, int], ...] = (),
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
        map_fuel_dots,
    )


class TestMakeResourceSearchHop:
    """Behavior tests for the nearest-clean-viewport fuel-dot hop planner."""

    def test_hops_to_nearest_dot(self) -> None:
        """The nearest atlas dot wins when several qualify."""
        decision = make_resource_search_hop(
            _ctx(map_fuel_dots=((150, 100), (130, 100), (100, 160))),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a dot hop from open ground")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 130
        assert command["target_y"] == 100

    def test_skips_own_tile_dot(self) -> None:
        """A dot on the bot's own tile is not a hop destination."""
        decision = make_resource_search_hop(
            _ctx(map_fuel_dots=((100, 100), (130, 100))),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the next dot when the nearest is the own tile")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 130
        assert command["target_y"] == 100

    def test_skips_dot_in_scanned_viewport(self) -> None:
        """A dot whose landing viewport is fully covered is skipped."""
        decision = make_resource_search_hop(
            _ctx(
                map_fuel_dots=((130, 100), (160, 100)),
                scanned_viewport_origins=[(122, 92)],
            ),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the fresh dot when the nearest is scanned")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 160
        assert command["target_y"] == 100

    def test_skips_impassable_dot(self) -> None:
        """A dot whose landing tile is water is skipped."""
        terrain = InMemoryTerrainMap(terrain_data={(130, 100): "W"})
        decision = make_resource_search_hop(
            _ctx(terrain=terrain, map_fuel_dots=((130, 100), (160, 100))),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the passable dot when the nearest is water")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 160
        assert command["target_y"] == 100

    def test_flecked_viewport_no_longer_disqualifies(self) -> None:
        """A near dot with one water tile in view beats a distant clean one.

        User contract 2026-07-18: walkability is a ranking weight, not
        a hard bar ("not a 100% rule ofc"). One water tile costs the
        near dot a sliver of walkable fraction; its far cheaper
        teleport dominates the score.
        """
        terrain = InMemoryTerrainMap(terrain_data={(125, 95): "W"})
        decision = make_resource_search_hop(
            _ctx(terrain=terrain, map_fuel_dots=((130, 100), (100, 160))),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the near flecked-viewport dot")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 130
        assert command["target_y"] == 100

    def test_denser_dot_cluster_outranks_lone_dot(self) -> None:
        """At comparable cost, a viewport holding more dots wins.

        User contract 2026-07-18: "prioritize viewports with more
        dots". Two equidistant candidates: one lone dot, one inside a
        three-dot cluster -- the cluster's landing viewport promises
        three pickups for the same teleport cost.
        """
        decision = make_resource_search_hop(
            _ctx(
                terrain=InMemoryTerrainMap(),
                map_fuel_dots=((100, 40), (100, 160), (103, 162), (98, 165)),
            ),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the cluster dot")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert (command["target_x"], command["target_y"]) in ((100, 160), (103, 162), (98, 165))

    def test_offmap_clipped_viewport_ranks_below_in_map(self) -> None:
        """At equal cost, a border-clipped viewport loses to a full one.

        Off-map tiles count as unwalkable, shrinking the clipped
        viewport's walkable fraction; with equidistant candidates the
        in-map dot's higher fraction wins the ranking.
        """
        decision = make_resource_search_hop(
            _ctx(
                terrain=InMemoryTerrainMap(),
                map_fuel_dots=((100, 130), (100, 252)),
                self_y=191,
            ),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected the in-map dot")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 100
        assert command["target_y"] == 130

    def test_skips_unaffordable_dot(self) -> None:
        """Fuel below every dot's teleport cost returns None."""
        decision = make_resource_search_hop(
            _ctx(fuel=80, map_fuel_dots=((130, 100), (160, 100))),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        assert decision is None

    def test_returns_none_when_every_dot_scanned(self) -> None:
        """Every dot's landing viewport covered -> None, caller raises."""
        decision = make_resource_search_hop(
            _ctx(
                map_fuel_dots=((130, 100),),
                scanned_viewport_origins=[(122, 92)],
            ),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        assert decision is None

    def test_opens_map_when_atlas_empty(self) -> None:
        """With no dots and no recent map open, the hop opens the map.

        The atlas arrives with the 0x4C MapData response, so the first
        hop of a session loads it via ``map_open``.
        """
        decision = make_resource_search_hop(
            _ctx(map_fuel_dots=()),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a map_open to load the dot atlas")
        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "map_for_dots"
        assert decision["updated_ai_state"]["last_map_open_ms"] == 100000

    def test_returns_none_when_atlas_empty_after_recent_map_open(self) -> None:
        """A dotless atlas right after a map open cannot loop on map_open."""
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 98000,
            }
        )
        decision = make_resource_search_hop(
            _ctx(map_fuel_dots=(), ai_state=ai_state),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        assert decision is None

    def test_opens_map_again_after_cooldown_with_empty_atlas(self) -> None:
        """A stale map open (past the cooldown) re-opens for the atlas."""
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 90000,
            }
        )
        decision = make_resource_search_hop(
            _ctx(map_fuel_dots=(), ai_state=ai_state),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a fresh map_open after the cooldown")
        assert decision["command"]["cmd_type"] == "map_open"

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
            _ctx(map_fuel_dots=((130, 100),)),
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
        """Without a terrain map, passability and cleanliness degrade to 1.0."""
        decision = make_resource_search_hop(
            _ctx(terrain=None, map_fuel_dots=((130, 100),)),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
        )

        if decision is None:
            raise AssertionError("expected a hop when terrain is absent")
        command = decision["command"]
        assert command["cmd_type"] == "teleport"
        assert command["target_x"] == 130

    def test_uses_ctx_base_state_when_ai_state_not_provided(self) -> None:
        """Omitting ``ai_state`` uses the context's base state for clearing."""
        decision = make_resource_search_hop(
            _ctx(map_fuel_dots=((130, 100),)),
            mode="COLLECT",
            score=900,
            reason="search_collect_local",
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
