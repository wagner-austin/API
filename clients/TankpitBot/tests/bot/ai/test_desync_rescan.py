"""Tests for the container-desync radar resync gate.

User ruling 2026-07-30: "if one item is stale or out of sync then its
worth a radar. not, 3 items. a single desync." Session 4 receipt:
three larder hops in a row landed on containers Yuppler had already
collected that session, each landing scan suppressed as verified
stock, three teleports wasted.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_state import (
    container_desync_pending,
    get_world_service,
    mark_container_desync,
    reset_world_state,
)
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


class TestDesyncRescan:
    """Tests for the desync-rescan cascade gate and its latch."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_pending_desync_outranks_remembered_container_pursuit(self) -> None:
        """A pending disproof produces one radar before any pickup."""
        world, self_state = make_world(
            scanned=False,
            fuel=150,
            containers={
                "105,105": make_container_state(
                    x=105,
                    y=105,
                    is_fuel=True,
                    volume=700,
                    timestamp_ms=100000,
                    failed_pickups=0,
                )
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
            }
        )
        mark_container_desync(99000)
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected collect decision")
        assert decision["behavior"]["reason_kind"] == "desync_rescan"
        assert decision["command"]["cmd_type"] == "radar"

    def test_no_desync_leaves_cascade_untouched(self) -> None:
        """Without a pending disproof the cascade picks up as usual."""
        world, self_state = make_world(
            fuel=150,
            containers={
                "100,101": make_container_state(
                    x=100,
                    y=101,
                    is_fuel=True,
                    volume=700,
                    timestamp_ms=100000,
                    failed_pickups=0,
                )
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
            }
        )
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected collect decision")
        assert decision["behavior"]["reason_kind"] != "desync_rescan"

    def test_radar_response_clears_the_latch(self) -> None:
        """The radar response is the resync -- it answers the disproof."""
        mark_container_desync(99000)
        assert container_desync_pending() is True

        update_world_state_from_radar(get_world_service(), [], [], [])

        assert container_desync_pending() is False

    def test_radar_cache_refresh_clears_the_latch(self) -> None:
        """A cache-refresh radar answer also clears the latch.

        Session 5 of run 20260730 burned all 22 extra radars in a 2 s
        loop: the server answered every rescan with "Radar cache
        refresh" and the first latch clear lived only on the
        full-delta path. Every response shape lands in
        ``mark_radar_scan_complete``, which now owns the clear.
        """
        from tankpit_bot.sniffer.world_state_radar import (
            update_world_state_from_radar_cache,
        )

        mark_container_desync(99000)
        assert container_desync_pending() is True

        update_world_state_from_radar_cache(get_world_service())

        assert container_desync_pending() is False


class TestRadarSpendEconomics:
    """The shared radar-spend rule and its consumers."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_covered_viewport_answers_the_desync_without_a_scan(self) -> None:
        """Live coverage clears the latch instead of spending a radar.

        Flag s9-4: two desync rescans of ground radared seconds
        earlier each consumed an extra and revealed nothing.
        """
        world, self_state = make_world(
            fuel=150,
            containers={
                "105,105": make_container_state(
                    x=105,
                    y=105,
                    is_fuel=True,
                    volume=700,
                    timestamp_ms=100000,
                    failed_pickups=0,
                )
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
            }
        )
        mark_container_desync(99000)
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        decision = decide_collect_mode(ctx)

        if decision is None:
            raise AssertionError("expected collect decision")
        assert decision["behavior"]["reason_kind"] != "desync_rescan"
        assert container_desync_pending() is False

    def test_spend_floor_only_binds_with_extras_stocked(self) -> None:
        """A radar-broke tank scans any uncovered sliver for free."""
        from tankpit_bot.bot.ai.context import radar_spend_worthwhile

        world, self_state = make_world(scanned=True)
        left = world["viewport"]["left"]
        top = world["viewport"]["top"]
        del world["scanned_tiles"][f"{left + 2},{top + 2}"]
        ai_state = make_scanned_ai_state()

        stocked = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")
        broke = DecideCtx(
            world, self_state, ai_state, make_inventory(default_count=0), 100000, None, ""
        )

        assert radar_spend_worthwhile(stocked) is False
        assert radar_spend_worthwhile(broke) is True

    def test_last_extra_is_not_dribbled_on_a_partial_reveal(self) -> None:
        """Radar reserve (user ruling 2026-07-31): the final extra holds.

        42 uncovered tiles clear the stocked 32-tile floor but not the
        last-extra 128-tile bar — running dry mid-session left the bot
        "dead in the water" restocking through the built-in radius-2
        scan, so the final paid sweep waits for a near-full reveal.
        """
        from tankpit_bot.bot.ai.context import radar_spend_worthwhile

        world, self_state = make_world(scanned=True)
        left = world["viewport"]["left"]
        top = world["viewport"]["top"]
        for dy in range(2, 5):
            for dx in range(2, 16):
                del world["scanned_tiles"][f"{left + dx},{top + dy}"]
        ai_state = make_scanned_ai_state()

        two_left = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(default_count=2),
            100000,
            None,
            "",
        )
        last_extra = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(default_count=1),
            100000,
            None,
            "",
        )

        assert radar_spend_worthwhile(two_left) is True
        assert radar_spend_worthwhile(last_extra) is False

    def test_last_extra_spends_on_a_near_full_reveal(self) -> None:
        """A fully uncovered viewport is worth the final extra radar."""
        from tankpit_bot.bot.ai.context import radar_spend_worthwhile

        world, self_state = make_world(scanned=False)
        ai_state = make_scanned_ai_state()

        last_extra = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(default_count=1),
            100000,
            None,
            "",
        )

        assert radar_spend_worthwhile(last_extra) is True


class TestDisplacedLandingScanEconomics:
    """The displaced-harvest radar obeys the spend economics."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_displaced_landing_in_live_coverage_skips_the_radar(self) -> None:
        """Flag s9-2: a displaced harvest landing in fully-scanned
        ground latched WITHOUT spending an extra."""
        from tankpit_bot.bot.ai.collect_mode import _scan_on_landing_decision

        world, self_state = make_world(
            fuel=900,
            containers={
                "104,100": make_container_state(
                    x=104,
                    y=100,
                    is_fuel=True,
                    volume=700,
                    timestamp_ms=100000,
                    failed_pickups=0,
                )
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(landing_scan_viewport=""),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "suppress_landing_scan": True,
                "resource_target_kind": "fuel",
                "resource_target_x": 104,
                "resource_target_y": 100,
            }
        )
        ctx = DecideCtx(world, self_state, ai_state, make_inventory(), 100000, None, "")

        decision, updated = _scan_on_landing_decision(ctx, ctx.base)

        assert decision is None
        assert updated["last_landing_scan_viewport"] != ""
        assert updated["suppress_landing_scan"] is False
