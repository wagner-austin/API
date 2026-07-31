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
