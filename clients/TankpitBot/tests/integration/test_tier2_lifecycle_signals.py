"""Tier 2 integration tests: 5 wire-level lifecycle signal assertions.

Each test drives a real protocol message through
``dispatch_world_state_update`` (or through the bot's stall machinery)
and asserts a single observable lifecycle effect. They lock in the
state transitions that the smoke + Tier 1 tests assume:

  1. test_stall_timeout_replans_to_idle
  2. test_teleport_landed_clears_action
  3. test_radar_scan_returns_to_idle
  4. test_mine_placement_updates_world
  5. test_combat_hit_advances_damage_state

Wire-byte and position values mirror the practice-vs-real
2026-06-20 capture so the assertions stay grounded in real data.
"""

from __future__ import annotations

from tankpit_bot.protocol import (
    CombinedTileUpdateDict,
    MovementResponseDict,
    ShootEventDict,
    TankInfoDict,
    TankStatusSyncDict,
)
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete,
    get_world_service,
    reset_world_state,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_teleport_landed,
    mark_teleport_landed,
)
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar


class TestStallTimeoutReplansToIdle:
    """A stall_timeout clears the bot's in-flight action."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_stall_timeout_clears_in_flight_action(self) -> None:
        """The shared stall-timeout helper sets the in-flight action to ``none``.

        ``_clear_stalled_action`` is the production code path that
        replans to IDLE when an action exceeds
        ``action_stall_timeout_ms``. We invoke it directly to assert
        the side effect without driving the full tick loop (which
        needs Playwright bring-up).

        Timing is forced by stamping ``started_ms`` 11 seconds before
        the real clock so the elapsed-vs-timeout check trips on the
        first call regardless of wall-clock fluctuation.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.states import make_in_flight_action, transition_to
        from tankpit_bot.bot.tick_loop_actions import _clear_stalled_action
        from tankpit_bot.browser.cdp_utils import get_current_time_ms

        bot = Bot("https://test.tankpit.com/", headless=True)
        # Walk the state machine through its valid INITIALIZING ->
        # WAITING_FOR_POSITION -> IDLE -> MOVING ladder so the stall
        # handler's replan-to-IDLE transition is permitted.
        bot._state_data = transition_to(bot._state_data, "WAITING_FOR_POSITION")
        bot._state_data = transition_to(bot._state_data, "IDLE")
        bot._state_data = transition_to(bot._state_data, "MOVING")
        # Stamp the action 11 seconds in the past (above the default
        # 10 000 ms ``action_stall_timeout_ms``).
        action = make_in_flight_action(
            kind="move",
            target_x=131,
            target_y=124,
            started_ms=get_current_time_ms() - 11_000,
        )
        bot._state_data = transition_to(bot._state_data, "MOVING", in_flight_action=action)

        cleared = _clear_stalled_action(bot, action)
        assert cleared is True
        assert bot._state_data["in_flight_action"]["kind"] == "none"
        assert bot.get_state() == "IDLE"


class TestTeleportLandedClearsAction:
    """A TeleportLanded container message flips the teleport-landed flag."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_teleport_landed_dispatch_flips_flag(self) -> None:
        """A ``teleport_landed`` dispatch flips the WS flag exactly once.

        ``check_and_clear_teleport_landed`` is the one-shot poll the
        bot uses to clear an in-flight teleport. The first read after a
        landing must return True; the second must return False so the
        bot does not double-clear.
        """
        from tankpit_bot.container import TeleportLandedDict

        ws = get_world_service()
        assert check_and_clear_teleport_landed(ws) is False

        dispatch_world_state_update(
            ws,
            TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C),
        )

        assert check_and_clear_teleport_landed(ws) is True
        # One-shot: second read returns False so the bot does not
        # double-clear a subsequent action.
        assert check_and_clear_teleport_landed(ws) is False

    def test_direct_mark_also_flips_flag(self) -> None:
        """The internal helper alone (without dispatch) also flips the flag.

        Documents the contract: the bot may mark the landing directly
        in response to a DOM event when the wire is silent.
        """
        ws = get_world_service()
        mark_teleport_landed(ws)
        assert check_and_clear_teleport_landed(ws) is True


class TestRadarScanReturnsToIdle:
    """0x4F CombinedTileUpdate / RadarScanResult flips the radar-complete flag."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_combined_tile_update_marks_radar_cache_refresh(self) -> None:
        """A 0x4F dispatch marks pending radar cache refresh so the bot replans.

        ``mark_pending_radar_cache_refresh`` is the one-shot signal the
        bot polls to know its in-flight radar action has resolved.
        Asserts the dispatch path actually marks it.
        """
        ws = get_world_service()
        assert ws.consume_pending_radar_cache_refresh() is False

        dispatch_world_state_update(
            ws,
            CombinedTileUpdateDict(
                msg_type=0x4F,
                cache_updates=[(40, 50, 600)],
                overlay_updates=[(40, 50, 12)],
            ),
        )

        assert ws.consume_pending_radar_cache_refresh() is True
        # One-shot consumer.
        assert ws.consume_pending_radar_cache_refresh() is False

    def test_radar_response_with_containers_marks_scan_complete(self) -> None:
        """A radar response with containers marks the radar-scan-complete flag.

        ``update_world_state_from_radar`` is the production helper used
        by the dispatch path for non-empty radar payloads; calling it
        sets ``radar_scan_complete`` so the bot's poll returns True.
        """
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        ws = get_world_service()
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=132, y=125, volume=600),
        ]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(ws, containers, mines)
        assert check_and_clear_radar_scan_complete() is True
        assert check_and_clear_radar_scan_complete() is False


class TestMinePlacementUpdatesWorld:
    """Tunneled 0x4B MinePlacement adds mines to world state.

    Uses the real-combat wire body discovered in the 2026-06-20
    practice-vs-real capture: a 7-position mine cluster placed around
    (133, 124) by Artax (tank 1301, blue team).
    """

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_seven_position_placement_adds_every_mine(self) -> None:
        """All 7 positions land in ``world_state["mines"]`` keyed by ``x,y``."""
        ws = get_world_service()
        # Seed the placer (Artax / tank 1301, blue team=2) so the
        # dispatch can attribute the mines.
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=131,
                y=122,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=72,
                carrying=0,
            ),
        )

        # The 7-position cluster, exact tile values from the capture.
        positions = [
            (133, 124),
            (132, 124),
            (133, 123),
            (134, 123),
            (134, 124),
            (133, 125),
            (132, 125),
        ]
        dispatch_world_state_update(
            ws,
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": positions,
            },
        )

        mines = ws.world_state["mines"]
        for x, y in positions:
            key = f"{x},{y}"
            assert key in mines, f"expected mine at {key}, got keys {sorted(mines)}"
            assert mines[key]["tank_id"] == 1301
            assert mines[key]["team"] == 2
            assert mines[key]["mine_type"] == 2


class TestCombatHitAdvancesDamageState:
    """A 0x2E TankStatusSync with a higher damage tier advances the tank.

    Damage tiers count DOWN toward deactivation in the wire schema
    (0=full, 3=light, 2=medium, 1=critical). We assert that the
    dispatch path lifts the tier from a registry entry's previous
    value to the new value.
    """

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_status_sync_drops_damage_tier(self) -> None:
        """A status-sync with ``damage_state=2`` lowers the tier from 3 to 2.

        The bot's hit-detection treats any drop in ``damage_state`` as
        confirmation that its shot connected. We seed the enemy at the
        higher (healthier) tier and assert the dispatched status-sync
        applies.
        """
        ws = get_world_service()
        # Register the enemy (Yuppler, purple team=1, tank=1229).
        dispatch_world_state_update(
            ws,
            TankInfoDict(
                msg_type=0x21,
                tank_id=1229,
                team=1,
                name="Yuppler",
                decoration_state=b"",
                persistent_tank_id=0,
            ),
        )
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1229,
                x=131,
                y=124,
                direction=0,
                damage_state=3,  # light
                rank=1,
                lb_score=107,
                carrying=0,
            ),
        )
        before = ws.world_state["tanks"]["1229"]["damage_state"]
        assert before == 3

        # A subsequent status-sync drops the tier to 2 (medium).
        dispatch_world_state_update(
            ws,
            TankStatusSyncDict(
                msg_type=0x2E,
                subtype=1,
                tank_id=1229,
                damage_state=2,
                rank=1,
                lb_score=107,
                promo_state=0,
                fuel=None,
            ),
        )

        after = ws.world_state["tanks"]["1229"]["damage_state"]
        assert after == 2

    def test_shoot_event_marks_combat_hit(self) -> None:
        """Our 0x53 ShootEvent landing on a tracked tank marks a hit.

        The tile-occupancy mechanism is the wire's authoritative hit
        signal per JS ``Gg.h``: a shot whose ``target_x``/``target_y``
        matches a non-self tank's position records that tank's id as
        the victim.
        """
        from tankpit_bot.protocol import TankEntryDict
        from tankpit_bot.sniffer.world_state_combat import check_and_clear_combat_hit

        ws = get_world_service()
        # Self (Artax / blue=2, tank=1301).
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=131,
                y=122,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=72,
                carrying=0,
            ),
        )
        # Enemy tank registered at the shot target tile.
        dispatch_world_state_update(
            ws,
            TankEntryDict(
                msg_type=0x28,
                team=1,
                tank_id=1229,
                rank=1,
                damage_state=3,
                score=0,
                x=131,
                y=124,
            ),
        )

        # Our shot lands on Yuppler's tile (131, 124).
        dispatch_world_state_update(
            ws,
            ShootEventDict(
                msg_type=0x53,
                team=2,
                shooter_id=1301,
                source_x=131,
                source_y=122,
                target_x=131,
                target_y=124,
                aim_x=131,
                aim_y=124,
                weapon=0,
            ),
        )

        assert check_and_clear_combat_hit(ws) is True
        assert ws.last_shot_victim_id == 1229
