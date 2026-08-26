"""Tests for session-level dispatch into world state.

Map data, build pickups, decorations, supervisor text, statistics,
broadcasts, promotions, and enemy detections.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestDispatchMapData:
    """Tests for dispatch_world_state_update with 0x4C MapData (Ig)."""

    def test_map_data_lifts_tank_positions(self) -> None:
        """0x4C advances every tank's authoritative position from the snapshot."""
        from tankpit_bot.protocol import MapDataDict, MapTankEntry, TankEntryDict

        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=7, rank=0, damage_state=0, score=0, x=0, y=0
        )
        dispatch_world_state_update(ws, entry)

        snapshot = MapDataDict(
            msg_type=0x4C,
            tanks=[
                MapTankEntry(x=100, y=120, tank_id=7, rank=2, damage=1, team=0),
            ],
            fuel_dots=[],
        )
        dispatch_world_state_update(ws, snapshot)

        tank = ws.world_state["tanks"]["7"]
        assert tank["x"] == 100
        assert tank["y"] == 120
        assert tank["damage_state"] == 1

    def test_map_data_marks_action_complete(self) -> None:
        """0x4C dispatch must flag map_data_processed so the bot's
        in-flight map_open action clears via the authoritative server signal
        instead of stalling for action_stall_timeout_ms (10 s) and replanning.

        Regression: the dispatcher was decoding MapData and emitting the
        ``map_data_snapshot`` diagnostic but forgetting to call
        ``ws.mark_map_data_processed()``, so ``_clear_completed_map_open``
        (which polls ``check_and_clear_map_data_processed``) always
        returned False. The bot looped open-close-map every 10 seconds
        (live run 2026-06-20: 30 cycles in a 5-min session with zero
        forward progress).
        """
        from tankpit_bot.protocol import MapDataDict

        ws = WorldService()
        # Pre-condition: flag is initially unset.
        assert ws.check_and_clear_map_data_processed() is False

        dispatch_world_state_update(
            ws,
            MapDataDict(msg_type=0x4C, tanks=[], fuel_dots=[]),
        )

        assert ws.check_and_clear_map_data_processed() is True
        # check_and_clear is one-shot -- a second read must return False.
        assert ws.check_and_clear_map_data_processed() is False


class TestDispatchBuildPickup:
    """Tests for dispatch_world_state_update with 0x42 BuildPickup (Jg)."""

    def test_build_pickup_updates_actor_position(self) -> None:
        """0x42 advances the acting tank's position to its source x/y."""
        from tankpit_bot.protocol import BuildPickupDict, TankEntryDict

        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=2, tank_id=12, rank=0, damage_state=0, score=0, x=0, y=0
        )
        dispatch_world_state_update(ws, entry)

        msg = BuildPickupDict(
            msg_type=0x42,
            tank_id=12,
            source_x=20,
            source_y=30,
            drop_x=21,
            drop_y=30,
            direction=8,
            obstacle_type=2,
            flag=0,
        )
        dispatch_world_state_update(ws, msg)

        tank = ws.world_state["tanks"]["12"]
        assert tank["x"] == 20
        assert tank["y"] == 30


class TestDispatchDecoration:
    """Tests for dispatch_world_state_update with 0x4E Decoration (Sf) messages."""

    def test_decoration_is_observation_only_no_state_mutation(self) -> None:
        """0x4E Sf is an announcement: emits a diagnostic, does not mutate tanks."""
        from tankpit_bot.protocol import DecorationDict, TankEntryDict

        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=2, tank_id=44, rank=0, damage_state=0, score=0, x=1, y=2
        )
        dispatch_world_state_update(ws, entry)
        before = ws.world_state["tanks"]["44"]

        msg = DecorationDict(msg_type=0x4E, tank_id=44, slot=1, level=2)
        dispatch_world_state_update(ws, msg)

        after = ws.world_state["tanks"]["44"]
        assert after == before

    def test_unknown_award_slot_dispatches_without_a_name(self) -> None:
        """A slot outside the known table books the numbers, no crash.

        The 0x4E fields are raw bytes; a future server-side award
        category must never take the session down (2026-08-26: the
        FIRST live 0x4E crashed both fleet bots on a reserved-key
        collision — this dispatch is the class's regression guard).
        """
        from tankpit_bot.protocol import DecorationDict

        ws = WorldService()
        msg = DecorationDict(msg_type=0x4E, tank_id=44, slot=9, level=4)

        dispatch_world_state_update(ws, msg)

        assert ws.world_state["tanks"] == {}


class TestDispatchSupervisorText:
    """Tests for dispatch_world_state_update with 0x3C SupervisorText (wg)."""

    def test_supervisor_text_is_observation_only_no_state_mutation(self) -> None:
        """0x3C wg emits a supervisor_text diagnostic; world state untouched.

        Wire format is Latin-1 over the XOR-decoded body per JS p().
        The bot never observed a SupervisorText body in the 150-session
        production corpus (the practice room doesn't get them), so this
        test exercises the dispatch path against a JS-spec'd payload.
        """
        from tankpit_bot.protocol import SupervisorTextDict

        ws = WorldService()
        before = ws.world_state
        msg = SupervisorTextDict(msg_type=0x3C, message="Test")
        dispatch_world_state_update(ws, msg)
        assert ws.world_state is before


class TestDispatchStatistics:
    """Tests for dispatch_world_state_update with 0x56 Statistics (Wg)."""

    def test_statistics_is_observation_only_no_state_mutation(self) -> None:
        """0x56 Wg emits a self_statistics diagnostic; world state untouched.

        Sourced from production capture (2026-06-19) via
        analysis_scripts/crack_tank_update.py: 239/239 corpus samples
        decode to sane minutes/seconds and monotonic playtime/score
        series, so the field semantics are pinned. The dispatcher
        simply forwards them as a diagnostic since they describe the
        own-tank session, not any world geometry.
        """
        from tankpit_bot.protocol import StatisticsDict

        ws = WorldService()
        before = ws.world_state

        # First production sample from the crack run:
        # hrs=40 min=18 sec=31 destroyed=30 deactivated=0 score=55931
        msg = StatisticsDict(
            msg_type=0x56,
            playtime_hours=40,
            playtime_minutes=18,
            playtime_seconds=31,
            destroyed=30,
            deactivated=0,
            score=55931,
        )
        dispatch_world_state_update(ws, msg)

        assert ws.world_state is before


class TestDispatchSessionBroadcasts:
    """Dispatcher coverage for 0x2F/0x31/0x60/0x7E session broadcasts."""

    def test_active_players_stores_roster_on_world_service(self) -> None:
        """0x2F ActivePlayers populates ``ws.active_players``."""
        from tankpit_bot.protocol import ActivePlayerEntry, ActivePlayersDict

        ws = WorldService()
        msg = ActivePlayersDict(
            msg_type=0x2F,
            players=[
                ActivePlayerEntry(tank_id=501, rank=5),
                ActivePlayerEntry(tank_id=1027, rank=2),
            ],
        )
        dispatch_world_state_update(ws, msg)

        assert ws.active_players == [(501, 5), (1027, 2)]

    def test_top10_stores_viewer_snapshot_on_world_service(self) -> None:
        """0x31 Top10 caches the viewer's score/position/team_filter."""
        from tankpit_bot.protocol import Top10Dict, Top10EntryDict

        ws = WorldService()
        msg = Top10Dict(
            msg_type=0x31,
            team_filter=255,
            viewer_score=66051,
            viewer_position=7,
            entries=[
                Top10EntryDict(
                    position=1,
                    score=1056816,
                    team=2,
                    rank=8,
                    name="Yupr",
                    tank_id=-1,
                ),
            ],
        )
        dispatch_world_state_update(ws, msg)

        assert ws.top10_viewer_score == 66051
        assert ws.top10_viewer_position == 7
        assert ws.top10_team_filter == 255

    def test_top10_with_zero_rows_still_updates_viewer_snapshot(self) -> None:
        """An empty Top10 list updates viewer fields but emits zero-row event."""
        from tankpit_bot.protocol import Top10Dict

        ws = WorldService()
        msg = Top10Dict(
            msg_type=0x31,
            team_filter=1,
            viewer_score=0,
            viewer_position=0,
            entries=[],
        )
        dispatch_world_state_update(ws, msg)

        assert ws.top10_viewer_score == 0
        assert ws.top10_viewer_position == 0
        assert ws.top10_team_filter == 1

    def test_ping_response_stamps_world_service(self) -> None:
        """0x60 PingResponse advances ``ws.last_ping_response_ms``."""
        from tankpit_bot.protocol import PingResponseDict

        ws = WorldService()
        before = ws.last_ping_response_ms

        dispatch_world_state_update(ws, PingResponseDict(msg_type=0x60))

        # Wall clock advances past zero; the exact ms isn't important.
        assert ws.last_ping_response_ms > before

    def test_connection_lost_is_diagnostic_only(self) -> None:
        """0x7E ConnectionLost does not mutate world state; diagnostic only."""
        from tankpit_bot.protocol import ConnectionLostDict

        ws = WorldService()
        before = ws.world_state

        dispatch_world_state_update(ws, ConnectionLostDict(msg_type=0x7E))

        assert ws.world_state is before


class TestDispatchPromotion:
    """Tests for dispatch_world_state_update with 0x2B Promotion (Rf) messages."""

    def test_promotion_updates_self_rank(self) -> None:
        """0x2B Rf sets self_state.rank to ``new_rank`` when self is joined."""
        from tankpit_bot.protocol import PromotionDict
        from tankpit_bot.state import update_self_from_movement_response

        ws = WorldService()
        ws.world_state = update_self_from_movement_response(
            ws.world_state,
            tank_id=99,
            x=10,
            y=20,
            team=1,
            rank=1,
            leaderboard_position=5,
            timestamp_ms=500,
        )

        msg = PromotionDict(msg_type=0x2B, new_rank=4, was_promoted=True)
        dispatch_world_state_update(ws, msg)

        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state must exist after a self-join + Promotion")
        assert self_state["rank"] == 4

    def test_promotion_noop_when_no_self_state(self) -> None:
        """0x2B before self has joined leaves world state unchanged."""
        from tankpit_bot.protocol import PromotionDict

        ws = WorldService()
        before = ws.world_state
        msg = PromotionDict(msg_type=0x2B, new_rank=3, was_promoted=False)
        dispatch_world_state_update(ws, msg)

        assert ws.world_state["self_state"] is None
        assert ws.world_state is before


class TestDispatchEnemyDetection:
    """Tests for dispatch_world_state_update with EnemyDetection (0x48) messages."""

    def test_dispatch_enemy_detection_creates_tank(self) -> None:
        """Dispatch 0x48 creates enemy tank entry via _update_enemy_from_detection."""
        from tankpit_bot.protocol import EnemyDetectionDict

        ws = WorldService()
        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=555,
            x=120,
            y=130,
            rank=3,
            team=2,
        )
        dispatch_world_state_update(ws, msg)

        state = ws.world_state
        assert "555" in state["tanks"]
        assert state["tanks"]["555"]["x"] == 120
        assert state["tanks"]["555"]["y"] == 130
        assert state["tanks"]["555"]["team"] == 2
        assert state["tanks"]["555"]["rank"] == 3

    def test_dispatch_enemy_detection_updates_existing_tank(self) -> None:
        """Dispatch 0x48 updates position of already-registered enemy tank."""
        from tankpit_bot.protocol import EnemyDetectionDict, TankEntryDict

        # First create a tank with an old position
        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=556, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(ws, entry)

        # Detection updates to new position
        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=556,
            x=200,
            y=210,
            rank=5,
            team=1,
        )
        dispatch_world_state_update(ws, msg)

        state = ws.world_state
        assert state["tanks"]["556"]["x"] == 200
        assert state["tanks"]["556"]["y"] == 210
