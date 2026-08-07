"""Tests for sniffer world state dispatch handling of tank-related messages."""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state.types import WorldStateDict
from tests.conftest import FakeFileSystem


class TestDispatchTankMessages:
    """Tests for dispatch_world_state_update with tank messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_tank_entry(self) -> None:
        """Test dispatch handles TankEntry (0x28) message."""
        from tankpit_bot.protocol import TankEntryDict

        msg = TankEntryDict(
            msg_type=0x28, team=0, tank_id=42, rank=0, damage_state=0, score=0, x=100, y=150
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "42" in state["tanks"]
        assert state["tanks"]["42"]["name"] == ""
        assert state["tanks"]["42"]["x"] == 100
        assert state["tanks"]["42"]["y"] == 150

    def test_dispatch_tank_status(self) -> None:
        """Test dispatch handles TankStatus (0x3E) message."""
        from tankpit_bot.protocol import TankStatusDict

        msg = TankStatusDict(
            msg_type=0x3E,
            team=2,
            rank=5,
            damage_state=0,
            tank_id=99,
            decoration_state=b"\x00",
            leaderboard_score=1000,
            leaderboard_position=3,
            name="TopPlayer",
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "99" in state["tanks"]
        assert state["tanks"]["99"]["name"] == "TopPlayer"
        assert state["tanks"]["99"]["team"] == 2
        assert state["tanks"]["99"]["rank"] == 5

    def test_dispatch_tank_remove(self) -> None:
        """0x58 TankRemove leaves the tank in the registry (changed 2026-06-22).

        0x58 fires when the server stops broadcasting per-tank
        updates to this client -- which happens on actual deaths but
        also on benign tracking churn (ghost_observe capture
        2026-06-20: orange-5 got 5 TankRemove events across 2
        actual kills). The earlier behaviour deleted the tank, which
        caused the bot to abandon pursuit of locked targets that
        merely teleported out of viewport. The dispatch is now a
        no-op for 0x58 -- the freshness gates and 0x41 Deactivation
        own the lifecycle.
        """
        from tankpit_bot.protocol import TankEntryDict, TankRemoveDict

        entry_msg = TankEntryDict(
            msg_type=0x28, team=0, tank_id=42, rank=0, damage_state=0, score=0, x=100, y=150
        )
        dispatch_world_state_update(get_world_service(), entry_msg)
        tanks = get_world_service().world_state["tanks"]
        assert "42" in tanks
        assert tanks["42"]["liveness"] == "alive"

        import logging

        remove_msg = TankRemoveDict(msg_type=0x58, tank_id=42)
        logger = logging.getLogger("tankpit_bot.runtime.events")
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture()
        original_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            dispatch_world_state_update(get_world_service(), remove_msg)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
        tanks = get_world_service().world_state["tanks"]
        assert "42" in tanks
        assert tanks["42"]["liveness"] == "alive"
        # The removal is timestamped as a diagnostic: the 0x58 starts
        # the server's ~12 s shoot-at-id grace window, so pursuit-miss
        # timing can be correlated against it (2026-07-19).
        removals = [
            r for r in records if r.getMessage() == "DIAGNOSTIC: diagnostic_kind=tank_removed"
        ]
        assert len(removals) == 1
        record_dict: dict[str, str | int | float | bool | dict[str, str | int | float | bool]] = (
            removals[0].__dict__
        )
        assert record_dict["runtime_fields"] == {
            "diagnostic_kind": "tank_removed",
            "tank_id": 42,
        }

    def test_dispatch_tank_exit_does_not_remove_tank(self) -> None:
        """0x29 TankExit is announcement-only; tank stays in world state.

        The actual removal arrives separately via 0x58 TankRemove or
        container tank_leave. JS Vf only prints a log line.
        """
        from tankpit_bot.protocol import TankEntryDict, TankExitDict

        entry_msg = TankEntryDict(
            msg_type=0x28, team=1, tank_id=77, rank=0, damage_state=0, score=0, x=10, y=20
        )
        dispatch_world_state_update(get_world_service(), entry_msg)
        assert "77" in get_world_service().world_state["tanks"]

        exit_msg = TankExitDict(
            msg_type=0x29,
            team=1,
            tank_id=77,
            was_silent=False,
            was_eliminated=True,
        )
        dispatch_world_state_update(get_world_service(), exit_msg)
        assert "77" in get_world_service().world_state["tanks"]

    # Container tank_registry dispatch tests deleted 2026-06-20: container
    # TankRegistry decoder removed after corpus sweep proved zero
    # production fires. Tank join now flows through 0x21 TankInfo /
    # 0x28 TankEntry from the protocol path.

    def test_dispatch_tunneled_terrain_update_sets_terrain_tile(self) -> None:
        """Test 0x4A terrain updates modify world terrain state."""
        from tankpit_bot.protocol import TerrainUpdateDict

        msg = TerrainUpdateDict(msg_type=0x4A, updates=[(8, 166, 2)])
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        tile = state["terrain"]["8,166"]
        assert tile["x"] == 8
        assert tile["y"] == 166
        assert tile["terrain_type"] == 2

    def test_dispatch_tunneled_mine_placement_adds_mines(self) -> None:
        """Test tunneled 0x4B mine placement updates world mine state."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=131,
                y=126,
                direction=8,
                damage_state=0,
                rank=1,
                lb_score=1313,
                carrying=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [
                    (131, 126),
                    (131, 125),
                    (132, 125),
                    (132, 126),
                    (132, 127),
                ],
            },
        )

        state = get_world_service().world_state
        assert state["mines"]["131,126"]["team"] == 2
        assert state["mines"]["131,126"]["tank_id"] == 1301
        assert state["mines"]["131,126"]["mine_type"] == 2
        assert state["mines"]["132,127"]["x"] == 132
        assert state["mines"]["132,127"]["y"] == 127

    def test_dispatch_tunneled_mine_placement_uses_known_tank_team(self) -> None:
        """Test tunneled 0x4B uses tracked tank team when placer is not self."""
        from tankpit_bot.protocol import TankEntryDict, TankInfoDict

        dispatch_world_state_update(
            get_world_service(),
            TankInfoDict(
                msg_type=0x21,
                tank_id=777,
                name="placer",
                team=3,
                decoration_state=b"",
                persistent_tank_id=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            TankEntryDict(
                msg_type=0x28,
                team=3,
                tank_id=777,
                rank=0,
                damage_state=0,
                score=0,
                x=40,
                y=41,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 1,
                "tank_id": 777,
                "positions": [(40, 41), (40, 42)],
            },
        )

        state = get_world_service().world_state
        assert state["mines"]["40,41"]["team"] == 3
        assert state["mines"]["40,42"]["team"] == 3
        assert state["mines"]["40,41"]["tank_id"] == 777

    def test_dispatch_tunneled_mine_placement_skips_unknown_team(self) -> None:
        """Test tunneled 0x4B does nothing when placer team is unknown."""
        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 9999,
                "positions": [(10, 11), (11, 11)],
            },
        )

        state = get_world_service().world_state
        assert state["mines"] == {}

    def test_mine_on_mine_destruction_real_capture(self) -> None:
        """Regression for the 3x3 placement that destroys adjacent enemy mines.

        Captured 2026-06-20 (practice-vs-real-20260620-150138, t+56.15s):
        Artax (team 2, blue) on the tile center placed 7 blue mines and
        the server simultaneously fired a 0x45 MineDetonation listing 2
        adjacent enemy mines (purple) destroyed by the same placement.
        Total = 9 = full 3x3 attempted around the placer per the game
        mechanic (server filters water / terrain / tanks / enemy mines;
        clear tiles get the mine, enemy-mine tiles get the detonation,
        impossible tiles get nothing).

        Post-state: 7 of 9 tiles have our blue mines; 2 tiles are empty
        (the detonated enemy mines did not become our mines -- per
        user, those gaps require re-placement to fill).
        """
        from tankpit_bot.protocol import MovementResponseDict
        from tankpit_bot.state import add_mine

        ws = get_world_service()
        # Establish self at the placement center.
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=133,
                y=124,
                direction=8,
                damage_state=0,
                rank=1,
                lb_score=1313,
                carrying=0,
            ),
        )
        # Seed the two enemy (purple, team=1) mines that the placement
        # is about to detonate.
        ws.world_state = add_mine(ws.world_state, 132, 123, 2, 1229, 1, 1)
        ws.world_state = add_mine(ws.world_state, 134, 125, 2, 1229, 1, 1)
        assert ws.world_state["mines"]["132,123"]["team"] == 1
        assert ws.world_state["mines"]["134,125"]["team"] == 1

        # Wire packet 1: 7 blue mines placed at the 7 clear tiles in the 3x3.
        dispatch_world_state_update(
            ws,
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [
                    (133, 124),
                    (132, 124),
                    (133, 123),
                    (134, 123),
                    (134, 124),
                    (133, 125),
                    (132, 125),
                ],
            },
        )
        # Wire packet 2 (same wire tick): the 2 enemy-mine tiles get
        # 0x45 MineDetonation -- enemy mines destroyed.
        dispatch_world_state_update(
            ws,
            {"msg_type": 0x45, "positions": [(132, 123), (134, 125)]},
        )

        mines = ws.world_state["mines"]
        # 7 own mines placed.
        own_mine_positions = [
            (133, 124),
            (132, 124),
            (133, 123),
            (134, 123),
            (134, 124),
            (133, 125),
            (132, 125),
        ]
        for x, y in own_mine_positions:
            assert mines[f"{x},{y}"]["team"] == 2
            assert mines[f"{x},{y}"]["tank_id"] == 1301
        # 2 detonated tiles are empty -- no own mine, no enemy mine.
        assert "132,123" not in mines
        assert "134,125" not in mines

    def test_mine_cascade_two_packet_chain_real_capture(self) -> None:
        """Regression for one shot detonating a mine + chain detonation.

        Captured 2026-06-20 (practice-vs-real-20260620-150138, t+62.15s):
        Artax shot tile (134, 126) -- the server emitted two 0x45
        MineDetonate packets in the same wire tick. First packet listed
        the directly hit mine [(134, 126)]; second packet listed the
        6 adjacent chain mines destroyed in the cascade
        [(135, 126), (134, 127), (133, 126), (135, 127), (135, 125),
        (133, 127)]. Total 7 tiles cleared.

        World-state must apply both packets atomically (each removes its
        listed positions) and end with all 7 tiles empty regardless of
        the mines' original team.
        """
        from tankpit_bot.state import add_mine

        ws = get_world_service()
        # Seed the 7 mines that the cascade is about to destroy --
        # mix of own (blue, team=2) and enemy (purple, team=1).
        seed = [
            (134, 126, 2, 1301),  # own, the directly-hit one
            (135, 126, 1, 1229),  # enemy
            (134, 127, 2, 1301),  # own
            (133, 126, 1, 1229),  # enemy
            (135, 127, 1, 1229),  # enemy
            (135, 125, 2, 1301),  # own
            (133, 127, 1, 1229),  # enemy
        ]
        for x, y, team, tid in seed:
            ws.world_state = add_mine(ws.world_state, x, y, 2, tid, team, 1)
        assert len(ws.world_state["mines"]) == 7

        # Wire packet 1: directly hit mine.
        dispatch_world_state_update(
            ws,
            {"msg_type": 0x45, "positions": [(134, 126)]},
        )
        # Wire packet 2 (same wire tick): chain detonation cascade.
        dispatch_world_state_update(
            ws,
            {
                "msg_type": 0x45,
                "positions": [
                    (135, 126),
                    (134, 127),
                    (133, 126),
                    (135, 127),
                    (135, 125),
                    (133, 127),
                ],
            },
        )

        # All 7 tiles empty after the cascade -- own and enemy alike.
        mines = ws.world_state["mines"]
        for x, y, _team, _tid in seed:
            assert f"{x},{y}" not in mines

    def test_dispatch_tunneled_mine_detonation_removes_mines(self) -> None:
        """Test tunneled 0x45 removes mines at decoded coordinates."""
        from tankpit_bot.protocol import MovementResponseDict

        dispatch_world_state_update(
            get_world_service(),
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=38,
                y=53,
                direction=8,
                damage_state=0,
                rank=1,
                lb_score=1313,
                carrying=0,
            ),
        )

        dispatch_world_state_update(
            get_world_service(),
            {
                "msg_type": 0x4B,
                "mine_type": 2,
                "tank_id": 1301,
                "positions": [(38, 52), (39, 53), (38, 54)],
            },
        )

        dispatch_world_state_update(
            get_world_service(), {"msg_type": 0x45, "positions": [(39, 53), (38, 54)]}
        )

        state = get_world_service().world_state
        assert "38,52" in state["mines"]
        assert "39,53" not in state["mines"]
        assert "38,54" not in state["mines"]

    def test_dispatch_9byte_2e_routes_through_og_and_updates_damage(self) -> None:
        """0x2E body of length 9 routes to Og.h TankStatusSync.

        Prior to 2026-06-19 these bodies fell to the broken container
        TankStatusShort layout (rank > 8 in 74/74 corpus samples).
        After the fix they decode as Og.h short-form -- and we test
        that the damage byte (Og.h ``a[3]``) flows correctly into
        world state.
        """
        from tankpit_bot.protocol import TankEntryDict, TankStatusSyncDict

        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=300, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(get_world_service(), entry)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=0x40,  # team byte with flag bit 6 set (matches corpus)
            tank_id=300,
            damage_state=3,
            rank=4,
            lb_score=1234,
            promo_state=0,
            fuel=None,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "300" in state["tanks"]
        assert state["tanks"]["300"]["damage_state"] == 3

    def test_tank_damage_transition_emits_diagnostic(self, fake_fs: FakeFileSystem) -> None:
        """A damage-tier change surfaces as a diagnostic instead of silence.

        Run 20260610-223x fired 27 homing shots with zero artifact
        evidence of their effect: the tier sync was consumed silently.
        Repeats of the same tier stay quiet; only transitions emit.
        """
        from pathlib import Path

        from tankpit_bot.diagnostics.event_stream import load_event_records
        from tankpit_bot.protocol import TankEntryDict, TankStatusSyncDict
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        artifacts = configure_bot_runtime_logging("20260610-230000")
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=300, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(get_world_service(), entry)

        for repeated_damage_state in (2, 2):
            msg = TankStatusSyncDict(
                msg_type=0x2E,
                subtype=0x40,
                tank_id=300,
                damage_state=repeated_damage_state,
                rank=4,
                lb_score=1234,
                promo_state=0,
                fuel=None,
            )
            dispatch_world_state_update(get_world_service(), msg)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "tank_damage_changed"
        ]
        assert len(records) == 1
        assert records[0]["fields"] == {
            "diagnostic_kind": "tank_damage_changed",
            "tank_id": 300,
            "tank_name": "",
            # The 0x28 entry's damage byte is the dual-purpose init
            # field and is deliberately dropped; an unobserved tier
            # defaults to DAMAGE_FULL = 3 (assume healthy).
            "previous_damage_state": 3,
            "damage_state": 2,
        }

    def test_tank_damage_for_unknown_tank_emits_nothing(self, fake_fs: FakeFileSystem) -> None:
        """A tier sync for an untracked tank has no transition to report."""
        from pathlib import Path

        from tankpit_bot.diagnostics.event_stream import load_event_records
        from tankpit_bot.protocol import TankStatusSyncDict
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        artifacts = configure_bot_runtime_logging("20260610-230000")
        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=0x40,
            tank_id=999,
            damage_state=3,
            rank=4,
            lb_score=1234,
            promo_state=0,
            fuel=None,
        )
        dispatch_world_state_update(get_world_service(), msg)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "tank_damage_changed"
        ]
        assert records == []

    # Container tank_leave / position_update dispatch tests deleted
    # 2026-06-20: container TankLeave and PositionUpdate decoders removed
    # after corpus sweep proved zero production fires. Tank removal flows
    # through 0x58 TankRemove and position updates through 0x3D
    # MovementResponse on the protocol path.

    # test_dispatch_enemy_movement_with_resolved_player_id deleted
    # 2026-06-19: container Movement / PlayerIdMapper removed. Protocol
    # 0x47 Movement carries tank_id directly per JS Lg.h.


class TestDispatchMapData:
    """Tests for dispatch_world_state_update with 0x4C MapData (Ig)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_map_data_lifts_tank_positions(self) -> None:
        """0x4C advances every tank's authoritative position from the snapshot."""
        from tankpit_bot.protocol import MapDataDict, MapTankEntry, TankEntryDict

        ws = get_world_service()
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

        ws = get_world_service()
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

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_build_pickup_updates_actor_position(self) -> None:
        """0x42 advances the acting tank's position to its source x/y."""
        from tankpit_bot.protocol import BuildPickupDict, TankEntryDict

        ws = get_world_service()
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

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_decoration_is_observation_only_no_state_mutation(self) -> None:
        """0x4E Sf is an announcement: emits a diagnostic, does not mutate tanks."""
        from tankpit_bot.protocol import DecorationDict, TankEntryDict

        ws = get_world_service()
        entry = TankEntryDict(
            msg_type=0x28, team=2, tank_id=44, rank=0, damage_state=0, score=0, x=1, y=2
        )
        dispatch_world_state_update(ws, entry)
        before = ws.world_state["tanks"]["44"]

        msg = DecorationDict(msg_type=0x4E, tank_id=44, slot=1, level=2)
        dispatch_world_state_update(ws, msg)

        after = ws.world_state["tanks"]["44"]
        assert after == before


class TestDispatchSupervisorText:
    """Tests for dispatch_world_state_update with 0x3C SupervisorText (wg)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_supervisor_text_is_observation_only_no_state_mutation(self) -> None:
        """0x3C wg emits a supervisor_text diagnostic; world state untouched.

        Wire format is Latin-1 over the XOR-decoded body per JS p().
        The bot never observed a SupervisorText body in the 150-session
        production corpus (the practice room doesn't get them), so this
        test exercises the dispatch path against a JS-spec'd payload.
        """
        from tankpit_bot.protocol import SupervisorTextDict

        ws = get_world_service()
        before = ws.world_state
        msg = SupervisorTextDict(msg_type=0x3C, message="Test")
        dispatch_world_state_update(ws, msg)
        assert ws.world_state is before


class TestDispatchStatistics:
    """Tests for dispatch_world_state_update with 0x56 Statistics (Wg)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

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

        ws = get_world_service()
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

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_active_players_stores_roster_on_world_service(self) -> None:
        """0x2F ActivePlayers populates ``ws.active_players``."""
        from tankpit_bot.protocol import ActivePlayerEntry, ActivePlayersDict

        ws = get_world_service()
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

        ws = get_world_service()
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

        ws = get_world_service()
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

        ws = get_world_service()
        before = ws.last_ping_response_ms

        dispatch_world_state_update(ws, PingResponseDict(msg_type=0x60))

        # Wall clock advances past zero; the exact ms isn't important.
        assert ws.last_ping_response_ms > before

    def test_connection_lost_is_diagnostic_only(self) -> None:
        """0x7E ConnectionLost does not mutate world state; diagnostic only."""
        from tankpit_bot.protocol import ConnectionLostDict

        ws = get_world_service()
        before = ws.world_state

        dispatch_world_state_update(ws, ConnectionLostDict(msg_type=0x7E))

        assert ws.world_state is before


class TestDispatchPromotion:
    """Tests for dispatch_world_state_update with 0x2B Promotion (Rf) messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_promotion_updates_self_rank(self) -> None:
        """0x2B Rf sets self_state.rank to ``new_rank`` when self is joined."""
        from tankpit_bot.protocol import PromotionDict
        from tankpit_bot.state import update_self_from_movement_response

        ws = get_world_service()
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

        ws = get_world_service()
        before = ws.world_state
        msg = PromotionDict(msg_type=0x2B, new_rank=3, was_promoted=False)
        dispatch_world_state_update(ws, msg)

        assert ws.world_state["self_state"] is None
        assert ws.world_state is before


class TestDispatchEnemyDetection:
    """Tests for dispatch_world_state_update with EnemyDetection (0x48) messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_enemy_detection_creates_tank(self) -> None:
        """Dispatch 0x48 creates enemy tank entry via _update_enemy_from_detection."""
        from tankpit_bot.protocol import EnemyDetectionDict

        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=555,
            x=120,
            y=130,
            rank=3,
            team=2,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "555" in state["tanks"]
        assert state["tanks"]["555"]["x"] == 120
        assert state["tanks"]["555"]["y"] == 130
        assert state["tanks"]["555"]["team"] == 2
        assert state["tanks"]["555"]["rank"] == 3

    def test_dispatch_enemy_detection_updates_existing_tank(self) -> None:
        """Dispatch 0x48 updates position of already-registered enemy tank."""
        from tankpit_bot.protocol import EnemyDetectionDict, TankEntryDict

        # First create a tank with an old position
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=556, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(get_world_service(), entry)

        # Detection updates to new position
        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=556,
            x=200,
            y=210,
            rank=5,
            team=1,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert state["tanks"]["556"]["x"] == 200
        assert state["tanks"]["556"]["y"] == 210


class TestSelfIdentityRecording:
    """The self 0x21 TankInfo fills the canonical account model."""

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_self_tank_info_records_identity(self) -> None:
        """A 0x21 matching the self tank id lands in self_account."""
        from tankpit_bot.protocol import TankInfoDict
        from tankpit_bot.state.self_mutations import update_self_position

        ws = get_world_service()
        ws.world_state = update_self_position(ws.world_state, 100, 100, 1000)
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("update_self_position must create a self state")
        ws.world_state = WorldStateDict(
            self_state={**self_state, "tank_id": 1301},
            tanks=ws.world_state["tanks"],
            containers=ws.world_state["containers"],
            mines=ws.world_state["mines"],
            terrain=ws.world_state["terrain"],
            viewport=ws.world_state["viewport"],
            scanned_tiles=ws.world_state["scanned_tiles"],
            timestamp_ms=ws.world_state["timestamp_ms"],
        )

        dispatch_world_state_update(
            ws,
            TankInfoDict(
                msg_type=0x21,
                tank_id=1301,
                name="Artax",
                team=2,
                decoration_state=b"\x1e\x00\x00\x00",
                persistent_tank_id=62913,
            ),
        )

        account = ws.self_account
        assert account["name"] == "Artax"
        assert account["persistent_tank_id"] == 62913
        assert account["decoration_state_hex"] == "1e000000"
        assert account["identity_observed_ms"] > 0

    def test_other_tank_info_leaves_the_account_model_alone(self) -> None:
        """Roster 0x21s for other tanks never touch self_account."""
        from tankpit_bot.protocol import TankInfoDict

        ws = get_world_service()
        dispatch_world_state_update(
            ws,
            TankInfoDict(
                msg_type=0x21,
                tank_id=777,
                name="stranger",
                team=3,
                decoration_state=b"",
                persistent_tank_id=999,
            ),
        )

        assert ws.self_account["name"] == ""
        assert ws.self_account["persistent_tank_id"] == -1
