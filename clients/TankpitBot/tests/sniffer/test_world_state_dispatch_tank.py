"""Tests for tank-message dispatch into world state.

The per-tank wire channels and the self-identity recording they drive.
Mine dispatch is :mod:`tests.sniffer.test_world_state_dispatch_mines`.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state.types import WorldStateDict
from tests._runtime_logging_support import capture_runtime_events
from tests.conftest import FakeFileSystem


class TestDispatchTankMessages:
    """Tank entry, status, removal, terrain, and damage dispatch."""

    def test_dispatch_tank_entry(self) -> None:
        """Test dispatch handles TankEntry (0x28) message."""
        from tankpit_bot.protocol import TankEntryDict

        ws = WorldService()
        msg = TankEntryDict(
            msg_type=0x28, team=0, tank_id=42, rank=0, damage_state=0, score=0, x=100, y=150
        )
        dispatch_world_state_update(ws, msg)

        state = ws.world_state
        assert "42" in state["tanks"]
        assert state["tanks"]["42"]["name"] == ""
        assert state["tanks"]["42"]["x"] == 100
        assert state["tanks"]["42"]["y"] == 150

    def test_dispatch_tank_status(self) -> None:
        """Test dispatch handles TankStatus (0x3E) message."""
        from tankpit_bot.protocol import TankStatusDict

        ws = WorldService()
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
        dispatch_world_state_update(ws, msg)

        state = ws.world_state
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

        ws = WorldService()
        entry_msg = TankEntryDict(
            msg_type=0x28, team=0, tank_id=42, rank=0, damage_state=0, score=0, x=100, y=150
        )
        dispatch_world_state_update(ws, entry_msg)
        tanks = ws.world_state["tanks"]
        assert "42" in tanks
        assert tanks["42"]["liveness"] == "alive"

        remove_msg = TankRemoveDict(msg_type=0x58, tank_id=42)
        with capture_runtime_events() as records:
            dispatch_world_state_update(ws, remove_msg)
        tanks = ws.world_state["tanks"]
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

        ws = WorldService()
        entry_msg = TankEntryDict(
            msg_type=0x28, team=1, tank_id=77, rank=0, damage_state=0, score=0, x=10, y=20
        )
        dispatch_world_state_update(ws, entry_msg)
        assert "77" in ws.world_state["tanks"]

        exit_msg = TankExitDict(
            msg_type=0x29,
            team=1,
            tank_id=77,
            was_silent=False,
            was_eliminated=True,
        )
        dispatch_world_state_update(ws, exit_msg)
        assert "77" in ws.world_state["tanks"]

    def test_dispatch_tunneled_terrain_update_sets_terrain_tile(self) -> None:
        """Test 0x4A terrain updates modify world terrain state."""
        from tankpit_bot.protocol import TerrainUpdateDict

        ws = WorldService()
        msg = TerrainUpdateDict(msg_type=0x4A, updates=[(8, 166, 2)])
        dispatch_world_state_update(ws, msg)

        state = ws.world_state
        tile = state["terrain"]["8,166"]
        assert tile["x"] == 8
        assert tile["y"] == 166
        assert tile["terrain_type"] == 2

    def test_dispatch_9byte_2e_routes_through_og_and_updates_damage(self) -> None:
        """0x2E body of length 9 routes to Og.h TankStatusSync.

        Prior to 2026-06-19 these bodies fell to the broken container
        TankStatusShort layout (rank > 8 in 74/74 corpus samples).
        After the fix they decode as Og.h short-form -- and we test
        that the damage byte (Og.h ``a[3]``) flows correctly into
        world state.
        """
        from tankpit_bot.protocol import TankEntryDict, TankStatusSyncDict

        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=300, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(ws, entry)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=0x40,  # team byte with flag bit 6 set (matches corpus)
            tank_id=300,
            damage_state=3,
            rank=4,
            lb_score=1234,
            promo_state=0,
            promo_bar_lit=None,
            fuel=None,
        )
        dispatch_world_state_update(ws, msg)

        state = ws.world_state
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

        ws = WorldService()
        artifacts = configure_bot_runtime_logging("20260610-230000")
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=300, rank=0, damage_state=0, score=0, x=50, y=60
        )
        dispatch_world_state_update(ws, entry)

        for repeated_damage_state in (2, 2):
            msg = TankStatusSyncDict(
                msg_type=0x2E,
                subtype=0x40,
                tank_id=300,
                damage_state=repeated_damage_state,
                rank=4,
                lb_score=1234,
                promo_state=0,
                promo_bar_lit=None,
                fuel=None,
            )
            dispatch_world_state_update(ws, msg)

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

        ws = WorldService()
        artifacts = configure_bot_runtime_logging("20260610-230000")
        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=0x40,
            tank_id=999,
            damage_state=3,
            rank=4,
            lb_score=1234,
            promo_state=0,
            promo_bar_lit=None,
            fuel=None,
        )
        dispatch_world_state_update(ws, msg)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "tank_damage_changed"
        ]
        assert records == []


class TestSelfIdentityRecording:
    """The self 0x21 TankInfo fills the canonical account model."""

    def test_self_tank_info_records_identity(self) -> None:
        """A 0x21 matching the self tank id lands in self_account."""
        from tankpit_bot.protocol import TankInfoDict
        from tankpit_bot.state.self_mutations import update_self_position

        ws = WorldService()
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

        ws = WorldService()
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
