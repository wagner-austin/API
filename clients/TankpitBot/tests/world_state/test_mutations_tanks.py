"""Tests for tank-registry mutations.

Removal, deactivation, and last-aim recording.
"""

from tankpit_bot.state import (
    deactivate_tank,
    make_empty_world_state,
    remove_tank,
)


class TestRemoveTank:
    """Tests for remove_tank (the 0x58 TankRemove handler)."""

    def test_keeps_tank_in_registry(self) -> None:
        """0x58 leaves the registry entry intact (changed 2026-06-22).

        Earlier behaviour deleted the tank, which caused the bot to
        abandon pursuit of locked targets that merely teleported out
        of viewport (live capture 2026-06-22). 0x58 is benign tracking
        churn: orange-5 got 5 TankRemove events across 2 actual kills
        (ghost_observe capture 2026-06-20). Only 0x41 Deactivation is
        an authoritative death signal; keeping 0x58 a no-op lets the
        freshness / liveness gates do the work.
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=42,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(50, 50),
                team=0,
                rank=1,
                name="Test",
                is_bot=False,
            ),
        )
        assert state["tanks"]["42"]["liveness"] == "alive"

        updated = remove_tank(state, tank_id=42, timestamp_ms=1000)

        assert "42" in updated["tanks"]
        assert updated["tanks"]["42"]["liveness"] == "alive"
        assert updated is state

    def test_no_op_for_nonexistent_tank(self) -> None:
        """A 0x58 for a tank we have never heard of is also a no-op."""
        state = make_empty_world_state()
        updated = remove_tank(state, tank_id=999, timestamp_ms=1000)

        assert updated is state


class TestDeactivateTank:
    """Tests for deactivate_tank (0x41 corpse-window handler)."""

    def test_map_snapshot_revives_a_deactivated_tank(self) -> None:
        """Rule 4: presence in map data flips a corpse belief to alive.

        Byte-proven 2026-08-05 (run bot-20260805-095935): the server's
        0x4C map is a strictly LIVING-tanks list — victims absent from
        all 58 in-corpse-window snapshots, present in all 204
        post-window ones. A deactivated tank listed in a map snapshot
        is therefore the living respawn; without this rule 27 of 32
        idle (wire-silent) respawns stayed phantom corpses and the
        session exited no_viable_targets in a full room.
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=505,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(202, 194),
                team=3,
                rank=1,
                name="orange-6",
                is_bot=True,
            ),
        )
        state = deactivate_tank(state, tank_id=505, timestamp_ms=1000)
        assert state["tanks"]["505"]["liveness"] == "deactivated"

        revived = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=505,
                timestamp_ms=30_000,
                is_wire_sourced=False,
                storage_source="world_state",
                fact_source="wire_0x4C_map_data",
                position_is_authoritative=True,
                position=(40, 61),
            ),
        )

        assert revived["tanks"]["505"]["liveness"] == "alive"
        assert revived["tanks"]["505"]["x"] == 40

    def test_radar_detection_does_not_revive_a_corpse(self) -> None:
        """Radar and DOM refinements (both flags False) stay excluded:
        their positions are estimates, not the server's living list."""
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=505,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(202, 194),
                team=3,
                rank=1,
                name="orange-6",
                is_bot=True,
            ),
        )
        state = deactivate_tank(state, tank_id=505, timestamp_ms=1000)

        still_dead = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=505,
                timestamp_ms=30_000,
                is_wire_sourced=False,
                storage_source="world_state",
                fact_source="wire_0x48_enemy_detect",
                position_is_authoritative=False,
                position=(40, 61),
            ),
        )

        assert still_dead["tanks"]["505"]["liveness"] == "deactivated"

    def test_marks_tank_deactivated(self) -> None:
        """Existing tank flips to ``liveness="deactivated"`` and keeps tile.

        Replays the 2026-06-20 ghost_visual kill cycle at the
        world-state layer: TankEntry establishes orange-8 at
        (170, 174); Deactivation marks it ``deactivated`` while
        preserving the death tile (the bot still reasons about that
        tile for mines, fuel deposits, etc.).
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=534,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(170, 174),
                team=3,
                rank=2,
                name="orange-8",
                is_bot=True,
            ),
        )
        assert state["tanks"]["534"]["liveness"] == "alive"

        updated = deactivate_tank(state, tank_id=534, timestamp_ms=1000)

        tank = updated["tanks"]["534"]
        assert tank["liveness"] == "deactivated"
        assert tank["x"] == 170
        assert tank["y"] == 174
        assert tank["timestamp_ms"] == 1000

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Deactivating an unknown tank id is a no-op."""
        state = make_empty_world_state()
        updated = deactivate_tank(state, tank_id=999, timestamp_ms=1000)
        assert updated is state


class TestSetTankLastAim:
    """Tests for ``set_tank_last_aim`` -- the 0x53 ShootEvent persistence path."""

    def test_unknown_tank_is_a_no_op(self) -> None:
        """A shoot event from a tank we have never seen leaves state unchanged.

        The next per-tank wire message will create the tank record;
        dropping the aim quietly is preferable to fabricating a tank
        from a shoot-event alone (no team / rank / name information).
        """
        from tankpit_bot.state.tank_mutations import set_tank_last_aim

        state = make_empty_world_state()

        result = set_tank_last_aim(
            state,
            tank_id=999,
            aim_x=100,
            aim_y=120,
            weapon=1,
            timestamp_ms=5000,
        )

        assert result is state
