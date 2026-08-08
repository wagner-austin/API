"""Tests for the walk-over mine hit stamp and the walk→teleport flip."""

from __future__ import annotations

from tankpit_bot.container import MineDetonationDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state import make_self_state


def _seed_self(ws: WorldService, x: int, y: int) -> None:
    """Place the bot's self record at the given tile.

    Args:
        ws: The world service the record belongs to.
        x: Self X.
        y: Self Y.
    """
    ws.world_state["self_state"] = make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )


class TestOwnMineHitStamp:
    """Tests for the 0x45 own-tile detonation stamp."""

    def test_detonation_on_own_tile_stamps_the_flip_window(self) -> None:
        """A 0x45 containing our tile is the walk-over signature."""
        ws = WorldService()
        _seed_self(ws, 100, 100)

        dispatch_world_state_update(
            ws,
            MineDetonationDict(msg_type=0x45, positions=[(100, 100)]),
        )

        assert ws.last_own_mine_hit_ms > 0
        assert ws.recent_own_mine_hit(ws.last_own_mine_hit_ms) is True

    def test_remote_detonation_does_not_stamp(self) -> None:
        """Mines dying elsewhere are not walk-overs."""
        ws = WorldService()
        _seed_self(ws, 100, 100)

        dispatch_world_state_update(
            ws,
            MineDetonationDict(msg_type=0x45, positions=[(120, 120)]),
        )

        assert ws.last_own_mine_hit_ms == 0

    def test_detonation_before_self_sync_does_not_stamp(self) -> None:
        """With no self record there is no own tile to match."""
        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MineDetonationDict(msg_type=0x45, positions=[(100, 100)]),
        )

        assert ws.last_own_mine_hit_ms == 0

    def test_flip_window_expires(self) -> None:
        """The walk→teleport flip lapses so walking resumes."""
        ws = WorldService()
        _seed_self(ws, 100, 100)
        dispatch_world_state_update(
            ws,
            MineDetonationDict(msg_type=0x45, positions=[(100, 100)]),
        )
        stamped = ws.last_own_mine_hit_ms

        assert ws.recent_own_mine_hit(stamped + 5_999) is True
        assert ws.recent_own_mine_hit(stamped + 6_000) is False
