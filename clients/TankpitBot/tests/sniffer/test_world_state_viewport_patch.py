"""Tests for viewport-update dispatch detail."""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state.types import make_viewport_state


class TestDispatchViewportPatchDetail:
    """Tests for viewport-update dispatch detail."""

    def test_dispatch_viewport_update_skips_self_and_zeroed_tanks(self) -> None:
        """Dispatch 0x5A skips self tank and already-invalidated (0,0) tanks."""
        from tankpit_bot.protocol import (
            MovementResponseDict,
            TankEntryDict,
            ViewportUpdateDict,
        )

        get_world_service().world_state["viewport"] = make_viewport_state(
            left=50,
            top=50,
            width=18,
            height=18,
        )

        # Create self at (55, 55) — inside viewport
        self_msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=820,
            x=55,
            y=55,
            direction=0,
            damage_state=0,
            rank=2,
            lb_score=5,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), self_msg)

        # Create enemy already at (0, 0) — already invalidated
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=821, rank=0, damage_state=0, score=0, x=0, y=0
        )
        dispatch_world_state_update(get_world_service(), entry)

        # Viewport update with no entities — should skip self and zeroed tank
        msg = ViewportUpdateDict(msg_type=0x5A, viewport_left=50, viewport_top=50, entities=[])
        dispatch_world_state_update(get_world_service(), msg)

        # Self position preserved (skipped because is_self)
        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["x"] == 55

    def test_dispatch_viewport_update_initializes_from_packet_origin(self) -> None:
        """Dispatch 0x5A uses packet origin even when prior viewport is default."""
        from tankpit_bot.protocol import TankEntryDict, ViewportUpdateDict

        # Default viewport: left=0, top=0, width=18 is still a valid origin.
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=801, rank=0, damage_state=0, score=0, x=5, y=5
        )
        dispatch_world_state_update(get_world_service(), entry)

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=0,
            viewport_top=0,
            entities=[],
        )
        dispatch_world_state_update(get_world_service(), msg)

        assert get_world_service().world_state["viewport"]["left"] == 0
        assert get_world_service().world_state["viewport"]["top"] == 0
        assert get_world_service().world_state["tanks"]["801"]["x"] == 5
        assert get_world_service().world_state["tanks"]["801"]["y"] == 5

    def test_dispatch_viewport_update_does_not_mark_viewport_confirmed(self) -> None:
        """Dispatch 0x5A alone should not count as a radar-confirmed scan."""
        from tankpit_bot.protocol import ViewportUpdateDict

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[],
        )

        dispatch_world_state_update(get_world_service(), msg)

        assert "51,29" not in get_world_service().world_state["scanned_tiles"]

    def test_dispatch_viewport_update_clears_failed_scan_mark(self) -> None:
        """Fresh visible viewport data clears a recent failed-scan mark."""
        from tankpit_bot.protocol import ViewportUpdateDict
        from tankpit_bot.sniffer.world_state import (
            is_scan_viewport_failed,
            mark_scan_viewport_failed,
        )

        mark_scan_viewport_failed(51, 29, 1000)
        assert is_scan_viewport_failed(51, 29, 1001) is True

        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=51,
            viewport_top=29,
            entities=[],
        )
        dispatch_world_state_update(get_world_service(), msg)

        assert is_scan_viewport_failed(51, 29, 1001) is False
