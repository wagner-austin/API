"""Integration test: 0x4C MapData clears the in-flight map_open action.

Mirrors ``test_map_data_marks_action_complete`` in
``tests/sniffer/test_world_state_dispatch_tank.py`` at the integration
boundary: drives the protocol-level ``MapDataDict`` through
``dispatch_world_state_update`` and asserts the world service's
``check_and_clear_map_data_processed`` flag flips True so the bot's
``_clear_completed_map_open`` poll succeeds.

Regression: 2026-06-20. The dispatcher decoded MapData and emitted the
``map_data_snapshot`` diagnostic but forgot to call
``ws.mark_map_data_processed()``; the bot looped open-close-map every
10 s for the entire 5-min ``make run`` session with zero forward
progress.
"""

from __future__ import annotations

from tankpit_bot.protocol import MapDataDict
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestMapOpenClearsOnMapData:
    """Integration test for the map_open clear signal."""

    def test_map_data_dispatch_flips_processed_flag(self) -> None:
        """A 0x4C MapData dispatch must flag map_data_processed.

        Pre-condition: flag is initially False (one-shot semantics
        guaranteed by ``check_and_clear_map_data_processed``).
        Post-condition: True on first read, False on second.
        """
        ws = get_world_service()
        assert ws.check_and_clear_map_data_processed() is False

        dispatch_world_state_update(
            ws,
            MapDataDict(msg_type=0x4C, tanks=[], fuel_dots=[]),
        )

        assert ws.check_and_clear_map_data_processed() is True
        # One-shot: a second read returns False so subsequent ticks do
        # not double-clear the bot's action.
        assert ws.check_and_clear_map_data_processed() is False

    def test_map_data_with_real_payload_also_flips_flag(self) -> None:
        """Real MapData payloads (fuel dots + tank entries) also flip the flag.

        Uses real positional values from the practice-vs-real capture
        2026-06-20 15:02:08: 608 fuel_dots and 37 tank entries. We
        assert the flag flips regardless of payload size.
        """
        from tankpit_bot.protocol import MapTankEntry

        ws = get_world_service()
        dispatch_world_state_update(
            ws,
            MapDataDict(
                msg_type=0x4C,
                tanks=[
                    MapTankEntry(x=131, y=126, tank_id=1301, rank=1, damage=3, team=2),
                ],
                fuel_dots=[],
            ),
        )

        assert ws.check_and_clear_map_data_processed() is True
