"""Tests for sniffer world state blob parsing (map response tank positions)."""

from __future__ import annotations

from tankpit_bot.container import WorldStateDict as WorldStateBlobDict
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update


class TestWorldStateBlobParsing:
    """Tests for world_state blob parsing (map response tank positions)."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    @staticmethod
    def _make_msg(blob: bytes) -> WorldStateBlobDict:
        """Wrap blob in a properly typed world_state dispatch message."""
        return WorldStateBlobDict(
            msg_type="world_state",
            subtype=0,
            length=len(blob),
            world_data=blob,
        )

    def _build_world_state_blob(
        self,
        dot_section: bytes,
        tank_entries: list[tuple[int, int, int, int, int]],
    ) -> bytes:
        """Build a world_state blob with a fuel-dot layer and tank entries.

        Args:
            dot_section: Raw skip-RLE fuel-dot layer bytes (cursor starts
                at world (1,1); each byte advances x; byte 255 is a pure
                skip; any other byte also drops a dot).
            tank_entries: List of (x, y, tank_id, team, rank) tuples.

        Returns:
            Raw bytes matching the verified format.
        """
        # 2-byte LE dot-section length
        data = bytearray(len(dot_section).to_bytes(2, "little"))
        data.extend(dot_section)
        # 5-byte tank entries: [x][y][id_lo][id_hi][packed]
        for x, y, tank_id, team, rank in tank_entries:
            id_lo = tank_id & 0xFF
            id_hi = (tank_id >> 8) & 0xFF
            packed = (team & 0x03) | ((rank & 0x0F) << 4)
            data.extend(bytes([x, y, id_lo, id_hi, packed]))
        return bytes(data)

    def test_parses_tank_positions_from_blob(self) -> None:
        """Blob with 3 tanks populates world state with correct positions."""
        blob = self._build_world_state_blob(
            dot_section=b"\xff" * 10,
            tank_entries=[
                (100, 120, 500, 1, 2),  # red, corporal
                (200, 50, 501, 2, 0),  # blue, recruit
                (134, 121, 1229, 3, 0),  # purple, recruit (our bot)
            ],
        )
        dispatch_world_state_update(get_world_service(), self._make_msg(blob))

        state = get_world_service().world_state
        assert "500" in state["tanks"]
        assert state["tanks"]["500"]["x"] == 100
        assert state["tanks"]["500"]["y"] == 120
        assert state["tanks"]["500"]["team"] == 1
        assert state["tanks"]["500"]["rank"] == 2

        assert "501" in state["tanks"]
        assert state["tanks"]["501"]["x"] == 200
        assert state["tanks"]["501"]["y"] == 50
        assert state["tanks"]["501"]["team"] == 2

        assert "1229" in state["tanks"]
        assert state["tanks"]["1229"]["x"] == 134
        assert state["tanks"]["1229"]["y"] == 121
        assert state["tanks"]["1229"]["team"] == 3

    def test_preserves_existing_tank_names(self) -> None:
        """Blob update preserves existing name and is_bot fields."""
        from tankpit_bot.sniffer.world_state_tanks import update_world_state_from_tank_info

        update_world_state_from_tank_info(get_world_service(), 500, team=1, name="EnemyBot")

        blob = self._build_world_state_blob(
            dot_section=b"\xff" * 5,
            tank_entries=[(150, 80, 500, 1, 3)],
        )
        dispatch_world_state_update(get_world_service(), self._make_msg(blob))

        state = get_world_service().world_state
        assert state["tanks"]["500"]["name"] == "EnemyBot"
        assert state["tanks"]["500"]["x"] == 150
        assert state["tanks"]["500"]["y"] == 80

    def test_blob_with_nontrivial_dot_section(self) -> None:
        """Blob with non-255 dot bytes emits dots into the cursor stream.

        2026-06-19: all production traffic now routes via tunneled 0x4C
        MapData; this synthetic test pins the dot-emission branch in
        _decode_fuel_dot_layer until the legacy world_state blob path
        is removed.
        """
        blob = self._build_world_state_blob(
            dot_section=bytes([1, 2, 3]),
            tank_entries=[(50, 60, 700, 0, 0)],
        )
        dispatch_world_state_update(get_world_service(), self._make_msg(blob))
        state = get_world_service().world_state
        assert "700" in state["tanks"]

    def test_empty_blob_no_crash(self) -> None:
        """Too-short blob is handled gracefully."""
        dispatch_world_state_update(get_world_service(), self._make_msg(b"\x00"))

        state = get_world_service().world_state
        assert len(state["tanks"]) == 0

    def test_zero_tanks_after_terrain(self) -> None:
        """Blob with terrain but no tank entries is handled gracefully."""
        blob = self._build_world_state_blob(dot_section=b"\xff" * 50, tank_entries=[])
        dispatch_world_state_update(get_world_service(), self._make_msg(blob))

        state = get_world_service().world_state
        assert len(state["tanks"]) == 0

    def test_large_terrain_count(self) -> None:
        """Blob with large terrain (694 bytes) + tanks parses correctly."""
        blob = self._build_world_state_blob(
            dot_section=b"\xff" * 694,
            tank_entries=[
                (9, 33, 504, 0, 1),  # red, private
                (134, 121, 1229, 3, 0),  # purple, recruit
            ],
        )
        dispatch_world_state_update(get_world_service(), self._make_msg(blob))

        state = get_world_service().world_state
        assert "504" in state["tanks"]
        assert state["tanks"]["504"]["x"] == 9
        assert state["tanks"]["504"]["y"] == 33
        assert state["tanks"]["504"]["team"] == 0
        assert state["tanks"]["504"]["rank"] == 1

    def test_terrain_count_exceeds_blob_length(self) -> None:
        """Blob with terrain_count larger than data is handled gracefully."""
        # terrain_count=1000 but blob only has 10 bytes total
        data = (1000).to_bytes(2, "little") + b"\x00" * 8
        dispatch_world_state_update(get_world_service(), self._make_msg(bytes(data)))

        state = get_world_service().world_state
        assert len(state["tanks"]) == 0
