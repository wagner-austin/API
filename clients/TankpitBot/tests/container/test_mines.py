"""Tests for tunneled mine container decoders.

Container TankRegistry / TankLeave / TankStatusShort / TankStatusSync /
TankUpdateCompact/Extended/Full were all deleted (last sweep
2026-06-20). The protocol path is the single source of truth. The
remaining tunneled decoders here are MinePlacement (0x4B) and
MineDetonation (0x45).
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    MineDetonationDict,
    MinePlacementDict,
)
from tankpit_bot.container.decoders.mines import (
    decode_mine_detonation,
    decode_mine_placement,
    is_mine_placement_structure,
)
from tests.container.test_data import (
    MINE_DETONATION_3,
    MINE_DETONATION_15,
    MINE_PLACEMENT_15,
    MINE_PLACEMENT_19,
)


class TestDecodeMinePlacement:
    """Tests for tunneled mine placement decoding."""

    def test_decodes_captured_5_position_placement(self) -> None:
        """Decodes captured 15-byte tunneled mine placement (count=5)."""
        result = decode_mine_placement(MINE_PLACEMENT_15)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 2
        assert result["tank_id"] == 1301
        assert result["positions"] == [
            (131, 126),
            (131, 125),
            (132, 125),
            (132, 126),
            (132, 127),
        ]

    def test_decodes_captured_7_position_placement(self) -> None:
        """Decodes 19-byte tunneled mine placement (count=7) from real
        combat capture practice-vs-real-20260620-150138 at 15:02:56.

        Regression: the prior decoder hardcoded 15 bytes and dropped
        this exact body to UnknownContainer.
        """
        result = decode_mine_placement(MINE_PLACEMENT_19)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 2
        assert result["tank_id"] == 1301
        assert result["positions"] == [
            (133, 124),
            (132, 124),
            (133, 123),
            (134, 123),
            (134, 124),
            (133, 125),
            (132, 125),
        ]

    def test_mine_placement_dict_keys(self) -> None:
        """MinePlacementDict has expected keys."""
        result: MinePlacementDict = decode_mine_placement(MINE_PLACEMENT_15)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 2
        assert result["tank_id"] == 1301
        assert len(result["positions"]) == 5

    def test_structure_accepts_variable_count(self) -> None:
        """Structure check accepts every count value whose body length matches."""
        assert is_mine_placement_structure(MINE_PLACEMENT_15) is True
        assert is_mine_placement_structure(MINE_PLACEMENT_19) is True
        # Count=1 -> 7 bytes
        body_count_1 = bytes.fromhex("4b02150501" + "857c")
        assert is_mine_placement_structure(body_count_1) is True

    def test_structure_rejects_count_mismatch(self) -> None:
        """Structure check rejects bodies whose length disagrees with count."""
        # Header says count=5 (15 bytes total) but body is truncated to 13 bytes.
        truncated = bytes.fromhex("4b02150505" + "857c847c8587")
        assert is_mine_placement_structure(truncated) is False

    def test_structure_rejects_wrong_subtype(self) -> None:
        """Structure check rejects bodies whose subtype byte isn't 0x4B."""
        # 15 bytes with subtype 0x99
        wrong_subtype = bytes.fromhex("99" + "02150505" + "837e837d847d847e847f")
        assert is_mine_placement_structure(wrong_subtype) is False

    def test_decoder_raises_on_wrong_subtype(self) -> None:
        """Decoder raises when subtype byte isn't 0x4B."""
        wrong_subtype = bytes.fromhex("99" + "02150505" + "837e837d847d847e847f")
        with pytest.raises(ContainerDecodeError, match="expected subtype 0x4B"):
            decode_mine_placement(wrong_subtype)

    def test_decoder_raises_on_count_length_mismatch(self) -> None:
        """Decoder raises when body length disagrees with count byte."""
        # Header says count=5 (need 15 bytes) but body is 14 bytes.
        truncated = bytes.fromhex("4b02150505" + "857c847c85878487")
        with pytest.raises(ContainerDecodeError, match="count=5 requires 15 bytes"):
            decode_mine_placement(truncated)

    def test_decoder_raises_on_too_short_header(self) -> None:
        """Decoder raises when body is shorter than the 5-byte header."""
        with pytest.raises(ContainerDecodeError, match="MinePlacement"):
            decode_mine_placement(bytes.fromhex("4b020102"))


class TestDecodeMineDetonation:
    """Tests for tunneled mine detonation decoding."""

    def test_decodes_solitary_mine_detonation(self) -> None:
        """Decodes 3-byte tunneled mine detonation correctly."""
        result = decode_mine_detonation(MINE_DETONATION_3)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [(44, 59)]

    def test_decodes_chain_reaction_mine_detonation(self) -> None:
        """Decodes 15-byte tunneled mine detonation correctly."""
        result = decode_mine_detonation(MINE_DETONATION_15)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [
            (38, 52),
            (39, 53),
            (38, 54),
            (37, 53),
            (39, 54),
            (39, 52),
            (37, 54),
        ]

    def test_mine_detonation_dict_keys(self) -> None:
        """MineDetonationDict has expected keys."""
        result: MineDetonationDict = decode_mine_detonation(MINE_DETONATION_3)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [(44, 59)]

    def test_raises_on_invalid_mine_detonation(self) -> None:
        """Raises on invalid tunneled mine detonation payload."""
        with pytest.raises(ContainerDecodeError):
            decode_mine_detonation(bytes.fromhex("442c3b"))
