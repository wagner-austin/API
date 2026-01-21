"""Tests for combat-related container decoders.

Tests for combat hit, deactivation kill, and deactivation death decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    CombatHitDict,
    ContainerDecodeError,
    DeactivationDeathDict,
    DeactivationKillDict,
    decode_combat_hit,
    decode_deactivation_death,
    decode_deactivation_kill,
)
from tests.container.test_data import (
    COMBAT_HIT_11_INCOMING,
    COMBAT_HIT_11_OUTGOING,
    DEACTIVATION_DEATH_7,
    DEACTIVATION_KILL_5,
)


class TestDecodeCombatHit:
    """Tests for combat hit decoding."""

    def test_decodes_outgoing_hit(self) -> None:
        """Decodes outgoing combat hit correctly."""
        result = decode_combat_hit(COMBAT_HIT_11_OUTGOING)
        assert result["msg_type"] == "combat_hit"
        assert result["direction"] == 0x09
        assert result["attacker_id"] == 0x07CD  # cd 07 little-endian
        assert result["is_outgoing"] is True
        assert len(result["combat_data"]) == 7  # bytes 4-10

    def test_decodes_incoming_hit(self) -> None:
        """Decodes incoming combat hit correctly."""
        result = decode_combat_hit(COMBAT_HIT_11_INCOMING)
        assert result["msg_type"] == "combat_hit"
        assert result["direction"] == 0x0B
        assert result["attacker_id"] == 0x07CD  # cd 07 little-endian
        assert result["is_outgoing"] is False
        assert len(result["combat_data"]) == 7

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_combat_hit(bytes([0x01] * 10))
        with pytest.raises(ContainerDecodeError):
            decode_combat_hit(bytes([0x01] * 12))

    def test_combat_hit_dict_keys(self) -> None:
        """CombatHitDict has expected keys."""
        result: CombatHitDict = decode_combat_hit(COMBAT_HIT_11_OUTGOING)
        assert result["msg_type"] == "combat_hit"
        assert result["direction"] == 0x09
        assert result["attacker_id"] == 0x07CD
        assert len(result["combat_data"]) == 7
        assert result["is_outgoing"] is True


class TestDecodeDeactivationKill:
    """Tests for deactivation kill message decoding."""

    def test_decodes_deactivation_kill(self) -> None:
        """Correctly decodes deactivation kill message."""
        result = decode_deactivation_kill(DEACTIVATION_KILL_5)
        assert result["msg_type"] == "deactivation_kill"
        # victim_id = 0xBB | (0x62 << 8) = 187 | 25088 = 25275
        assert result["victim_id"] == 25275
        # killer_id = 0x9C | (0x0E << 8) = 156 | 3584 = 3740
        assert result["killer_id"] == 3740

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_kill(bytes([0x41] + [0x01] * 3))
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_kill(bytes([0x41] + [0x01] * 5))

    def test_deactivation_kill_dict_keys(self) -> None:
        """DeactivationKillDict has expected keys."""
        result: DeactivationKillDict = decode_deactivation_kill(DEACTIVATION_KILL_5)
        assert result["msg_type"] == "deactivation_kill"
        assert result["victim_id"] == 25275
        assert result["killer_id"] == 3740


class TestDecodeDeactivationDeath:
    """Tests for deactivation death message decoding."""

    def test_decodes_deactivation_death(self) -> None:
        """Correctly decodes deactivation death message."""
        result = decode_deactivation_death(DEACTIVATION_DEATH_7)
        assert result["msg_type"] == "deactivation_death"
        assert result["flags"] == 0x07
        # killer_id = 0x86 | (0x16 << 8) = 134 | 5632 = 5766
        assert result["killer_id"] == 5766
        assert result["extra_data"] == bytes.fromhex("0c7f1f")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_death(bytes([0x43] + [0x01] * 5))
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_death(bytes([0x43] + [0x01] * 7))

    def test_deactivation_death_dict_keys(self) -> None:
        """DeactivationDeathDict has expected keys."""
        result: DeactivationDeathDict = decode_deactivation_death(DEACTIVATION_DEATH_7)
        assert result["msg_type"] == "deactivation_death"
        assert result["flags"] == 0x07
        assert result["killer_id"] == 5766
        assert result["extra_data"] == bytes.fromhex("0c7f1f")
