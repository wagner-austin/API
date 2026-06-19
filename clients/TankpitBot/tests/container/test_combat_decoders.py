"""Tests for combat-related container decoders.

Only deactivation_death (0x43 subtype) remains in the container path.
  * 0x53 ShootEvent -> tankpit_bot.protocol.decode_shoot_event
  * 0x41 Deactivation -> tankpit_bot.protocol.decode_deactivation
Both moved to the protocol layer in the 2026-06-19 dual-path collapse.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    DeactivationDeathDict,
    decode_deactivation_death,
)
from tests.container.test_data import DEACTIVATION_DEATH_7


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
