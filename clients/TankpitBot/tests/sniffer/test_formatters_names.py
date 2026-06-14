"""Tests for sniffer name formatter functions (rank, team, damage)."""

from __future__ import annotations

from tankpit_bot.capture import format_sig_key
from tankpit_bot.sniffer import damage_name, rank_name, team_name

# =============================================================================
# Format Sig Key Tests
# =============================================================================


class TestFormatSigKey:
    """Tests for format_sig_key function."""

    def test_printable_ascii(self) -> None:
        """Test printable ASCII characters are shown."""
        result = format_sig_key(0x41)
        assert result == "0x41 'A'"

    def test_non_printable(self) -> None:
        """Test non-printable characters show question mark."""
        result = format_sig_key(0x01)
        assert result == "0x01 '?'"


# =============================================================================
# Rank Name Tests
# =============================================================================


class TestRankName:
    """Tests for rank_name function."""

    def test_known_ranks(self) -> None:
        """Test known rank values return correct names."""
        assert rank_name(0) == "recruit"
        assert rank_name(1) == "private"
        assert rank_name(2) == "corporal"
        assert rank_name(3) == "sergeant"
        assert rank_name(4) == "lieutenant"
        assert rank_name(5) == "captain"
        assert rank_name(6) == "major"
        assert rank_name(7) == "colonel"
        assert rank_name(8) == "general"

    def test_unknown_rank(self) -> None:
        """Test unknown rank values return formatted string."""
        assert rank_name(9) == "r9"
        assert rank_name(99) == "r99"

    def test_negative_rank(self) -> None:
        """Test negative rank values return formatted string."""
        assert rank_name(-1) == "r-1"


# =============================================================================
# Damage Name Tests
# =============================================================================


class TestDamageName:
    """Tests for damage_name function."""

    def test_known_damage_states(self) -> None:
        """The tier counts DOWN toward deactivation.

        Live run 20260610-231x: every fight ran 0 -> 3 -> 2 -> 1 under
        sustained fire and all five kills with tier data died from
        tier 1, so 1 is critical and 3 is light.
        """
        assert damage_name(0) == "full"
        assert damage_name(1) == "critical"
        assert damage_name(2) == "medium"
        assert damage_name(3) == "light"

    def test_unknown_damage(self) -> None:
        """Test unknown damage values return formatted string."""
        assert damage_name(4) == "d4"
        assert damage_name(99) == "d99"


# =============================================================================
# Team Name Tests
# =============================================================================


class TestTeamName:
    """Tests for team_name function."""

    def test_known_teams(self) -> None:
        """Test known team values return correct names."""
        assert team_name(0) == "red"
        assert team_name(1) == "purple"
        assert team_name(2) == "blue"
        assert team_name(3) == "orange"

    def test_unknown_team(self) -> None:
        """Test unknown team values return formatted string."""
        assert team_name(4) == "t4"
        assert team_name(99) == "t99"
