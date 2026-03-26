"""Tests for CombatFeedback type."""

from __future__ import annotations

from tankpit_bot.bot.combat_feedback import CombatFeedback


class TestCombatFeedbackType:
    """Tests for CombatFeedback literal type."""

    def test_hit_is_valid(self) -> None:
        """'hit' is a valid CombatFeedback value."""
        feedback: CombatFeedback = "hit"
        assert feedback == "hit"

    def test_miss_is_valid(self) -> None:
        """'miss' is a valid CombatFeedback value."""
        feedback: CombatFeedback = "miss"
        assert feedback == "miss"

    def test_empty_is_valid(self) -> None:
        """Empty string is a valid CombatFeedback value."""
        feedback: CombatFeedback = ""
        assert feedback == ""
