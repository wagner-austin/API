"""Tests for the CombatFeedback vocabulary."""

from __future__ import annotations

from tankpit_bot.bot.combat_feedback import CombatFeedback


class TestCombatFeedbackVocabulary:
    """The shot-feedback vocabulary the tick loop hands the strategy."""

    def test_members_are_the_four_feedback_words(self) -> None:
        """HIT, MISS, REJECTED and NONE carry the words the feedback reader returns."""
        assert [(member.name, member.value) for member in CombatFeedback] == [
            ("HIT", "hit"),
            ("MISS", "miss"),
            ("REJECTED", "rejected"),
            ("NONE", ""),
        ]

    def test_none_is_the_only_empty_member(self) -> None:
        """NONE is the indeterminate state and the only member that reads as empty."""
        assert [member for member in CombatFeedback if not member.value] == [CombatFeedback.NONE]
