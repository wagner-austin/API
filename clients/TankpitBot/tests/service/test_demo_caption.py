"""The demo's caption vocabulary and the fold that times it.

The vocabulary is held to the bot's whole reason enum, so a reason the
bot learns later cannot reach a stranger's screen as a code word. The
fold is driven with real decoded event records, the shape the fleet
reads off a bot's events artifact.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from platform_core.json_utils import JSONTypeError, load_json_str, narrow_json_to_dict

from tankpit_bot.bot.ai.scoring_types import ReasonKind
from tankpit_bot.runtime_records import RuntimeEventRecordDict, decode_runtime_event_record
from tankpit_bot.service.demo_caption import (
    CAPTION_HISTORY_LIMIT,
    CAPTION_WINDOW_MS,
    CaptionAccumulator,
    CaptionDict,
    encode_caption,
    event_time_ms,
)
from tankpit_bot.service.demo_caption_words import DESTROYED_WORDS, REASON_WORDS
from tankpit_bot.service.fuel_line import fuel_total, is_fuel_line


def _record(line: str) -> RuntimeEventRecordDict:
    """Decode one JSONL event line the way the fleet's reader does.

    Args:
        line: One events-artifact line.

    Returns:
        The decoded record.
    """
    return decode_runtime_event_record(narrow_json_to_dict(load_json_str(line)))


def _decision(second: int, reason: str) -> RuntimeEventRecordDict:
    """An executor decision event carrying a typed reason.

    Args:
        second: Seconds past 20:00:00 on the test day.
        reason: The ``behavior_reason_kind`` word.

    Returns:
        The decoded record.
    """
    return _record(
        f'{{"timestamp":"2026-10-05T20:00:{second:02d}","level":"INFO","logger":"l",'
        f'"mode":"bot","channel":"AI","message":"HUNT score=800",'
        f'"behavior_reason_kind":"{reason}","tick_n":{second}}}'
    )


def _fuel(second: int, total: int) -> RuntimeEventRecordDict:
    """A WORLD fuel-change event.

    Args:
        second: Seconds past 20:00:00 on the test day.
        total: The new fuel total.

    Returns:
        The decoded record.
    """
    return _record(
        f'{{"timestamp":"2026-10-05T20:00:{second:02d}","level":"INFO","logger":"l",'
        f'"mode":"bot","channel":"WORLD","message":"Fuel: 900 -> {total} (-45)"}}'
    )


def _death(second: int) -> RuntimeEventRecordDict:
    """The bot's own death receipt.

    Args:
        second: Seconds past 20:00:00 on the test day.

    Returns:
        The decoded record.
    """
    return _record(
        f'{{"timestamp":"2026-10-05T20:00:{second:02d}","level":"INFO","logger":"l",'
        f'"mode":"bot","channel":"DIAGNOSTIC","message":"diagnostic_kind=self_deactivated",'
        f'"diagnostic_kind":"self_deactivated"}}'
    )


def _at(second: int) -> int:
    """Epoch milliseconds of a test-day second, read as the fold reads it.

    Args:
        second: Seconds past 20:00:00 on the test day.

    Returns:
        Epoch milliseconds.
    """
    return int(datetime(2026, 10, 5, 20, 0, second).timestamp() * 1000)


class TestVocabulary:
    """Every reason has words, and the words read as captions."""

    def test_every_reason_the_bot_can_give_has_words(self) -> None:
        """The table is the enum, member for member."""
        assert set(REASON_WORDS) == set(ReasonKind)

    def test_every_caption_is_a_short_phrase_and_a_sentence(self) -> None:
        """No code words: a phrase, then one sentence ending in a stop."""
        for reason, words in [*REASON_WORDS.items(), (None, DESTROYED_WORDS)]:
            assert words["doing"][0].isupper(), reason
            assert "_" not in words["doing"] + words["why"], reason
            assert len(words["doing"]) <= 32, reason
            assert words["why"].endswith("."), reason


class TestFuelLine:
    """The shared fuel-line parser."""

    def test_a_fuel_line_yields_its_post_arrow_total(self) -> None:
        """``Fuel: X -> Y (d)`` reads as Y."""
        assert is_fuel_line("Fuel: 1100 -> 1055 (-45)")
        assert fuel_total("Fuel: 1100 -> 1055 (-45)") == 1055

    def test_a_line_without_a_plain_total_reads_as_unknown(self) -> None:
        """A total that is not a number is -1, never a guess."""
        assert fuel_total("Fuel: 1100 -> ?") == -1

    def test_other_lines_are_not_fuel_lines(self) -> None:
        """Only the prefix makes a fuel line."""
        assert not is_fuel_line("Fuelish: 3")


class TestFold:
    """Captions from decisions, deaths and fuel, timed and deduplicated."""

    def test_event_time_reads_the_bots_local_wall_clock(self) -> None:
        """A zone-less timestamp is read in the folding process's zone."""
        assert event_time_ms("2026-10-05T20:00:07") == _at(7)

    def test_nothing_is_said_before_the_first_decision(self) -> None:
        """Fuel alone says nothing: there is no behaviour to caption yet."""
        fold = CaptionAccumulator()
        fold.absorb([_fuel(1, 800)])
        assert fold.window(_at(2)) == []

    def test_a_decision_becomes_a_caption_with_the_fuel_known_then(self) -> None:
        """The reason's words, stamped with the event's own moment."""
        fold = CaptionAccumulator()
        fold.absorb([_fuel(1, 800), _decision(2, "teleport_target")])

        words = REASON_WORDS[ReasonKind.TELEPORT_TARGET]
        assert fold.window(_at(3)) == [
            CaptionDict(at_ms=_at(2), doing=words["doing"], why=words["why"], fuel=800)
        ]

    def test_a_repeated_decision_adds_nothing(self) -> None:
        """The same words at the same fuel are one caption, not one per tick."""
        fold = CaptionAccumulator()
        fold.absorb([_decision(2, "shoot_target"), _decision(4, "shoot_target")])
        assert len(fold.window(_at(5))) == 1

    def test_fuel_changes_and_deaths_are_captions_of_their_own(self) -> None:
        """A fuel change re-states the behaviour; a death replaces it."""
        fold = CaptionAccumulator()
        fold.absorb([_decision(2, "walk_for_fuel"), _fuel(3, 640), _death(4)])

        captions = fold.window(_at(5))
        walk = REASON_WORDS[ReasonKind.WALK_FOR_FUEL]
        assert [(c["at_ms"], c["doing"], c["fuel"]) for c in captions] == [
            (_at(2), walk["doing"], -1),
            (_at(3), walk["doing"], 640),
            (_at(4), DESTROYED_WORDS["doing"], 640),
        ]

    def test_a_reason_outside_the_vocabulary_is_refused(self) -> None:
        """A release mismatch surfaces; it never reaches the page as a word."""
        fold = CaptionAccumulator()
        with pytest.raises(JSONTypeError, match="behavior_reason_kind"):
            fold.absorb([_decision(2, "invented_reason")])

    def test_the_window_keeps_the_caption_in_force_at_its_start(self) -> None:
        """Old captions drop out, but the one still true is kept."""
        fold = CaptionAccumulator()
        fold.absorb(
            [
                _decision(1, "find_enemies"),
                _decision(2, "teleport_target"),
                _decision(40, "shoot_target"),
            ]
        )

        captions = fold.window(_at(41))
        assert _at(41) - CAPTION_WINDOW_MS > _at(2)
        assert [c["at_ms"] for c in captions] == [_at(2), _at(40)]

    def test_the_history_is_bounded_by_count(self) -> None:
        """A long run holds at most the limit, newest kept."""
        fold = CaptionAccumulator()
        fold.absorb([_fuel(second % 60, second) for second in range(CAPTION_HISTORY_LIMIT)])
        fold.absorb([_decision(0, "find_enemies")])
        fold.absorb([_fuel(30, 5000 + index) for index in range(CAPTION_HISTORY_LIMIT + 10)])

        captions = fold.window(_at(30))
        assert len(captions) == CAPTION_HISTORY_LIMIT
        assert captions[-1]["fuel"] == 5000 + CAPTION_HISTORY_LIMIT + 9

    def test_a_caption_encodes_every_field(self) -> None:
        """The wire shape is the whole caption."""
        caption = CaptionDict(at_ms=5, doing="Waiting", why="Because.", fuel=7)
        assert encode_caption(caption) == {
            "at_ms": 5,
            "doing": "Waiting",
            "why": "Because.",
            "fuel": 7,
        }
