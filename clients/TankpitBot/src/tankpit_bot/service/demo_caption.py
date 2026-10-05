"""The public demo's live captions, folded from the bot's own events.

A caption says what a demo bot is doing and why, in plain words
(:mod:`tankpit_bot.service.demo_caption_words`), for a viewer who does
not know the game. It is driven by the bot's own state, never read off
the picture: the reason each dispatched decision carries
(``behavior_reason_kind`` on the executor's AI event), the death
receipt (the ``self_deactivated`` diagnostic) and the fuel lines.

The fold keeps a short TIMED HISTORY rather than only the latest
caption, because the video is behind the bot. The stream reaches a
viewer a segment or more after the moment it shows, so the newest
caption is ahead of the picture by that much. Each caption therefore
carries the wall-clock moment it became true; the encoder stamps every
segment with its own wall-clock start (``program_date_time``, same
container, same clock), and the page shows the caption that was true
at the frame it is playing.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from platform_core.json_utils import JSONObject
from platform_core.members import as_member
from typing_extensions import TypedDict

from tankpit_bot.bot.ai.scoring_types import ReasonKind
from tankpit_bot.runtime_records import RuntimeEventRecordDict
from tankpit_bot.service.demo_caption_words import (
    DESTROYED_WORDS,
    REASON_WORDS,
    CaptionWordsDict,
)
from tankpit_bot.service.fuel_line import fuel_total, is_fuel_line

CAPTION_WINDOW_MS = 30_000
"""How far back the published history reaches, in milliseconds.

A viewer's picture trails the bot by the encoder's segment, the
player's buffer and the network, about ten seconds in practice; thirty
covers that three times over. The caption in force at the start of the
window is kept too, so a quiet bot still has one."""

CAPTION_HISTORY_LIMIT = 128
"""Most captions held per bot, a memory bound independent of time.

A caption is added only when what it says changes, at most a few per
two-second tick, so 128 spans well past :data:`CAPTION_WINDOW_MS`."""

REASON_FIELD = "behavior_reason_kind"
"""The AI event field carrying the decision's typed reason."""

DEATH_DIAGNOSTIC = "self_deactivated"
"""The diagnostic kind the bot logs once per death of its own tank."""


class CaptionDict(TypedDict):
    """One caption, from the moment it became true.

    Attributes:
        at_ms: Wall-clock epoch milliseconds of the event that made it
            true, on the bot container's clock.
        doing: What the tank is doing, a few words.
        why: Why, one plain sentence.
        fuel: The tank's fuel at that moment, ``-1`` before the first
            fuel line.
    """

    at_ms: int
    doing: str
    why: str
    fuel: int


def event_time_ms(timestamp: str) -> int:
    """Convert an event's timestamp to epoch milliseconds.

    Event timestamps are the bot's local wall clock with no zone
    (``runtime_logging_handlers``); this reads them in the local zone
    of the process folding them, which on the fleet is the same
    container as the bot.

    Args:
        timestamp: ``YYYY-MM-DDTHH:MM:SS`` from an event record.

    Returns:
        Epoch milliseconds.
    """
    return int(datetime.fromisoformat(timestamp).timestamp() * 1000)


class CaptionAccumulator:
    """Captions for one run, folded forward as the events grow."""

    def __init__(self) -> None:
        """Start with nothing said."""
        self._captions: deque[CaptionDict] = deque(maxlen=CAPTION_HISTORY_LIMIT)
        self._words: CaptionWordsDict | None = None
        self._fuel = -1

    def _say(self, at_ms: int) -> None:
        """Record the current words and fuel, unless nothing changed.

        Args:
            at_ms: When the change happened.
        """
        if self._words is None:
            return
        caption = CaptionDict(
            at_ms=at_ms,
            doing=self._words["doing"],
            why=self._words["why"],
            fuel=self._fuel,
        )
        if self._captions:
            last = self._captions[-1]
            if (last["doing"], last["why"], last["fuel"]) == (
                caption["doing"],
                caption["why"],
                caption["fuel"],
            ):
                return
        self._captions.append(caption)

    def absorb(self, records: list[RuntimeEventRecordDict]) -> None:
        """Fold more records of the same run, in file order.

        Args:
            records: The next records, oldest first.

        Raises:
            JSONTypeError: A decision names a reason outside the bot's
                vocabulary; the release that wrote the events and the
                one folding them disagree, which is a defect to see.
        """
        for record in records:
            reason = record["fields"].get(REASON_FIELD)
            if isinstance(reason, str):
                self._words = REASON_WORDS[as_member(reason, REASON_FIELD, ReasonKind)]
                self._say(event_time_ms(record["timestamp"]))
            elif record["fields"].get("diagnostic_kind") == DEATH_DIAGNOSTIC:
                self._words = DESTROYED_WORDS
                self._say(event_time_ms(record["timestamp"]))
            elif is_fuel_line(record["message"]):
                self._fuel = fuel_total(record["message"])
                self._say(event_time_ms(record["timestamp"]))

    def window(self, now_ms: int) -> list[CaptionDict]:
        """Return the captions a viewer may still be watching.

        Args:
            now_ms: The current wall-clock epoch milliseconds.

        Returns:
            Every caption newer than :data:`CAPTION_WINDOW_MS` ago,
            preceded by the one in force at the window's start; oldest
            first, and empty before the bot's first decision.
        """
        start_ms = now_ms - CAPTION_WINDOW_MS
        captions = list(self._captions)
        first = 0
        for index, caption in enumerate(captions):
            if caption["at_ms"] <= start_ms:
                first = index
        return captions[first:]


def encode_caption(caption: CaptionDict) -> JSONObject:
    """Encode one caption for the wire.

    Args:
        caption: The caption.

    Returns:
        JSON-serializable object.
    """
    return {
        "at_ms": caption["at_ms"],
        "doing": caption["doing"],
        "why": caption["why"],
        "fuel": caption["fuel"],
    }


__all__ = [
    "CAPTION_HISTORY_LIMIT",
    "CAPTION_WINDOW_MS",
    "DEATH_DIAGNOSTIC",
    "REASON_FIELD",
    "CaptionAccumulator",
    "CaptionDict",
    "encode_caption",
    "event_time_ms",
]
