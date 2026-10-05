"""One bot instance's events artifact, folded as it grows.

The fleet control page wants two summaries per bot every second: the
whole-run digest (kills, damage, inventory arc) and the live tail
(current state, fuel, last few log lines), and the public demo page
wants a third, its captions (:mod:`tankpit_bot.service.demo_caption`).
All are reductions over the same event stream, so this module reads
that stream ONCE per refresh and feeds every fold from it.

Before 2026-09-01 each summary re-read and re-decoded the entire
artifact on every cache miss. That is fine for a finished run and
ruinous for a live one: six minutes of play is a 13 MB file, and the
page was re-parsing all of it, twice, per bot, every two seconds --
which is why reconnecting to a running fleet took so long. The bytes
a poll actually needs are the few hundred appended since the last
poll, and that is now all it reads.
"""

from __future__ import annotations

from collections import deque

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from tankpit_bot.diagnostics.event_tail import EventTail
from tankpit_bot.diagnostics.run_digest_fold import RunDigestAccumulator
from tankpit_bot.diagnostics.run_digest_types import RunDigestDict
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.runtime_records import RuntimeEventRecordDict
from tankpit_bot.service.demo_caption import CaptionAccumulator, CaptionDict
from tankpit_bot.service.fuel_line import fuel_total, is_fuel_line

#: Channels whose lines the page's activity feed shows.
FEED_CHANNELS = ("AI", "WORLD", "STATE", "COMBAT")

#: How many feed lines the page shows, oldest first.
FEED_LENGTH = 6


class ActivityAccumulator:
    """The live tail -- current state, fuel, and recent lines.

    Folded forward rather than scanned backwards. The result is the
    same values the old reverse scan produced (the LAST state, tick
    and fuel the stream carries, and the last :data:`FEED_LENGTH`
    feed-channel lines), reached without holding the run in memory.
    """

    def __init__(self) -> None:
        """Start with nothing observed."""
        self._state = ""
        self._tick = -1
        self._fuel = -1
        self._feed: deque[JSONObject] = deque(maxlen=FEED_LENGTH)

    def absorb(self, records: list[RuntimeEventRecordDict]) -> None:
        """Fold more records into the tail state, in file order.

        Args:
            records: The next records of the same run, oldest first.

        Returns:
            None.
        """
        for record in records:
            state_value = record["fields"].get("bot_state")
            if isinstance(state_value, str):
                self._state = state_value
            tick_value = record["fields"].get("tick_n")
            if isinstance(tick_value, int):
                self._tick = tick_value
            message = record["message"]
            if is_fuel_line(message):
                self._fuel = fuel_total(message)
            if record["channel"] in FEED_CHANNELS:
                self._feed.append(
                    {
                        "time": record["timestamp"].split("T")[-1],
                        "channel": record["channel"],
                        "message": message.splitlines()[0][:120],
                    }
                )

    def snapshot(self) -> JSONObject:
        """Return the tail as the control page consumes it.

        Returns:
            The current state, tick, fuel total, and feed lines
            oldest first.
        """
        feed: list[JSONValue] = list(self._feed)
        return {
            "available": True,
            "state": self._state,
            "tick": self._tick,
            "fuel": self._fuel,
            "feed": feed,
        }


class InstanceStream:
    """A single bot instance's artifact, read forward and folded three ways.

    Owns the read cursor and the three accumulators (the digest, the
    activity tail, and the public demo's captions) so a caller
    refreshes once and then snapshots any summary for free. A new run
    under the same path resets every fold -- the cursor reports the
    restart and this class is what acts on it.
    """

    def __init__(self, instance: str) -> None:
        """Bind a stream to one instance's events artifact.

        Args:
            instance: Instance name whose ``runs/bot/<instance>``
                namespace holds the artifact.
        """
        self._path = bot_run_dir(instance) / "latest.events.jsonl"
        self._tail = EventTail(self._path)
        self._digest = RunDigestAccumulator(str(self._path))
        self._activity = ActivityAccumulator()
        self._captions = CaptionAccumulator()
        self._record_count = 0
        self._spoiled = ""

    @property
    def record_count(self) -> int:
        """Return how many records of the current run have been folded.

        Returns:
            Zero until the run's first event lands, which is what
            makes a summary "not available yet" rather than empty.
        """
        return self._record_count

    def refresh(self) -> None:
        """Read whatever the bot has appended and fold it into all three.

        A fold that fails part-way SPOILS the stream for the rest of
        the run. The records it managed to absorb are already in the
        accumulators and the read cursor has already moved past them,
        so continuing would quietly report a digest with a hole in it
        -- numbers that look plausible and are wrong. Refusing every
        later refresh instead keeps the artifact's own contract: a
        record the strict decoder rejects is surfaced, never skipped.
        A new run under the same path clears the spoil, because a
        restart resets the folds anyway.

        Returns:
            None.

        Raises:
            OSError: If the artifact does not exist -- the instance
                has not written its first event yet.
            JSONTypeError: When a complete line fails strict event
                decoding, or a decision names a reason outside the
                bot's vocabulary (which also spoils the run).
            ValueError: When a decoded record cannot be folded (a
                malformed field or timestamp), and on every later
                refresh of the same spoiled run.
        """
        records, restarted = self._tail.next_records()
        if restarted:
            self._digest = RunDigestAccumulator(str(self._path))
            self._activity = ActivityAccumulator()
            self._captions = CaptionAccumulator()
            self._record_count = 0
            self._spoiled = ""
        if self._spoiled:
            raise ValueError(f"{self._path} stopped folding at: {self._spoiled}")
        try:
            self._digest.absorb(records)
            self._activity.absorb(records)
            self._captions.absorb(records)
        except (ValueError, JSONTypeError) as error:
            # Recorded and re-raised, never softened: the flag exists
            # so the NEXT poll fails the same way instead of serving
            # a half-folded run as if it were whole.
            self._spoiled = str(error)
            raise
        self._record_count += len(records)

    def digest(self) -> RunDigestDict:
        """Return the whole-run digest as of the last refresh.

        Returns:
            An independent digest snapshot.
        """
        return self._digest.snapshot()

    def activity(self) -> JSONObject:
        """Return the live tail as of the last refresh.

        Returns:
            The activity payload the control page renders.
        """
        return self._activity.snapshot()

    def captions(self, now_ms: int) -> list[CaptionDict]:
        """Return the public demo's caption window as of the last refresh.

        Args:
            now_ms: The current wall-clock epoch milliseconds.

        Returns:
            The captions a viewer may still be watching, oldest first.
        """
        return self._captions.window(now_ms)


__all__ = [
    "FEED_CHANNELS",
    "FEED_LENGTH",
    "ActivityAccumulator",
    "InstanceStream",
]
