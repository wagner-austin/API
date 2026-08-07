"""Per-instance telemetry for the fleet surface: stats and live activity.

Two read-only summaries, both computed from the instance's events
artifact (``runs/bot/<instance>/latest.events.jsonl``) and both cached
for :data:`TELEMETRY_CACHE_TTL_MS`:

* **stats** — the run-digest reduction (kills, deaths, rank countdown,
  inventory, five-minute timeline) the control page shows as numbers.
* **activity** — the live tail: current bot state, current fuel, and
  the last few AI/WORLD/STATE lines, so the page can show what the
  bot is DOING right now, not just its totals.

The cache exists because the page polls every second while a digest
parse re-reads the whole events file: capping recomputation at one
parse per :data:`TELEMETRY_CACHE_TTL_MS` per instance makes the poll
rate a UI choice instead of an IO amplifier.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.run_digest import build_run_digest
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.runtime_logging import RuntimeEventRecordDict

TELEMETRY_CACHE_TTL_MS = 2000
"""Maximum age of a cached summary before the events file is re-read."""

_FEED_CHANNELS = ("AI", "WORLD", "STATE", "COMBAT")
_FEED_LENGTH = 6
_FUEL_PREFIX = "Fuel: "


def _last_fuel_total(records: list[RuntimeEventRecordDict]) -> int:
    """Return the newest ``Fuel: X -> Y`` total, or ``-1`` when unseen.

    Args:
        records: Event records in file order.

    Returns:
        The last fuel total, or ``-1``.
    """
    for record in reversed(records):
        message = record["message"]
        if message.startswith(_FUEL_PREFIX):
            tail = message[len(_FUEL_PREFIX) :].split("->")[-1].strip()
            total = tail.split(" ")[0].split("(")[0].strip()
            return int(total) if total.isdigit() else -1
    return -1


def _feed_lines(records: list[RuntimeEventRecordDict]) -> list[JSONValue]:
    """Return the last feed-channel lines, oldest first.

    Args:
        records: Event records in file order.

    Returns:
        Up to :data:`_FEED_LENGTH` rendered lines.
    """
    feed: list[JSONValue] = []
    for record in reversed(records):
        if record["channel"] not in _FEED_CHANNELS:
            continue
        line: JSONObject = {
            "time": record["timestamp"].split("T")[-1],
            "channel": record["channel"],
            "message": record["message"].splitlines()[0][:120],
        }
        feed.append(line)
        if len(feed) >= _FEED_LENGTH:
            break
    feed.reverse()
    return feed


def _last_state_and_tick(records: list[RuntimeEventRecordDict]) -> tuple[str, int]:
    """Return the newest bot state and tick number the stream carries.

    Args:
        records: Event records in file order.

    Returns:
        ``(state, tick)`` — empty string / ``-1`` when never stated.
    """
    state = ""
    tick = -1
    for record in reversed(records):
        fields = record["fields"]
        state_value = fields.get("bot_state")
        if isinstance(state_value, str) and not state:
            state = state_value
        tick_value = fields.get("tick_n")
        if isinstance(tick_value, int) and tick < 0:
            tick = tick_value
        if state and tick >= 0:
            break
    return state, tick


class FleetTelemetry:
    """Cached stats/activity summaries for fleet instances."""

    def __init__(self) -> None:
        """Start with an empty cache."""
        self._cache: dict[tuple[str, str], tuple[int, JSONObject]] = {}

    def _fresh(self, kind: str, instance: str) -> JSONObject | None:
        """Return the cached summary when it is still young enough.

        Args:
            kind: ``"stats"`` or ``"activity"``.
            instance: Instance name.

        Returns:
            The cached payload, or ``None`` when absent or stale.
        """
        entry = self._cache.get((kind, instance))
        if entry is None:
            return None
        computed_ms, payload = entry
        if top_hooks.get_current_time_ms() - computed_ms > TELEMETRY_CACHE_TTL_MS:
            return None
        return payload

    def _store(self, kind: str, instance: str, payload: JSONObject) -> JSONObject:
        """Cache and return one summary.

        Args:
            kind: ``"stats"`` or ``"activity"``.
            instance: Instance name.
            payload: The freshly computed summary.

        Returns:
            The same payload.
        """
        self._cache[(kind, instance)] = (top_hooks.get_current_time_ms(), payload)
        return payload

    def stats(self, instance: str) -> JSONObject:
        """Summarize the instance's latest run from its events.

        Args:
            instance: Instance name (registry membership is the
                caller's concern).

        Returns:
            ``{"available": False}`` when no events exist yet, else
            the digest reduction with ``"available": True``.
        """
        cached = self._fresh("stats", instance)
        if cached is not None:
            return cached
        events_path = bot_run_dir(instance) / "latest.events.jsonl"
        try:
            digest = build_run_digest(events_path)
        except (OSError, ValueError, JSONTypeError):
            return self._store("stats", instance, {"available": False})
        timeline_kills: list[JSONValue] = [row["kills"] for row in digest["timeline"]]
        inventory_last: list[JSONValue] = list(digest["inventory_last"])
        return self._store(
            "stats",
            instance,
            {
                "available": True,
                "kills": digest["kills"],
                "deaths": digest["deaths"],
                "shots": digest["shots"],
                "teleports": digest["teleports"],
                "pickups": digest["pickups"],
                "displacements": digest["displacements"],
                "duration_s": digest["duration_s"],
                "clean_exit": digest["clean_exit"],
                "exit_reason": digest["exit_reason"],
                "rank_name": digest["rank_name"],
                "rank_number": digest["rank_number"],
                "promotion_points": digest["promotion_points"],
                "started_at": digest["started_at"],
                "inventory_last": inventory_last,
                "timeline_kills": timeline_kills,
            },
        )

    def activity(self, instance: str) -> JSONObject:
        """Return the live tail of the instance's events stream.

        Args:
            instance: Instance name (registry membership is the
                caller's concern).

        Returns:
            ``{"available": False}`` when no events exist yet, else
            the current bot state, the last seen fuel total, and the
            last :data:`_FEED_LENGTH` feed lines, oldest first.
        """
        cached = self._fresh("activity", instance)
        if cached is not None:
            return cached
        events_path = bot_run_dir(instance) / "latest.events.jsonl"
        try:
            records = load_event_records(events_path)
        except (OSError, JSONTypeError):
            return self._store("activity", instance, {"available": False})
        if not records:
            return self._store("activity", instance, {"available": False})
        state, tick = _last_state_and_tick(records)
        return self._store(
            "activity",
            instance,
            {
                "available": True,
                "state": state,
                "tick": tick,
                "fuel": _last_fuel_total(records),
                "feed": _feed_lines(records),
            },
        )


__all__ = [
    "TELEMETRY_CACHE_TTL_MS",
    "FleetTelemetry",
]
