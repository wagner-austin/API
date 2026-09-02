"""Per-instance telemetry for the fleet surface: stats and live activity.

Two read-only summaries, both folded from the instance's events
artifact (``runs/bot/<instance>/latest.events.jsonl``) by
:class:`~tankpit_bot.service.fleet_stream.InstanceStream`:

* **stats** — the run-digest reduction (kills, deaths, rank countdown,
  inventory, five-minute timeline) the control page shows as numbers.
* **activity** — the live tail: current bot state, current fuel, and
  the last few AI/WORLD/STATE lines, so the page can show what the
  bot is DOING right now, not just its totals.

Two layers of work-avoidance, and they are not the same thing:

* The **stream** is incremental, so a refresh costs the bytes the bot
  appended since the previous refresh rather than the whole run. This
  is what stopped a five-bot fleet from re-parsing ~65 MB every two
  seconds (2026-09-01).
* The **cache** bounds how often a poll can trigger a refresh at all
  (:data:`TELEMETRY_CACHE_TTL_MS`), so the page's poll rate stays a UI
  choice rather than an IO multiplier.

Both summaries share one cursor: the page asks for stats and activity
together, so whichever lands second finds nothing new to fold.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.fleet_stream import InstanceStream

TELEMETRY_CACHE_TTL_MS = 2000
"""Maximum age of a cached summary before the artifact is re-read."""

log = get_logger(__name__)

_UNAVAILABLE: JSONObject = {"available": False}


class FleetTelemetry:
    """Cached stats/activity summaries for fleet instances."""

    def __init__(self) -> None:
        """Start with no streams and an empty cache."""
        self._streams: dict[str, InstanceStream] = {}
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

    def _refreshed_stream(self, instance: str) -> InstanceStream | None:
        """Return the instance's stream, advanced to the artifact's end.

        The stream is created on first use and kept afterwards: its
        value IS its position, so discarding it would put the fleet
        back to whole-file reads.

        Args:
            instance: Instance name.

        Returns:
            The refreshed stream, or ``None`` when the instance has no
            readable events yet — a bot mid-boot has no artifact, and
            a run whose first line is still being written has no
            complete record.
        """
        stream = self._streams.get(instance)
        if stream is None:
            stream = InstanceStream(instance)
            self._streams[instance] = stream
        try:
            stream.refresh()
        except (OSError, ValueError, JSONTypeError) as error:
            # The artifact boundary: absent (bot still booting) or
            # unreadable (a line the strict decoder rejects). Both are
            # reported to the page as "no summary yet" rather than
            # failing the poll, and both are logged so an artifact
            # that never becomes readable is visible here.
            log.debug("Fleet telemetry unavailable for %s: %s", instance, error)
            return None
        if stream.record_count == 0:
            return None
        return stream

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
        stream = self._refreshed_stream(instance)
        if stream is None:
            return self._store("stats", instance, dict(_UNAVAILABLE))
        digest = stream.digest()
        timeline_kills: list[JSONValue] = [row["kills"] for row in digest["timeline"]]
        inventory_last: list[JSONValue] = list(digest["inventory_last"])
        inventory_first: list[JSONValue] = list(digest["inventory_first"])
        return self._store(
            "stats",
            instance,
            {
                "available": True,
                "kills": digest["kills"],
                "deaths": digest["deaths"],
                "shots": digest["shots"],
                "hits": digest["hits"],
                "misses": digest["misses"],
                "zero_yield_radars": digest["zero_yield_radars"],
                "damage_dealt": digest["damage_dealt"],
                "damage_taken": digest["damage_taken"],
                "teleports": digest["teleports"],
                "pickups": digest["pickups"],
                "displacements": digest["displacements"],
                "duration_s": digest["duration_s"],
                "clean_exit": digest["clean_exit"],
                "exit_reason": digest["exit_reason"],
                "rank_name": digest["rank_name"],
                "leaderboard_position": digest["leaderboard_position"],
                "promotion_points": digest["promotion_points"],
                "started_at": digest["started_at"],
                "inventory_first": inventory_first,
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
            last few feed lines, oldest first.
        """
        cached = self._fresh("activity", instance)
        if cached is not None:
            return cached
        stream = self._refreshed_stream(instance)
        if stream is None:
            return self._store("activity", instance, dict(_UNAVAILABLE))
        return self._store("activity", instance, stream.activity())

    def forget(self, instance: str) -> None:
        """Drop an instance's stream and cached summaries.

        Called when the fleet removes an instance from its registry,
        so a long-lived manager does not hold a cursor (and a folded
        digest) for a bot nobody is watching any more.

        Args:
            instance: Instance name to forget.

        Returns:
            None.
        """
        self._streams.pop(instance, None)
        for kind in ("stats", "activity"):
            self._cache.pop((kind, instance), None)


__all__ = [
    "TELEMETRY_CACHE_TTL_MS",
    "FleetTelemetry",
]
