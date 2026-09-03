"""Radar and map-data bookkeeping for one session's world service.

Scan-completion latches, the extra-radar flag, and the radar viewport
bounds. Mixed into :class:`~tankpit_bot.sniffer.world_service.WorldService`,
which owns the state these annotate.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.state import WorldStateDict
from tankpit_bot.state.viewport_geometry import (
    regular_radar_bounds,
    viewport_radar_bounds,
)

_RADAR_CACHE_REFRESH_WINDOW_MS = 2000
"""Pairing window between a zero-delta tunneled radar result and its
consumer. A zero-delta answer means the server's map cache already
matched ours; the mark is consumed by the NEXT decide pass (~2 s
cadence) to skip redundant refresh work. One replan cycle is the
whole lifetime the pairing needs — a wider window would let a stale
zero-delta mark suppress a refresh that a later, different scan
actually earned. The consume also zeroes the mark unconditionally,
so the window only ever spans one observation."""


class WorldServiceRadarMixin:
    """Radar and map-data bookkeeping for one session's world service.

    The attributes below are DECLARATIONS, not assignments: the
    session's ``__init__`` remains their single owner, so this split
    does not move any per-session state.
    """

    world_state: WorldStateDict
    radar_scan_complete: bool
    map_data_processed: bool
    map_data_ingested_ms: int
    viewport_update_processed: bool
    pending_radar_uses_extra: bool
    pending_radar_empty_delta_ms: int
    container_desync_ms: int
    mine_reveal_pending_ms: int

    def mark_radar_scan_complete(self) -> None:
        """Record that the server completed a radar scan.

        Also answers any pending container-desync latch: EVERY radar
        response shape lands here (full 0x4F delta, cache refresh,
        empty-delta resolution), and the ruling is one radar per
        desync. Session 5 of run 20260730 burned all 22 extra radars
        in a 2 s loop because the first latch clear lived only on the
        full-delta path while the server answered with cache
        refreshes.
        """
        self.radar_scan_complete = True
        self.container_desync_ms = 0
        # The same one-scan-per-latch rule covers the own-mine-hit
        # reveal: any radar response answers it.
        self.mine_reveal_pending_ms = 0

    def check_and_clear_radar_scan_complete(self) -> bool:
        """Check if a radar scan completed since last check, then clear.

        Returns:
            True if radar completion was observed.
        """
        result = self.radar_scan_complete
        self.radar_scan_complete = False
        return result

    def mark_map_data_processed(self) -> None:
        """Record that a MAP_DATA world-state blob was parsed into positions.

        Besides the completion flag, the INGESTION TIME is stamped:
        ``map_data_ingested_ms`` is what "the map snapshot is fresh"
        must mean. The hunt's no-viable-targets exit previously aged
        the snapshot from the map open's DISPATCH time — run
        bot-20260825-212920 ended on a phantom "fresh empty map"
        because the final open completed on an orphan flag while the
        dying wire delivered no data at all; dispatch recency said
        2 s, data recency said far beyond the cooldown.
        """
        self.map_data_processed = True
        self.map_data_ingested_ms = _test_hooks.get_current_time_ms()

    def mark_viewport_update_processed(self) -> None:
        """Record that a 0x5A ViewportUpdate was ingested.

        The scope action's completion signal ([[viewport-shift-protocol]]):
        every viewport-origin change arrives as a 0x5A, so the in-flight
        scope's wait gate reads this exactly as map_open reads
        ``map_data_processed``. Teleport landings also produce 0x5A —
        the scope dispatch clears any stale mark first, so the flag
        means "a 0x5A arrived since THIS pan was sent".
        """
        self.viewport_update_processed = True

    def check_and_clear_viewport_update_processed(self) -> bool:
        """Check if a 0x5A ViewportUpdate arrived since last check, then clear.

        Returns:
            True if a viewport update was ingested since the last call.
        """
        result = self.viewport_update_processed
        self.viewport_update_processed = False
        return result

    def check_and_clear_map_data_processed(self) -> bool:
        """Check if a MAP_DATA world-state blob was parsed since last check.

        Returns:
            True if a MAP_DATA payload was successfully ingested since the
            last call to this function.
        """
        result = self.map_data_processed
        self.map_data_processed = False
        return result

    def record_radar_command(self, *, use_extra_radar: bool) -> None:
        """Record which radar geometry the next server scan should use.

        Args:
            use_extra_radar: True for extra-radar viewport scans, False for
                the built-in 5x5 radar.
        """
        self.pending_radar_uses_extra = use_extra_radar

    def current_radar_uses_extra(self) -> bool:
        """Return True when the pending/current scan uses extra radar geometry."""
        return self.pending_radar_uses_extra

    def mark_pending_radar_empty_delta(self) -> None:
        """Record that a zero-delta tunneled radar result was observed."""
        self.pending_radar_empty_delta_ms = _test_hooks.get_current_time_ms()

    def consume_pending_radar_empty_delta(self) -> bool:
        """Return True if a recent zero-delta tunneled radar result is pending."""
        if self.pending_radar_empty_delta_ms <= 0:
            return False
        now = _test_hooks.get_current_time_ms()
        recent = now - self.pending_radar_empty_delta_ms <= _RADAR_CACHE_REFRESH_WINDOW_MS
        self.pending_radar_empty_delta_ms = 0
        return recent

    def radar_bounds(self) -> tuple[int, int, int, int]:
        """Return inclusive current radar coverage bounds.

        Returns:
            Inclusive ``(left, top, right, bottom)`` radar bounds.
        """
        self_state = self.world_state["self_state"]
        if self_state is None:
            return viewport_radar_bounds(self.world_state["viewport"])
        if self.pending_radar_uses_extra:
            return viewport_radar_bounds(self.world_state["viewport"])
        return regular_radar_bounds(self_state["x"], self_state["y"], self_state["rank"])
