"""World state service — owns all mutable game state as instance attributes.

Replaces the 16 module-level globals in ``world_state.py`` with a single
injectable service. Dispatch modules receive a ``WorldService`` instance
instead of importing ``world_state`` as ``_ws`` and reaching into private
module attributes.

Production code creates one ``WorldService`` per session; tests create a
fresh instance per test (no global resets needed).
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
    ItemType,
)
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    update_self_position,
    viewport_scan_key,
)
from tankpit_bot.state.viewport_geometry import (
    regular_radar_bounds,
    viewport_radar_bounds,
    viewport_visible_bounds,
)

log = get_logger(__name__)

ITEM_TYPES: list[ItemType] = [
    "armor_shields",
    "dual_shots",
    "missile_shots",
    "homing_shots",
    "extra_radars",
]

WEAPON_BYTE_TO_ITEM: dict[int, ItemType] = {
    1: "dual_shots",
    2: "missile_shots",
    3: "homing_shots",
}

_FAILED_MOVE_TTL_MS = 30000
_FAILED_SCAN_VIEWPORT_TTL_MS = 30000
_RADAR_CACHE_REFRESH_WINDOW_MS = 2000


def _make_empty_inventory() -> InventoryState:
    """Create an empty inventory state with all items at zero.

    Returns:
        InventoryState with all counts at 0 and enabled False.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=0, enabled=False),
        dual_shots=InventoryItem(count=0, enabled=False),
        missile_shots=InventoryItem(count=0, enabled=False),
        homing_shots=InventoryItem(count=0, enabled=False),
        extra_radars=InventoryItem(count=0, enabled=False),
    )


class WorldService:
    """Owns all mutable game state for one session.

    Instance attributes mirror the 16 module-level globals that were
    previously in ``world_state.py``. Dispatch modules receive a
    ``WorldService`` instance and mutate it directly.
    """

    def __init__(self) -> None:
        """Initialize empty world state for a new session."""
        self.world_state: WorldStateDict = make_empty_world_state()
        self.terrain_map: _test_hooks.TerrainMapProtocol | None = None
        self.room_images: dict[str, str] = {}
        self.selected_room: str | None = None
        self.inventory_state: InventoryState = _make_empty_inventory()
        self.got_confirmed_hit: bool = False
        self.got_our_shot_response: bool = False
        # Tank id present on the target tile of our most recent shot, or
        # -1 if the tile was empty. Set by ``mark_combat_hit`` (legacy
        # function name; now consumes 0x53 ShootEvent fields after the
        # 2026-06-19 decoder unification). Consumed by combat_feedback
        # to distinguish hits on the intended target from incidental
        # hits (e.g. homing seeker locking onto a closer enemy than the
        # bot commanded).
        self.last_shot_victim_id: int = -1
        # Tank id our planner intended to shoot at on the most recent
        # ``shoot`` dispatch (the ``target_id`` arg to ``shoot_at``).
        # Used by the 0x53 ShootEvent dispatcher to attribute homing /
        # missile seeker resolutions: when a tracking weapon resolves
        # at ``(tx, ty)``, the server's seeker chased the locked target
        # to that tile, so ``(tx, ty)`` is a fresh wire datum for the
        # locked target's position (and works even when the target is
        # off-viewport and 0x2E TankStatusSync stops broadcasting). The
        # dispatcher uses it to update the locked target's tracked
        # position so the next shot aims at fresh server-known coords.
        self.last_shot_combat_target_id: int = -1
        # Inventory snapshot taken at the most recent shoot dispatch.
        # Used by combat_feedback to confirm hits via ammo delta: the
        # server only debits dual / missile / homing ammo on a hit, so
        # a decrement in any of those slots between snapshot time and
        # feedback time confirms the shot landed -- even when the
        # 0x53 ``victim_id`` lookup misses because the target is off
        # the bot's viewport (live run 2026-06-24 12:43: 4 pursuit
        # homings logged as misses because ``_find_tank_at_tile`` could
        # not see purple-9 off-viewport). ``None`` when no shoot is
        # pending or the delta has already been consumed.
        self.pending_shot_inventory_snapshot: InventoryState | None = None
        self.killed_tank_ids: set[int] = set()
        self.tank_death_anchors: dict[int, tuple[int, int]] = {}
        self.teleport_landed: bool = False
        self.radar_scan_complete: bool = False
        self.map_data_processed: bool = False
        self.pending_radar_cache_refresh_ms: int = 0
        self.pending_radar_empty_delta_ms: int = 0
        self.pending_radar_uses_extra: bool = True
        self.failed_move_targets: dict[str, int] = {}
        self.failed_scan_viewports: dict[str, int] = {}
        self.last_command_error: int = -1
        # ContainerPickup de-duplication. The server broadcasts each
        # 0x43 pickup TWICE within ~200 ms (one to the picker, one to
        # the world view) -- empirical 43.9% duplicate rate across 13
        # sniff sessions (2026-06-20). The dispatcher records each
        # pickup signature here with its receipt timestamp; the next
        # arrival of the same signature within
        # ``PICKUP_DEDUP_WINDOW_MS`` is suppressed.
        self.recent_pickup_signatures: dict[tuple[tuple[int, int, int], ...], int] = {}
        # Career stats from the most recent 0x56 Statistics broadcast.
        # The wire sends these every ~10 s; the AI uses ``destroyed`` to
        # gate "I have N kills, time to play conservatively" decisions,
        # ``deactivated`` for the K/D ratio in scorecards, and
        # ``playtime_seconds_total`` to enforce a session length cap
        # without depending on wall-clock drift. ``-1`` means the wire
        # has not yet sent a Statistics broadcast this session.
        self.career_destroyed: int = -1
        self.career_deactivated: int = -1
        self.career_score: int = -1
        self.career_playtime_seconds_total: int = -1
        self.career_stats_last_update_ms: int = 0
        # 0x2F ActivePlayers roster of (tank_id, rank) tuples from the
        # most recent server broadcast. Empty until the server sends
        # the first one (usually after join-confirm).
        self.active_players: list[tuple[int, int]] = []
        # 0x31 Top10 latest snapshot. ``-1`` means the wire hasn't sent
        # a Top10 broadcast yet.
        self.top10_viewer_score: int = -1
        self.top10_viewer_position: int = -1
        self.top10_team_filter: int = -1
        # 0x60 PingResponse wall-clock. The bot's session-health
        # monitor reads this to detect long server silences without
        # racing the WebSocket layer.
        self.last_ping_response_ms: int = 0

    # -----------------------------------------------------------------
    # World state accessors
    # -----------------------------------------------------------------

    def get_world_state(self) -> WorldStateDict:
        """Get the current world state.

        Returns:
            Current WorldStateDict with containers, mines, self_state, etc.
        """
        return self.world_state

    def get_terrain_map(self) -> _test_hooks.TerrainMapProtocol | None:
        """Get the current terrain map, loading if needed.

        Returns:
            TerrainMap instance, or None if terrain GIF not found.
        """
        return self._load_terrain_map_if_needed()

    # -----------------------------------------------------------------
    # Radar / scan event flags
    # -----------------------------------------------------------------

    def mark_radar_scan_complete(self) -> None:
        """Record that the server completed a radar scan."""
        self.radar_scan_complete = True

    def check_and_clear_radar_scan_complete(self) -> bool:
        """Check if a radar scan completed since last check, then clear.

        Returns:
            True if radar completion was observed.
        """
        result = self.radar_scan_complete
        self.radar_scan_complete = False
        return result

    def mark_map_data_processed(self) -> None:
        """Record that a MAP_DATA world-state blob was parsed into positions."""
        self.map_data_processed = True

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

    # -----------------------------------------------------------------
    # Radar cache refresh tracking
    # -----------------------------------------------------------------

    def mark_pending_radar_cache_refresh(self) -> None:
        """Record that a recent combined-tile update may belong to a radar scan."""
        self.pending_radar_cache_refresh_ms = get_current_time_ms()

    def consume_pending_radar_cache_refresh(self) -> bool:
        """Return True if a recent combined-tile update should count as radar."""
        if self.pending_radar_cache_refresh_ms <= 0:
            return False
        now = get_current_time_ms()
        recent = now - self.pending_radar_cache_refresh_ms <= _RADAR_CACHE_REFRESH_WINDOW_MS
        self.pending_radar_cache_refresh_ms = 0
        return recent

    def mark_pending_radar_empty_delta(self) -> None:
        """Record that a zero-delta tunneled radar result was observed."""
        self.pending_radar_empty_delta_ms = get_current_time_ms()

    def consume_pending_radar_empty_delta(self) -> bool:
        """Return True if a recent zero-delta tunneled radar result is pending."""
        if self.pending_radar_empty_delta_ms <= 0:
            return False
        now = get_current_time_ms()
        recent = now - self.pending_radar_empty_delta_ms <= _RADAR_CACHE_REFRESH_WINDOW_MS
        self.pending_radar_empty_delta_ms = 0
        return recent

    # -----------------------------------------------------------------
    # Position updates
    # -----------------------------------------------------------------

    def update_world_state_from_position(self, x: int, y: int) -> None:
        """Update world state with new self position.

        Args:
            x: Self X coordinate.
            y: Self Y coordinate.
        """
        self.world_state = update_self_position(self.world_state, x, y, get_current_time_ms())

    # -----------------------------------------------------------------
    # Failed move / scan tracking
    # -----------------------------------------------------------------

    def mark_move_target_failed(self, x: int, y: int, timestamp_ms: int) -> None:
        """Record a move destination that stalled and timed out.

        Args:
            x: Failed destination X coordinate.
            y: Failed destination Y coordinate.
            timestamp_ms: When the failure was detected.
        """
        key = f"{x},{y}"
        self.failed_move_targets[key] = timestamp_ms
        log.info("MOVE: marked (%d,%d) as failed target", x, y)

    def is_move_target_failed(self, x: int, y: int, now_ms: int) -> bool:
        """Check if a move target was recently marked as failed.

        Args:
            x: Destination X coordinate.
            y: Destination Y coordinate.
            now_ms: Current timestamp for TTL check.

        Returns:
            True if the target failed recently and should be avoided.
        """
        key = f"{x},{y}"
        failed_ms = self.failed_move_targets.get(key)
        if failed_ms is None:
            return False
        return (now_ms - failed_ms) < _FAILED_MOVE_TTL_MS

    def clear_failed_move_targets(self) -> None:
        """Clear all failed move targets. Called on fresh radar data."""
        self.failed_move_targets.clear()

    def mark_scan_viewport_failed(
        self,
        viewport_left: int,
        viewport_top: int,
        timestamp_ms: int,
    ) -> None:
        """Record a viewport whose radar scan stalled and timed out.

        Args:
            viewport_left: Failed viewport left X coordinate.
            viewport_top: Failed viewport top Y coordinate.
            timestamp_ms: When the failure was detected.
        """
        key = viewport_scan_key(viewport_left, viewport_top)
        self.failed_scan_viewports[key] = timestamp_ms
        log.info(
            "SCAN: marked viewport (%d,%d) as failed target",
            viewport_left,
            viewport_top,
        )

    def is_scan_viewport_failed(
        self,
        viewport_left: int,
        viewport_top: int,
        now_ms: int,
    ) -> bool:
        """Check whether a viewport recently had a stalled radar scan.

        Args:
            viewport_left: Viewport left X coordinate.
            viewport_top: Viewport top Y coordinate.
            now_ms: Current timestamp for TTL evaluation.

        Returns:
            True if radar recently stalled for that viewport.
        """
        key = viewport_scan_key(viewport_left, viewport_top)
        failed_ms = self.failed_scan_viewports.get(key)
        if failed_ms is None:
            return False
        return (now_ms - failed_ms) < _FAILED_SCAN_VIEWPORT_TTL_MS

    def clear_failed_scan_viewport(self, viewport_left: int, viewport_top: int) -> None:
        """Clear a failed-scan mark for a specific viewport origin.

        Args:
            viewport_left: Viewport left X coordinate.
            viewport_top: Viewport top Y coordinate.
        """
        key = viewport_scan_key(viewport_left, viewport_top)
        self.failed_scan_viewports.pop(key, None)

    # -----------------------------------------------------------------
    # Room / terrain map management
    # -----------------------------------------------------------------

    def register_room_image(self, room_id: str, image: str) -> None:
        """Register a room's field image from a ROOM_LIST message.

        Args:
            room_id: Room ID (e.g. "2").
            image: Field image filename (e.g. "field42.gif").
        """
        self.room_images[room_id] = image

    def set_selected_room(self, room_id: str) -> None:
        """Track which room was selected from a SELECT message.

        Resets the terrain map so the correct one loads on next render.

        Args:
            room_id: Room ID that was selected.
        """
        self.selected_room = room_id
        self.terrain_map = None
        image = self.room_images.get(room_id)
        log.info("Selected room %s (field image: %s)", room_id, image or "unknown")
        emit_diagnostic(
            diagnostic_kind="session_room_joined",
            room_id=room_id,
            field_image=image if image is not None else "unknown",
        )

    # -----------------------------------------------------------------
    # Viewport / radar geometry helpers
    # -----------------------------------------------------------------

    def viewport_bounds(self) -> tuple[int, int, int, int]:
        """Return inclusive visible viewport bounds.

        Returns:
            Inclusive ``(left, top, right, bottom)`` viewport bounds.
        """
        return viewport_visible_bounds(self.world_state["viewport"])

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
        return regular_radar_bounds(self_state["x"], self_state["y"])

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _find_field_gif(self, image: str) -> Path | None:
        """Find the local GIF file for a field image name.

        Args:
            image: Field image filename from server (e.g. "field42.gif").

        Returns:
            Path to the local GIF file, or None if not found.
        """
        stem = image.removesuffix(".gif")
        candidates = [
            Path(f"{stem}_r.gif"),
            Path(f"{stem}-r.gif"),
        ]
        for path in candidates:
            if _test_hooks.path_exists(path):
                return path
        return None

    def _load_terrain_map_if_needed(self) -> _test_hooks.TerrainMapProtocol | None:
        """Load terrain map for the selected room.

        Returns:
            TerrainMap instance, or None if file not found.
        """
        if self.terrain_map is not None:
            return self.terrain_map

        if self.selected_room is None:
            log.warning("No selected room is available for terrain-map loading")
            return None
        image = self.room_images.get(self.selected_room)
        if image is None:
            log.warning("No registered room image for selected room %s", self.selected_room)
            return None
        gif_path = self._find_field_gif(image)
        if gif_path is None:
            log.warning("No local GIF found for %s (room %s)", image, self.selected_room)
            return None
        self.terrain_map = _test_hooks.load_terrain_map(gif_path)
        log.info("Loaded terrain map from %s (room %s)", gif_path, self.selected_room)
        return self.terrain_map


__all__ = [
    "ITEM_TYPES",
    "WEAPON_BYTE_TO_ITEM",
    "WorldService",
]
