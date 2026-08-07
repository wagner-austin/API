"""Failed-move and rejection memory for one session's world service.

The TTL'd records of move targets the server refused and viewports a
scan never covered. Mixed into
:class:`~tankpit_bot.sniffer.world_service.WorldService`, which owns the
state these annotate.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.state import (
    viewport_scan_key,
)

log = get_logger(__name__)

_FAILED_MOVE_TTL_MS = 30000

_FAILED_SCAN_VIEWPORT_TTL_MS = 30000


class WorldServiceMovementMixin:
    """Failed-move and rejection memory for one session's world service.

    The attributes below are DECLARATIONS, not assignments: the
    session's ``__init__`` remains their single owner, so this split
    does not move any per-session state.
    """

    failed_move_targets: dict[str, int]
    failed_scan_viewports: dict[str, int]
    movement_rejections: list[int]

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

    def record_movement_rejection(self, timestamp_ms: int) -> None:
        """Record a server cant_go refusal of a movement leg.

        Every move, walk-pickup, or teleport dispatch the server
        answers with ``cant_go`` lands here regardless of the
        command's kind — the shared fact is "the tank tried to move
        and the server said no." Consumers count refusals in a
        trailing window to detect a movement-dead tank (run
        bot-20260730-110x ticks 95-107: twelve consecutive rejected
        walk-pickups under fire while the escape kept planning walks).

        Args:
            timestamp_ms: When the rejection arrived.
        """
        self.movement_rejections.append(timestamp_ms)

    def recent_movement_rejections(self, now_ms: int, window_ms: int) -> int:
        """Count movement rejections inside the trailing window.

        Prunes entries older than the window so the record never
        grows beyond live relevance.

        Args:
            now_ms: Current wall-clock ms.
            window_ms: Trailing window length.

        Returns:
            Number of rejections with ``timestamp > now - window``.
        """
        floor = now_ms - window_ms
        self.movement_rejections = [ts for ts in self.movement_rejections if ts > floor]
        return len(self.movement_rejections)

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
