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

_LANDING_REFUSAL_TTL_MS = 30000
"""How long a refused landing blocks the requested tile's ring.

The same 30 s family as the move/scan marks: long enough that the
2 s replan cycle cannot re-certify the identical hop (the 2026-08-21
marooning ran 4 identical refusals in 10 s; the 08-05 ancestor ran
534 over 43 minutes), short enough that a genuinely cleared ring is
retried promptly. Within the window the forced repair radar
([[radar-mechanics]] § the s9-2 correction) reveals the actual mines,
so the ordinary mine beliefs own the answer from then on."""

_ROUTINE_DISPLACEMENT_CHEBYSHEV = 1
"""A one-tile landing shift is the ring-1 displacement law working
([[teleport-mechanics]]: E -> N -> W -> S within ring 1) — the mission
continuing, never a refusal. Beyond one tile no ejection exists: the
mined law (137/137 archived receipts, 2026-08-21) is that a fully
blocked ring REFUSES the hop and the tank stays at its origin."""


class WorldServiceMovementMixin:
    """Failed-move and rejection memory for one session's world service.

    The attributes below are DECLARATIONS, not assignments: the
    session's ``__init__`` remains their single owner, so this split
    does not move any per-session state.
    """

    failed_move_targets: dict[str, int]
    failed_scan_viewports: dict[str, int]
    movement_rejections: list[int]
    landing_refusals: dict[str, int]

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

    def mark_landing_refused(
        self, requested_x: int, requested_y: int, chebyshev: int, timestamp_ms: int
    ) -> None:
        """Record a refused teleport as ring-blocked evidence.

        The mined law ([[teleport-mechanics]] § the refusal law,
        137/137 archived receipts): when the requested tile AND its
        whole ring-1 are blocked, the server refuses the hop — the
        tank stays at its origin, uncharged. The bot perceives this as
        "landed at origin", far from the request; for four months that
        receipt fed nothing, which is what let the identical hop
        re-certify indefinitely (the 08-05 534-refusal session, the
        2026-08-21 marooning). What one refusal PROVES is exactly
        "requested tile + ring-1 all blocked" — nothing more, which is
        why the evidence zone is ring-1, never the origin distance.
        Routine one-tile ring-1 displacements are real movement and
        never recorded.

        Args:
            requested_x: The teleport's requested X.
            requested_y: The teleport's requested Y.
            chebyshev: Chebyshev distance from request to landing —
                the refusal signature is >= 2 (landed back at origin).
            timestamp_ms: When the refusal was observed.
        """
        if chebyshev <= _ROUTINE_DISPLACEMENT_CHEBYSHEV:
            return
        self.landing_refusals[f"{requested_x},{requested_y}"] = timestamp_ms
        log.info(
            "TELEPORT: landing refusal at (%d,%d) - ring-1 proven blocked",
            requested_x,
            requested_y,
        )

    def hostile_landing_keys(self, now_ms: int) -> frozenset[str]:
        """Expand fresh landing refusals into blocked landing tiles.

        Consumed by the composed decision terrain each tick, exactly
        as the hostile-mine set is: the refused tile and its ring-1
        (the zone one refusal actually proves) are not attainable
        landings until the evidence ages out (or the forced repair
        radar's reveals let the ordinary mine beliefs answer).

        Args:
            now_ms: Current wall-clock ms for the TTL check.

        Returns:
            Tile keys of each fresh refusal's requested tile + ring-1.
        """
        keys: set[str] = set()
        for tile_key, marked_ms in list(self.landing_refusals.items()):
            if now_ms - marked_ms >= _LANDING_REFUSAL_TTL_MS:
                del self.landing_refusals[tile_key]
                continue
            center_x, center_y = tile_key.split(",")
            cx, cy = int(center_x), int(center_y)
            for x in range(max(0, cx - 1), min(255, cx + 1) + 1):
                for y in range(max(0, cy - 1), min(255, cy + 1) + 1):
                    keys.add(f"{x},{y}")
        return frozenset(keys)

    def has_fresh_landing_refusal(self, now_ms: int) -> bool:
        """Return whether any landing refusal is currently fresh.

        The repair-radar gate reads this: a fresh refusal means the
        local mine beliefs just failed an exam, and scanned coverage
        must not suppress the scan that repairs them.

        Args:
            now_ms: Current wall-clock ms for the TTL check.

        Returns:
            True when at least one refusal is inside its TTL.
        """
        return any(
            now_ms - marked_ms < _LANDING_REFUSAL_TTL_MS
            for marked_ms in self.landing_refusals.values()
        )

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
