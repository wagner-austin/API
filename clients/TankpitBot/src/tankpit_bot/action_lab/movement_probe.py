"""Live movement action probe harness."""

from __future__ import annotations

from typing import Literal, Protocol

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_locations import build_distinct_ground_targets
from tankpit_bot.action_lab.movement_probe_types import (
    MovementProbeAttemptResultDict,
    MovementProbeSessionDict,
    encode_movement_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import capture_page_client_snapshot
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.teleport import TeleportProbe
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)
_POLL_INTERVAL_MS = 100.0
_MOVEMENT_PROBE_TARGET_STEP = 4
_MOVEMENT_PROBE_TARGET_MAX_RADIUS = 24


class MovementProbeError(Exception):
    """Raised when the movement probe cannot proceed."""


class MovementOutcomeProbeProtocol(BufferedMessageSourceProtocol, Protocol):
    """Minimal probe protocol needed for movement settlement waits."""

    def _update_state_from_world(self) -> None:
        """Advance internal state from the current world snapshot."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state."""


def _require_positive(value: int, field: str) -> int:
    """Return a positive integer value or raise."""
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _find_first_sent_label_timestamp(
    messages: list[CapturedMessage],
    *,
    start_index: int,
    label: str,
) -> int | None:
    """Return the first sent-frame timestamp with the requested label."""
    for message in messages[start_index:]:
        if message["direction"] != "sent":
            continue
        if message.get("sent_origin") != "bot_injected":
            continue
        if message.get("sent_label") != label:
            continue
        return message["timestamp_ms"]
    return None


def _wait_for_move_outcome(
    page: action_session.WaitPageProtocol,
    probe: MovementOutcomeProbeProtocol,
    *,
    target_x: int,
    target_y: int,
    move_started_ms: int,
    timeout_ms: int,
) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
    """Wait for an exact movement arrival or movement timeout."""
    while action_hooks.get_current_time_ms() - move_started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe)
        probe._update_state_from_world()
        self_state = probe.get_self_state()
        if self_state is None:
            raise MovementProbeError("self state disappeared while waiting for movement")
        if self_state["x"] == target_x and self_state["y"] == target_y:
            completion_timestamp_ms = action_hooks.get_current_time_ms()
            return (
                "arrived_exact",
                completion_timestamp_ms,
                completion_timestamp_ms - move_started_ms,
                self_state["x"],
                self_state["y"],
            )
        page.wait_for_timeout(_POLL_INTERVAL_MS)
    self_state = probe.get_self_state()
    if self_state is None:
        raise MovementProbeError("self state disappeared after movement timeout")
    completion_timestamp_ms = action_hooks.get_current_time_ms()
    return (
        "move_timeout",
        completion_timestamp_ms,
        completion_timestamp_ms - move_started_ms,
        self_state["x"],
        self_state["y"],
    )


def format_movement_probe_summary(session: MovementProbeSessionDict) -> str:
    """Format a compact summary for a movement probe session."""
    arrived_exact = 0
    move_timeout = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "arrived_exact":
            arrived_exact += 1
        else:
            move_timeout += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Movement probe complete: "
        f"attempts={len(session['attempts'])} "
        f"arrived_exact={arrived_exact} "
        f"move_timeout={move_timeout} "
        f"queue_map_open_during_move={session['queue_map_open_during_move']} "
        f"bootstrap_ms={bootstrap_ms}"
    )


def _create_movement_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> MovementProbe:
    """Construct the concrete movement probe implementation."""
    return MovementProbe(
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )


def _get_probe_terrain_map() -> TerrainMapProtocol | None:
    """Return the active terrain map for movement target selection."""
    return get_terrain_map()


def _build_probe_targets(
    origin_x: int,
    origin_y: int,
    terrain: TerrainMapProtocol,
    *,
    max_targets: int,
) -> list[TeleportTargetDict]:
    """Build default movement probe targets near the provided origin."""
    return build_distinct_ground_targets(
        origin_x,
        origin_y,
        terrain,
        count=max_targets,
        step=_MOVEMENT_PROBE_TARGET_STEP,
        max_radius=_MOVEMENT_PROBE_TARGET_MAX_RADIUS,
    )


class MovementProbe(TeleportProbe):
    """Live movement probe that isolates walk settlement behavior."""

    def _build_default_targets(self, *, max_targets: int) -> list[TeleportTargetDict]:
        """Return default local movement targets near spawn."""
        terrain = _get_probe_terrain_map()
        if terrain is None:
            raise MovementProbeError("terrain map is unavailable")
        self_state = self._require_self_state()
        return _build_probe_targets(
            self_state["x"],
            self_state["y"],
            terrain,
            max_targets=max_targets,
        )

    def _probe_single_movement_target(
        self,
        target: TeleportTargetDict,
        *,
        move_timeout_ms: int,
        queue_map_open_during_move: bool,
        map_open_delay_ms: int,
        settle_delay_ms: int,
    ) -> MovementProbeAttemptResultDict:
        """Run one movement attempt against the live server.

        Captures a page-client snapshot immediately before the move
        command dispatches and again after settlement; these snapshots
        provide a side-by-side view of what the live JS client believed
        about the tank's state at each boundary.
        """
        page = self._require_page()
        cdp = self._cdp
        if cdp is None:
            raise MovementProbeError("cdp session is unavailable")
        world_before = self.get_world_state()
        self_state_before = self._require_self_state()
        fuel_before = self_state_before["fuel"]
        world_timestamp_before = world_before["timestamp_ms"]
        snapshot_before = capture_page_client_snapshot(cdp)

        self._reset_probe_state_to_idle()
        message_start_index = len(self.messages)
        move_started_ms = action_hooks.get_current_time_ms()
        if not self.move_to(target["x"], target["y"]):
            raise MovementProbeError("move command dispatch failed")

        map_open_requested_ms: int | None = None
        if queue_map_open_during_move:
            if map_open_delay_ms > 0:
                page.wait_for_timeout(float(map_open_delay_ms))
                action_hooks.drain_buffered_messages(self)
            # The wire ``CMD_MAP_OPEN`` only opens the map; re-sending against an
            # already-open map is a server-side no-op (no fresh map sync). If the
            # live JS client already shows the overlay -- e.g. a prior attempt
            # opened it and the player/probe never closed it -- skip the dispatch
            # and record the skip via ``map_open_requested_ms=None``. Mirrors the
            # short-circuit in ``run_tracked_acquisition_phase``.
            mid_move_snapshot = capture_page_client_snapshot(cdp)
            if mid_move_snapshot["map_visible"] is True:
                emit_diagnostic(
                    diagnostic_kind="movement_probe_map_already_showing",
                    target_x=target["x"],
                    target_y=target["y"],
                    map_open_delay_ms=map_open_delay_ms,
                )
            else:
                map_open_requested_ms = action_hooks.get_current_time_ms()
                if not self.open_map():
                    raise MovementProbeError(
                        "map_open command dispatch failed during movement probe"
                    )

        (
            status,
            completion_timestamp_ms,
            move_elapsed_ms,
            settled_x,
            settled_y,
        ) = _wait_for_move_outcome(
            page,
            self,
            target_x=target["x"],
            target_y=target["y"],
            move_started_ms=move_started_ms,
            timeout_ms=move_timeout_ms,
        )
        self._reset_probe_state_to_idle()
        if settle_delay_ms > 0:
            page.wait_for_timeout(float(settle_delay_ms))

        self_state_after = self.get_self_state()
        if self_state_after is None:
            raise MovementProbeError("self state is unavailable after movement probe attempt")
        snapshot_after = capture_page_client_snapshot(cdp)
        return MovementProbeAttemptResultDict(
            target=target,
            status=status,
            move_started_ms=move_started_ms,
            map_open_requested_ms=map_open_requested_ms,
            map_open_message_timestamp_ms=_find_first_sent_label_timestamp(
                self.messages,
                start_index=message_start_index,
                label="map_open",
            ),
            completion_timestamp_ms=completion_timestamp_ms,
            move_elapsed_ms=move_elapsed_ms,
            fuel_before=fuel_before,
            fuel_after=self_state_after["fuel"],
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=self.get_world_state()["timestamp_ms"],
            settled_x=settled_x,
            settled_y=settled_y,
            message_start_index=message_start_index,
            message_end_index=len(self.messages),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    def execute_probe(
        self,
        *,
        explicit_targets: list[TeleportTargetDict] | None,
        max_targets: int,
        initial_sync_timeout_ms: int,
        move_timeout_ms: int,
        queue_map_open_during_move: bool,
        map_open_delay_ms: int,
        settle_delay_ms: int,
    ) -> MovementProbeSessionDict:
        """Run the live movement probe session."""
        _require_positive(max_targets, "max_targets")
        _require_positive(initial_sync_timeout_ms, "initial_sync_timeout_ms")
        _require_positive(move_timeout_ms, "move_timeout_ms")
        if map_open_delay_ms < 0:
            raise ValueError("map_open_delay_ms must be non-negative")
        if settle_delay_ms < 0:
            raise ValueError("settle_delay_ms must be non-negative")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> MovementProbeSessionDict:
            targets = (
                explicit_targets[:max_targets]
                if explicit_targets is not None
                else self._build_default_targets(max_targets=max_targets)
            )
            if not targets:
                raise MovementProbeError("movement probe requires at least one target")
            attempts: list[MovementProbeAttemptResultDict] = []
            for target in targets:
                attempts.append(
                    self._probe_single_movement_target(
                        target,
                        move_timeout_ms=move_timeout_ms,
                        queue_map_open_during_move=queue_map_open_during_move,
                        map_open_delay_ms=map_open_delay_ms,
                        settle_delay_ms=settle_delay_ms,
                    )
                )
            first_attempt_started_ms = attempts[0]["move_started_ms"] if attempts else None
            session_envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_attempt_started_ms,
            )
            return MovementProbeSessionDict(
                session_id=session_envelope.session_id,
                start_timestamp_ms=session_envelope.start_timestamp_ms,
                end_timestamp_ms=session_envelope.end_timestamp_ms,
                base_url=session_envelope.base_url,
                spawn_x=session_envelope.spawn_x,
                spawn_y=session_envelope.spawn_y,
                max_targets=max_targets,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=session_envelope.startup_timing,
                move_timeout_ms=move_timeout_ms,
                settle_delay_ms=settle_delay_ms,
                queue_map_open_during_move=queue_map_open_during_move,
                map_open_delay_ms=map_open_delay_ms,
                targets=targets,
                attempts=attempts,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def run_movement_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    explicit_targets: list[TeleportTargetDict] | None = None,
    max_targets: int = 3,
    initial_sync_timeout_ms: int = 10000,
    move_timeout_ms: int = 5000,
    queue_map_open_during_move: bool = False,
    map_open_delay_ms: int = 0,
    settle_delay_ms: int = 500,
) -> MovementProbeSessionDict:
    """Run a live movement probe and save the session JSON."""

    def _run_session(probe: MovementProbe) -> MovementProbeSessionDict:
        return probe.execute_probe(
            explicit_targets=explicit_targets,
            max_targets=max_targets,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            move_timeout_ms=move_timeout_ms,
            queue_map_open_during_move=queue_map_open_during_move,
            map_open_delay_ms=map_open_delay_ms,
            settle_delay_ms=settle_delay_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_movement_probe,
        run_session=_run_session,
        encoder=encode_movement_probe_session,
        summary_formatter=format_movement_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "MovementProbe",
    "MovementProbeError",
    "_find_first_sent_label_timestamp",
    "_wait_for_move_outcome",
    "format_movement_probe_summary",
    "run_movement_probe",
]
