"""TypedDict models for live teleport probe sessions."""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
)


class TeleportTargetDict(TypedDict):
    """Requested destination for a teleport probe attempt.

    Attributes:
        label: Human-readable label for the destination.
        x: Requested world X tile coordinate.
        y: Requested world Y tile coordinate.
    """

    label: str
    x: int
    y: int


class TeleportAttemptResultDict(TypedDict):
    """Outcome of one teleport probe attempt.

    Attributes:
        target: Requested target for the attempt.
        teleport_cycle_id: Teleport phase cycle id for the attempt.
        status: Attempt result classification.
        map_open_started_ms: Timestamp when the map-open toggle was sent.
        map_sync_timestamp_ms: Timestamp of the first fresh world sync after
            the map-open toggle, if any.
        teleport_started_ms: Timestamp when the teleport command was sent, if any.
        completion_timestamp_ms: Timestamp when the attempt reached a terminal outcome.
        map_sync_elapsed_ms: Milliseconds from map-open send to fresh sync, if any.
        teleport_elapsed_ms: Milliseconds from teleport send to terminal outcome, if any.
        fuel_before: Fuel immediately before the map-open command.
        fuel_after: Fuel observed at completion, if self state is available.
        world_timestamp_before: World-state timestamp before the attempt began.
        world_timestamp_after: World-state timestamp at completion.
        landed_signal_received: Whether a teleport-landed confirmation was observed.
        landed_x: Actual landed X coordinate, if available.
        landed_y: Actual landed Y coordinate, if available.
        message_start_index: Index of the first raw captured message for the attempt.
        message_end_index: Exclusive index after the last raw captured message for the attempt.
        page_snapshots: Page-client diagnostic snapshots captured during the attempt.
    """

    target: TeleportTargetDict
    teleport_cycle_id: int
    status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"]
    map_open_started_ms: int
    map_sync_timestamp_ms: int | None
    teleport_started_ms: int | None
    completion_timestamp_ms: int
    map_sync_elapsed_ms: int | None
    teleport_elapsed_ms: int | None
    fuel_before: int
    fuel_after: int | None
    world_timestamp_before: int
    world_timestamp_after: int
    landed_signal_received: bool
    landed_x: int | None
    landed_y: int | None
    message_start_index: int
    message_end_index: int
    page_snapshots: list[TeleportPageSnapshotDict]


class TeleportPageSnapshotDict(PageClientSnapshotDict):
    """Page-client snapshot annotated with a teleport-attempt phase.

    Extends the universal :class:`PageClientSnapshotDict` with the
    teleport-specific phase label so multiple snapshots in one attempt can
    be distinguished by where in the sequence they were captured. All
    other fields are inherited verbatim from the universal snapshot.

    Attributes:
        phase: Attempt phase when the snapshot was captured.
    """

    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]


class TeleportStartupTimingDict(TypedDict):
    """Startup timing milestones for a live teleport probe session.

    Attributes:
        game_ready_timestamp_ms: Timestamp when the game-ready wait completed.
        intel_ready_timestamp_ms: Timestamp when probe intel collection completed.
        initial_sync_started_ms: Timestamp when initial world/self sync wait began.
        initial_world_timestamp_ms: Timestamp when the initial self state became available.
        command_ready_timestamp_ms: Timestamp when startup state advancement reached IDLE.
        first_attempt_started_ms: Timestamp when the first teleport attempt began, if any.
        game_ready_to_intel_ready_ms: Delay from game-ready to intel completion.
        intel_ready_to_initial_world_ms: Delay from intel completion to initial self state.
        initial_world_to_command_ready_ms: Delay from first self state to command-ready state.
        command_ready_to_first_attempt_ms: Delay from command-ready to first attempt, if any.
    """

    game_ready_timestamp_ms: int
    intel_ready_timestamp_ms: int
    initial_sync_started_ms: int
    initial_world_timestamp_ms: int
    command_ready_timestamp_ms: int
    first_attempt_started_ms: int | None
    game_ready_to_intel_ready_ms: int
    intel_ready_to_initial_world_ms: int
    initial_world_to_command_ready_ms: int
    command_ready_to_first_attempt_ms: int | None


class TeleportProbeSessionDict(TypedDict):
    """Complete live teleport probe session.

    Attributes:
        session_id: Probe session identifier.
        start_timestamp_ms: Session start timestamp in milliseconds.
        end_timestamp_ms: Session end timestamp in milliseconds.
        base_url: Target URL used for the session.
        spawn_x: Initial spawn X coordinate after joining the game.
        spawn_y: Initial spawn Y coordinate after joining the game.
        teleport_strategy: Selected teleport sequencing strategy.
        max_targets: Maximum number of targets requested for the session, if limited.
        capture_session_path: Path to the replayable raw capture session JSON.
        initial_sync_timeout_ms: Configured initial self-state sync timeout.
        startup_timing: Startup timing milestones before the first attempt.
        map_sync_timeout_ms: Configured map-sync timeout.
        teleport_timeout_ms: Configured teleport timeout.
        settle_delay_ms: Delay inserted after each completed attempt.
        targets: Requested target list for the session.
        attempts: Recorded attempt outcomes in order.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"]
    max_targets: int | None
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    map_sync_timeout_ms: int
    teleport_timeout_ms: int
    settle_delay_ms: int
    targets: list[TeleportTargetDict]
    attempts: list[TeleportAttemptResultDict]
