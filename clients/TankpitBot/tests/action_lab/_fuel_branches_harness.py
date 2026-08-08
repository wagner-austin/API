"""Shared builders for the fuel-probe branch tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Literal,
    Protocol,
)

from tests.action_lab._fuel_probe_harness import (
    _ProbeHarness,
    fuel_targets_module,
)
from tests.action_lab._replay_page import ReplayClock
from tests.action_lab.conftest import (
    ground_terrain,
    rock_wall,
)

from tankpit_bot._test_hooks import (
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_probe import FuelProbe
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportStartupTimingDict,
    TeleportTargetDict,
)
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.state import (
    ContainerStateDict,
    make_container_state,
)


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    """Build a sample page-client snapshot for fuel-probe branch tests."""
    return PageClientSnapshotDict(
        timestamp_ms=timestamp_ms,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=10,
        last_page_client_send_age_ms=20,
        last_bot_send_age_ms=30,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        world_collections={},
        map_fields={},
    )


class _WaitForTeleportOutcomeProtocol(Protocol):
    """Callable protocol for teleport outcome waiting."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict: ...


class _FuelProbeModuleProtocol(Protocol):
    """Typed access to patchable fuel probe globals."""

    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol
    _find_visible_fuel_target: Callable[[FuelProbe], ContainerStateDict | None]
    _visible_fuel_requires_reposition: Callable[[FuelProbe, ContainerStateDict], bool]
    _find_visible_fuel_landing_tile: Callable[
        [FuelProbe, ContainerStateDict],
        tuple[int, int] | None,
    ]
    visible_fuel_requires_reposition: Callable[[FuelProbe, ContainerStateDict], bool]
    find_visible_fuel_landing_tile: Callable[
        [FuelProbe, ContainerStateDict],
        tuple[int, int] | None,
    ]
    get_terrain_map: Callable[[], TerrainMapProtocol | None]


_fuel_module_import = __import__("tankpit_bot.action_lab.fuel_probe", fromlist=["fuel_probe"])


fuel_probe_module: _FuelProbeModuleProtocol = _fuel_module_import


class _SequenceProbeHarness(_ProbeHarness):
    """Probe harness with per-call command dispatch outcomes."""

    def __init__(
        self,
        clock: ReplayClock,
        *,
        open_map_results: list[bool] | None = None,
        teleport_results: list[bool] | None = None,
    ) -> None:
        super().__init__(clock)
        self._open_map_results = [True] if open_map_results is None else open_map_results
        self._teleport_results = [True] if teleport_results is None else teleport_results

    def open_map(self) -> bool:
        return self._open_map_results.pop(0)

    def teleport_to(self, x: int, y: int) -> bool:
        _ = (x, y)
        return self._teleport_results.pop(0)


def _target() -> TeleportTargetDict:
    """Build a sample fuel teleport target."""
    return TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)


def _startup_timing() -> TeleportStartupTimingDict:
    """Build startup timing for summary tests."""
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=300,
        intel_ready_timestamp_ms=350,
        initial_sync_started_ms=400,
        initial_world_timestamp_ms=450,
        command_ready_timestamp_ms=460,
        first_attempt_started_ms=500,
        game_ready_to_intel_ready_ms=50,
        intel_ready_to_initial_world_ms=100,
        initial_world_to_command_ready_ms=10,
        command_ready_to_first_attempt_ms=40,
    )


def _attempt(
    status: (
        Literal[
            "picked_up_fuel",
            "no_fuel_visible",
            "radar_timeout",
            "map_sync_timeout",
            "reposition_map_sync_timeout",
            "teleport_timeout",
            "reposition_teleport_timeout",
            "pickup_timeout",
        ]
    ),
) -> FuelProbeAttemptResultDict:
    """Build a minimal typed fuel attempt payload."""
    return {
        "target": _target(),
        "teleport_cycle_ids": [1],
        "radar_cycle_id": None,
        "move_cycle_id": None,
        "pickup_cycle_id": None,
        "status": status,
        "map_open_started_ms": 1000,
        "map_sync_timestamp_ms": 1200,
        "teleport_started_ms": 1300,
        "radar_started_ms": 1400,
        "radar_sync_timestamp_ms": 1500,
        "reposition_map_open_started_ms": None,
        "reposition_map_sync_timestamp_ms": None,
        "reposition_teleport_started_ms": None,
        "pickup_started_ms": None,
        "completion_timestamp_ms": 1600,
        "fuel_before": 700,
        "fuel_after": 650,
        "landed_signal_received": True,
        "landed_x": 124,
        "landed_y": 100,
        "fuel_target_x": None,
        "fuel_target_y": None,
        "fuel_target_volume": None,
        "phase_overlaps": [],
        "decision_basis": None,
        "message_start_index": 0,
        "message_end_index": 1,
        "snapshot_before": _snapshot(0),
        "snapshot_after": _snapshot(0),
    }


def _session_with_statuses(
    statuses: list[
        Literal[
            "picked_up_fuel",
            "no_fuel_visible",
            "radar_timeout",
            "map_sync_timeout",
            "reposition_map_sync_timeout",
            "teleport_timeout",
            "reposition_teleport_timeout",
            "pickup_timeout",
        ]
    ],
) -> FuelProbeSessionDict:
    """Build a summary session with selected terminal statuses."""
    return FuelProbeSessionDict(
        session_id="fuel-session",
        start_timestamp_ms=100,
        end_timestamp_ms=1000,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        target_pickups=2,
        max_attempts=len(statuses),
        capture_session_path="fuel_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
        attempts=[_attempt(status) for status in statuses],
    )


def _set_common_probe_hooks(teleport_outcome: _WaitForTeleportOutcomeProtocol) -> None:
    """Configure common test hooks for fuel-probe branch tests."""
    fuel_probe_module._wait_for_teleport_outcome = teleport_outcome
    fuel_targets_module._find_visible_fuel_target = lambda probe: make_container_state(
        101, 100, True, 300
    )
    fuel_targets_module._visible_fuel_requires_reposition = lambda probe, fuel_target: True
    fuel_targets_module._find_visible_fuel_landing_tile = lambda probe, fuel_target: (102, 100)


def _set_real_targeting_with_reposition(
    teleport_outcome: _WaitForTeleportOutcomeProtocol,
    probe: _ProbeHarness,
) -> None:
    """Configure REAL targeting helpers with terrain that forces reposition.

    Places a fuel container at (105, 100) inside the probe's default viewport
    (left=92, top=92, 16x16) and installs a rock wall at x=92..107 column 102.
    The real ``find_visible_fuel_target`` returns the container, the real
    ``is_collection_reachable_in_viewport`` returns False (BFS can't detour
    inside viewport bounds — see feedback_viewport_mechanics memory), so
    ``requires_reposition=True`` falls out of real logic. The real landing
    tile is (105, 100) — the container's own coord.

    Teleport outcome is still callback-driven because the state machine's
    terminal status depends on the test's specific scenario.
    """
    fuel_probe_module._wait_for_teleport_outcome = teleport_outcome
    terrain_provider = _make_rock_wall_terrain_provider()
    probe.world.terrain_map = terrain_provider()
    probe.world.terrain_map = terrain_provider()
    fuel_container = make_container_state(105, 100, True, 300)
    probe._world_state["containers"][f"{fuel_container['x']},{fuel_container['y']}"] = (
        fuel_container
    )


def _make_rock_wall_terrain_provider() -> Callable[[], TerrainMapProtocol | None]:
    def _provider() -> TerrainMapProtocol | None:
        return ground_terrain(rock_wall(102, range(92, 108)))

    return _provider
