"""Shared harness for the fuel-probe test modules.

``test_fuel_probe.py`` was 2,948 lines -- the largest file in the repo.
It is now six focused test modules over this harness and its scenario
sibling. This module holds the module handles that let a test reach
probe internals, the world/terrain/snapshot builders, the
outcome-callback factories, and the three probe subclasses the tests
drive. The two multi-step scenario runners are
:mod:`tests.action_lab._fuel_probe_scenarios`.

The hook-restore fixture every one of those modules relies on is
autouse in ``tests/action_lab/conftest.py``, next to its sibling
``restore_action_hooks`` -- a fixture cannot travel by import without
becoming an unused-name violation at each call site.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from pathlib import Path
from typing import (
    Literal,
    Protocol,
)

from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.fakes import InMemoryTerrainMap

from tankpit_bot._test_hooks import (
    PageProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.fuel_probe import FuelProbe
from tankpit_bot.action_lab.fuel_probe_attempt_contracts import (
    RunTrackedTeleportAttemptProtocol,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
)
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupTimeoutSizerProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    ContainerStateDict,
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import make_viewport_state
from tankpit_bot.types import (
    CapturedMessage,
)

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


_fuel_module_import = __import__("tankpit_bot.action_lab.fuel_probe", fromlist=["fuel_probe"])


fuel_probe_module: _FuelProbeModuleProtocol = _fuel_module_import


_fuel_targeting_module_import = __import__(
    "tankpit_bot.action_lab.fuel_targeting", fromlist=["fuel_targeting"]
)


fuel_targeting_module: _FuelTargetingModuleProtocol = _fuel_targeting_module_import


class _FuelTargetsModuleProtocol(Protocol):
    """Typed access to the patchable fuel-target module globals.

    Target selection and pickup-outcome waiting moved to
    :mod:`tankpit_bot.action_lab.fuel_probe_targets` when the 748-line
    probe module was split; the probe reaches them through that module,
    so this is where tests swap them.
    """

    _find_visible_fuel_target: Callable[[FuelProbe], ContainerStateDict | None]
    _visible_fuel_requires_reposition: Callable[[FuelProbe, ContainerStateDict], bool]
    _find_visible_fuel_landing_tile: Callable[
        [FuelProbe, ContainerStateDict], tuple[int, int] | None
    ]
    _wait_for_pickup_outcome: _WaitForPickupOutcomeProtocol
    # The public names the phase callables bind directly, since the
    # ``_for_phase`` bridges were pure delegation and were deleted.
    visible_fuel_requires_reposition: Callable[[FuelProbe, ContainerStateDict], bool]
    find_visible_fuel_landing_tile: Callable[
        [FuelProbe, ContainerStateDict], tuple[int, int] | None
    ]
    # Target selection reads terrain from THIS module's namespace, so a
    # scenario exercising the real helpers patches the provider here.
    get_terrain_map: Callable[[], TerrainMapProtocol | None]


_fuel_targets_import = __import__(
    "tankpit_bot.action_lab.fuel_probe_targets", fromlist=["fuel_probe_targets"]
)


fuel_targets_module: _FuelTargetsModuleProtocol = _fuel_targets_import


class _FuelProbeModuleProtocol(Protocol):
    """Typed access to patchable fuel probe module globals."""

    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol
    # The probe binds the shared teleport runner into its own
    # namespace, so tests swap it here rather than at its owner.
    run_tracked_teleport_attempt: RunTrackedTeleportAttemptProtocol
    run_tracked_pickup_phase: _RunTrackedPickupPhaseProtocol
    get_terrain_map: Callable[[], TerrainMapProtocol | None]
    FuelProbe: type[FuelProbe]


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


class _WaitForPickupOutcomeProtocol(Protocol):
    """Callable protocol for pickup outcome waiting."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]: ...


class _RunTrackedPickupPhaseProtocol(Protocol):
    """Callable protocol for the shared pickup-phase runner."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        attempt_label: str,
        target_x: int,
        target_y: int,
        current_x: int,
        current_y: int,
        fuel_before_pickup: int,
        pickup_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        get_completed_outcome: PickupImmediateOutcomeProtocol,
        wait_for_outcome: PickupOutcomeWaiterProtocol,
        compute_timeout: PickupTimeoutSizerProtocol,
    ) -> tuple[
        ActionPhaseCycleDict,
        ActionPhaseCycleDict,
        int,
        Literal["picked_up_fuel", "pickup_timeout"],
        int,
        int,
    ]: ...


class _FuelTargetingModuleProtocol(Protocol):
    """Typed access to patchable equipment_targeting globals — the SHARED
    fuel targeting module that ``_visible_fuel_requires_reposition`` and
    ``_find_visible_fuel_landing_tile`` ultimately call into. Both rely on
    ``get_terrain_map()`` from this module's namespace, so test scenarios
    that exercise the real targeting helpers must patch the terrain provider
    at BOTH ``fuel_probe_module`` AND this module."""

    get_terrain_map: Callable[[], TerrainMapProtocol | None]


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    """Build a sample page-client snapshot for fuel-probe fixtures."""
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
        map_fields={},
        world_collections={},
    )


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    """Build a world with one self tank."""
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=1,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=make_viewport_state(left=x - 8, top=y - 8, width=16, height=16),
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _terrain(passable: set[tuple[int, int]]) -> TerrainMapProtocol:
    return InMemoryTerrainMap.from_passable_set(passable)


def _build_wait_results(
    status: Literal[
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
    map_sync_result: int | None,
    radar_sync_result: int | None,
) -> list[int | None]:
    """Build ordered world-sync results for one probe scenario."""
    wait_results = [map_sync_result, radar_sync_result]
    if status == "reposition_map_sync_timeout":
        wait_results.append(None)
    elif status == "reposition_teleport_timeout":
        wait_results.append(1800)
    return wait_results


def _make_world_sync_waiter(
    wait_results: list[int | None],
) -> Callable[
    [
        action_session.WaitPageProtocol,
        action_session.BufferedWorldStateProviderProtocol,
        int,
        int,
    ],
    int | None,
]:
    """Return a deterministic world-sync waiter callback."""

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

    return _wait_for_world_sync


def _make_teleport_outcome_callback(
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
) -> _WaitForTeleportOutcomeProtocol:
    """Return a teleport outcome callback for one scenario."""

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
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
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            teleport_cycle_id,
            message_start_index,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        if teleport_status == "teleport_timeout":
            resolved_status: Literal[
                "landed_exact",
                "landed_offset",
                "map_sync_timeout",
                "teleport_timeout",
            ] = "teleport_timeout"
        elif teleport_status == "reposition_teleport_timeout":
            resolved_status = (
                "teleport_timeout"
                if target["label"].startswith("fuel_reposition_")
                else "landed_exact"
            )
        else:
            resolved_status = "landed_exact"
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status=resolved_status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=640,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=resolved_status == "landed_exact",
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    return _teleport_outcome


def _make_pickup_outcome_callback(
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> _WaitForPickupOutcomeProtocol:
    """Return a pickup outcome callback for one scenario."""

    def _pickup_outcome(
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
        _ = (
            page,
            probe,
            target_x,
            target_y,
            pickup_started_ms,
            fuel_before,
            timeout_ms,
        )
        if pickup_status is not None:
            return (pickup_status, 2000, 900)
        return ("pickup_timeout", 2000, 640)

    return _pickup_outcome


class _ProbeHarness(FuelProbe):
    """Fuel probe subclass with controllable world state."""

    def __init__(self, clock: ReplayClock) -> None:
        ws = WorldService()
        super().__init__(
            "https://tankpit.com/play",
            headless=True,
            prefer_account=False,
            world=ws,
        )
        self._world_state = _make_world(1000, 100, 100, 700)
        self._fake_page = ClockAdvancingPage(clock)
        self._cdp = StubSnapshotCDPSession()
        self._messages = []
        self.map_open_result = True
        self.teleport_result = True
        self.radar_result = True
        self.move_result = True
        self.move_calls: list[tuple[int, int]] = []

    def _require_page(self) -> PageProtocol:
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._world_state["self_state"]

    def open_map(self) -> bool:
        return self.map_open_result

    def teleport_to(self, x: int, y: int) -> bool:
        _ = (x, y)
        return self.teleport_result

    def use_radar(self) -> bool:
        return self.radar_result

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return self.move_result


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, FuelProbe):
    """Fuel probe subclass that stubs browser/bootstrap internals."""

    def __init__(self) -> None:
        FuelProbe.__init__(self, "https://tankpit.com/play", headless=False, prefer_account=True)
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 700)
        self._messages = []
        self.results: list[FuelProbeAttemptResultDict] = []

    def _probe_single_fuel_target(
        self,
        *,
        target: TeleportTargetDict,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
        teleport_strategy: Literal[
            "sync_before_teleport", "immediate_after_map_open"
        ] = "immediate_after_map_open",
    ) -> FuelProbeAttemptResultDict:
        _ = (
            target,
            map_sync_timeout_ms,
            teleport_timeout_ms,
            radar_timeout_ms,
            pickup_timeout_ms,
            settle_delay_ms,
            teleport_strategy,
        )
        result = self.results[0]
        if len(self.results) > 1:
            self.results = self.results[1:]
        return result


class _FakeFuelProbe(FuelProbe):
    def __init__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
        )

    def execute_probe(
        self,
        *,
        target_pickups: int,
        max_attempts: int,
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeSessionDict:
        return FuelProbeSessionDict(
            session_id="fuel-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            target_pickups=target_pickups,
            max_attempts=max_attempts,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            startup_timing={
                "game_ready_timestamp_ms": 100,
                "intel_ready_timestamp_ms": 150,
                "initial_sync_started_ms": 200,
                "initial_world_timestamp_ms": 400,
                "command_ready_timestamp_ms": 450,
                "first_attempt_started_ms": 500,
                "game_ready_to_intel_ready_ms": 50,
                "intel_ready_to_initial_world_ms": 250,
                "initial_world_to_command_ready_ms": 50,
                "command_ready_to_first_attempt_ms": 50,
            },
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None
