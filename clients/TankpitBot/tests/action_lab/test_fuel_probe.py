"""Tests for the live fuel action probe harness."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
    StubSnapshotCDPSession,
    WorldStateOverrideMixin,
)
from tests.action_lab.conftest import ground_terrain, rock_wall
from tests.conftest import FakeFileSystem
from tests.fakes import InMemoryTerrainMap
from typing_extensions import Unpack

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
    CDPSessionProtocol,
    PageProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbe,
    FuelProbeError,
    _clear_stale_radar_completion,
    _effective_pickup_timeout_ms,
    _find_visible_fuel_target,
    _format_visible_fuel_entries,
    _get_completed_pickup_outcome,
    _wait_for_pickup_outcome,
    format_fuel_probe_summary,
    run_fuel_probe,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
    decode_fuel_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.pickup_phase import (
    PickupImmediateOutcomeProtocol,
    PickupOutcomeWaiterProtocol,
    PickupPhaseError,
    PickupTimeoutSizerProtocol,
)
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterKwargs,
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.state import (
    ContainerStateDict,
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    coord_key,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


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


class _FuelProbeModuleProtocol(Protocol):
    """Typed access to patchable fuel probe module globals."""

    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol
    _find_visible_fuel_target: Callable[[FuelProbe, bool], ContainerStateDict | None]
    _visible_fuel_requires_reposition: Callable[[FuelProbe, ContainerStateDict], bool]
    _find_visible_fuel_landing_tile: Callable[
        [FuelProbe, ContainerStateDict], tuple[int, int] | None
    ]
    _wait_for_pickup_outcome: _WaitForPickupOutcomeProtocol
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


_fuel_module_import = __import__("tankpit_bot.action_lab.fuel_probe", fromlist=["fuel_probe"])
fuel_probe_module: _FuelProbeModuleProtocol = _fuel_module_import


class _FuelTargetingModuleProtocol(Protocol):
    """Typed access to patchable equipment_targeting globals — the SHARED
    fuel targeting module that ``_visible_fuel_requires_reposition`` and
    ``_find_visible_fuel_landing_tile`` ultimately call into. Both rely on
    ``get_terrain_map()`` from this module's namespace, so test scenarios
    that exercise the real targeting helpers must patch the terrain provider
    at BOTH ``fuel_probe_module`` AND this module."""

    get_terrain_map: Callable[[], TerrainMapProtocol | None]


_fuel_targeting_module_import = __import__(
    "tankpit_bot.action_lab.fuel_targeting", fromlist=["fuel_targeting"]
)
fuel_targeting_module: _FuelTargetingModuleProtocol = _fuel_targeting_module_import


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
        viewport=ViewportStateDict(left=x - 8, top=y - 8, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
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
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
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


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore patched hooks after each test."""
    original_get_time = action_hooks.get_current_time_ms
    original_check_radar = action_hooks.check_and_clear_radar_scan_complete
    original_drain = action_hooks.drain_buffered_messages
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_radar_sync = action_hooks.wait_for_radar_sync
    original_get_terrain_map = fuel_probe_module.get_terrain_map
    original_targeting_terrain = fuel_targeting_module.get_terrain_map
    original_wait_outcome = fuel_probe_module._wait_for_teleport_outcome
    original_find_visible = fuel_probe_module._find_visible_fuel_target
    original_requires_reposition = fuel_probe_module._visible_fuel_requires_reposition
    original_find_landing = fuel_probe_module._find_visible_fuel_landing_tile
    original_wait_pickup = fuel_probe_module._wait_for_pickup_outcome
    original_probe_class = fuel_probe_module.FuelProbe
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.check_and_clear_radar_scan_complete = original_check_radar
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.wait_for_world_sync = original_wait_sync
    action_hooks.wait_for_radar_sync = original_wait_radar_sync
    fuel_probe_module.get_terrain_map = original_get_terrain_map
    fuel_targeting_module.get_terrain_map = original_targeting_terrain
    fuel_probe_module._wait_for_teleport_outcome = original_wait_outcome
    fuel_probe_module._find_visible_fuel_target = original_find_visible
    fuel_probe_module._visible_fuel_requires_reposition = original_requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = original_find_landing
    fuel_probe_module._wait_for_pickup_outcome = original_wait_pickup
    fuel_probe_module.FuelProbe = original_probe_class


def test_clear_stale_radar_completion_drains_all_pending_flags() -> None:
    """Radar completion drain clears all leaked flags before a new scan."""
    completions = [True, True, False]

    def _check_radar_complete() -> bool:
        return completions.pop(0)

    action_hooks.check_and_clear_radar_scan_complete = _check_radar_complete

    _clear_stale_radar_completion()

    assert completions == []


def test_effective_pickup_timeout_scales_with_distance() -> None:
    """Pickup timeout grows with travel distance and never shrinks below base."""
    assert (
        _effective_pickup_timeout_ms(
            current_x=100,
            current_y=100,
            target_x=101,
            target_y=100,
            base_timeout_ms=3000,
        )
        == 3000
    )
    assert (
        _effective_pickup_timeout_ms(
            current_x=162,
            current_y=94,
            target_x=160,
            target_y=86,
            base_timeout_ms=3000,
        )
        == 6000
    )


def test_find_visible_fuel_target_returns_best_visible_container() -> None:
    """Fuel target selection chooses the visible high-volume fuel container."""
    probe = _ProbeHarness(ReplayClock(1000))
    world = probe.get_world_state()
    world["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    world["containers"][coord_key(102, 100)] = make_container_state(
        102,
        100,
        True,
        500,
        timestamp_ms=world["timestamp_ms"],
    )
    fuel_probe_module.get_terrain_map = lambda: _terrain({(100, 100), (101, 100), (102, 100)})

    fuel_target = _find_visible_fuel_target(probe)

    assert fuel_target == world["containers"][coord_key(102, 100)]


def test_format_visible_fuel_entries_returns_unavailable_without_terrain() -> None:
    """Visible-fuel diagnostics report unavailable without a terrain map."""
    probe = _ProbeHarness(ReplayClock(1000))
    fuel_probe_module.get_terrain_map = lambda: None

    summary = _format_visible_fuel_entries(probe, fuel_target=None)

    assert summary == "unavailable"


def test_format_visible_fuel_entries_returns_unavailable_without_self_state() -> None:
    """Visible-fuel diagnostics report unavailable without self state."""
    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["self_state"] = None
    fuel_probe_module.get_terrain_map = lambda: _terrain({(100, 100)})

    summary = _format_visible_fuel_entries(probe, fuel_target=None)

    assert summary == "unavailable"


def test_format_visible_fuel_entries_returns_none_when_no_visible_fuel_is_tracked() -> None:
    """Visible-fuel diagnostics exclude non-fuel and out-of-viewport containers."""
    probe = _ProbeHarness(ReplayClock(1000))
    world = probe.get_world_state()
    world["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        False,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    world["containers"][coord_key(200, 200)] = make_container_state(
        200,
        200,
        True,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    fuel_probe_module.get_terrain_map = lambda: _terrain({(100, 100), (101, 100), (200, 200)})

    summary = _format_visible_fuel_entries(probe, fuel_target=None)

    assert summary == "none"


def test_format_visible_fuel_entries_marks_stale_entries_and_truncates() -> None:
    """Visible-fuel diagnostics mark stale entries and truncate long summaries."""
    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state = _make_world(40001, 100, 100, 700)
    world = probe.get_world_state()
    passable_tiles = {(100, 100)}
    selected_target = make_container_state(101, 100, True, 300, timestamp_ms=0)
    world["containers"][coord_key(101, 100)] = selected_target
    passable_tiles.add((101, 100))
    stale_positions = [
        (102, 100),
        (103, 100),
        (104, 100),
        (105, 100),
        (106, 100),
        (107, 100),
        (101, 101),
        (102, 101),
    ]
    for x, y in stale_positions:
        world["containers"][coord_key(x, y)] = make_container_state(
            x,
            y,
            True,
            300,
            timestamp_ms=0,
        )
        passable_tiles.add((x, y))
    fuel_probe_module.get_terrain_map = lambda: _terrain(passable_tiles)

    summary = _format_visible_fuel_entries(probe, fuel_target=selected_target)

    assert "reason=stale actionable=False selected=True" in summary
    assert "...+1 more" in summary


def test_find_visible_fuel_target_requires_terrain_and_self_state() -> None:
    """Fuel target selection raises when required state is missing."""
    probe = _ProbeHarness(ReplayClock(1000))
    fuel_probe_module.get_terrain_map = lambda: None
    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        _find_visible_fuel_target(probe)

    fuel_probe_module.get_terrain_map = lambda: _terrain({(100, 100)})
    probe._world_state["self_state"] = None
    with pytest.raises(FuelProbeError, match="self state is unavailable"):
        _find_visible_fuel_target(probe)


def test_wait_for_pickup_outcome_detects_fuel_gain_and_disappearance() -> None:
    """Pickup wait succeeds on fuel gain or container disappearance."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = probe._fake_page
    worlds = [_make_world(1000, 100, 100, 300), _make_world(1100, 100, 100, 450)]
    for world in worlds:
        world["containers"][coord_key(101, 100)] = make_container_state(
            101,
            100,
            True,
            300,
            timestamp_ms=world["timestamp_ms"],
        )
    probe._world_state = worlds[0]

    def _advance() -> None:
        if len(worlds) > 1:
            worlds.pop(0)
        probe._world_state = worlds[0]

    page.on_wait = _advance
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider: 0

    status, completed_ms, fuel_after = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=300,
        timeout_ms=1000,
    )

    assert (status, completed_ms, fuel_after) == ("picked_up_fuel", 1100, 450)

    probe._world_state = _make_world(1000, 100, 100, 700)
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )

    def _remove_container(provider: BufferedMessageSourceProtocol) -> int:
        _ = provider
        probe.get_world_state()["containers"].pop(coord_key(101, 100), None)
        return 1

    action_hooks.drain_buffered_messages = _remove_container

    disappeared = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=1000,
    )

    assert disappeared == ("pickup_timeout", 2000, 450)


def test_wait_for_pickup_outcome_times_out_and_handles_missing_self_state() -> None:
    """Pickup wait handles timeout and missing-self-state failures."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = probe._fake_page
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider: 0

    timed_out = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=150,
    )

    assert timed_out == ("pickup_timeout", 1200, 700)

    def _clear_self(provider: BufferedMessageSourceProtocol) -> int:
        _ = provider
        probe.get_world_state()["self_state"] = None
        return 1

    probe = _ProbeHarness(clock)
    action_hooks.drain_buffered_messages = _clear_self
    with pytest.raises(FuelProbeError, match="self state disappeared while waiting"):
        _wait_for_pickup_outcome(
            page,
            probe,
            target_x=101,
            target_y=100,
            pickup_started_ms=1000,
            fuel_before=700,
            timeout_ms=1000,
        )

    probe = _ProbeHarness(clock)
    probe._world_state["self_state"] = None
    action_hooks.drain_buffered_messages = lambda provider: 0
    with pytest.raises(FuelProbeError, match="self state disappeared after fuel pickup timeout"):
        _wait_for_pickup_outcome(
            page,
            probe,
            target_x=101,
            target_y=100,
            pickup_started_ms=1000,
            fuel_before=700,
            timeout_ms=0,
        )


def test_get_completed_pickup_outcome_detects_pickup_and_missing_self_state() -> None:
    """Immediate pickup helper detects queued pickup events and validates self state."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    probe._world_state["self_state"] = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=850,
        leaderboard_position=1,
    )

    completed = _get_completed_pickup_outcome(
        probe,
        target_x=101,
        target_y=100,
        fuel_before=700,
    )

    assert completed == ("picked_up_fuel", 1000, 850)

    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    probe._world_state["self_state"] = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=700,
        leaderboard_position=1,
    )
    probe._world_state["containers"].pop(coord_key(101, 100), None)

    assert (
        _get_completed_pickup_outcome(
            probe,
            target_x=101,
            target_y=100,
            fuel_before=700,
        )
        is None
    )

    probe = _ProbeHarness(ReplayClock(1000))
    probe._world_state["self_state"] = None

    with pytest.raises(FuelProbeError, match="self state disappeared while waiting"):
        _get_completed_pickup_outcome(
            probe,
            target_x=101,
            target_y=100,
            fuel_before=700,
        )


def test_run_pickup_attempt_converts_pickup_phase_error() -> None:
    """Fuel pickup wrapper converts shared pickup-phase failures."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    fuel_target = make_container_state(101, 100, True, 300)
    pickup_attr = "run_tracked_pickup_phase"
    original_run_pickup = fuel_probe_module.run_tracked_pickup_phase

    def _raise_pickup_phase_error(
        page_arg: action_session.WaitPageProtocol,
        probe_arg: FuelProbe,
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
    ]:
        _ = (
            page_arg,
            probe_arg,
            attempt_label,
            target_x,
            target_y,
            current_x,
            current_y,
            fuel_before_pickup,
            pickup_timeout_ms,
            dispatch_failure_error,
            get_completed_outcome,
            wait_for_outcome,
            compute_timeout,
        )
        raise PickupPhaseError("shared pickup failure")

    setattr(fuel_probe_module, pickup_attr, _raise_pickup_phase_error)
    try:
        with pytest.raises(FuelProbeError, match="shared pickup failure"):
            probe._run_pickup_attempt(
                page=page,
                target=target,
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1200,
                teleport_started_ms=1300,
                radar_started_ms=1600,
                radar_sync_timestamp_ms=1700,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_timeout_ms=3000,
                fuel_before=900,
                teleport_result=TeleportAttemptResultDict(
                    target=target,
                    teleport_cycle_id=1,
                    status="landed_exact",
                    map_open_started_ms=1000,
                    map_sync_timestamp_ms=1200,
                    teleport_started_ms=1300,
                    completion_timestamp_ms=1500,
                    map_sync_elapsed_ms=200,
                    teleport_elapsed_ms=200,
                    fuel_before=900,
                    fuel_after=840,
                    world_timestamp_before=950,
                    world_timestamp_after=1450,
                    landed_signal_received=True,
                    landed_x=124,
                    landed_y=100,
                    message_start_index=0,
                    message_end_index=0,
                    page_snapshots=[],
                ),
                fuel_target=fuel_target,
                message_start_index=0,
                teleport_cycle_ids=[1],
                radar_cycle_id=2,
                decision_basis=None,
                snapshot_before=_snapshot(1000),
                capture_snapshot=lambda: _snapshot(1900),
            )
    finally:
        setattr(fuel_probe_module, pickup_attr, original_run_pickup)


def test_format_fuel_probe_summary_counts_statuses() -> None:
    """Fuel probe summary reports all terminal status buckets."""
    session = FuelProbeSessionDict(
        session_id="fuel-session",
        start_timestamp_ms=100,
        end_timestamp_ms=200,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        target_pickups=3,
        max_attempts=6,
        capture_session_path="fuel_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
        attempts=[
            FuelProbeAttemptResultDict(
                target={"label": "a", "x": 1, "y": 1},
                teleport_cycle_ids=[1],
                radar_cycle_id=None,
                move_cycle_id=None,
                pickup_cycle_id=None,
                status="picked_up_fuel",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=1400,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=1500,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=700,
                landed_signal_received=True,
                landed_x=1,
                landed_y=1,
                fuel_target_x=2,
                fuel_target_y=1,
                fuel_target_volume=200,
                phase_overlaps=[],
                decision_basis=None,
                snapshot_before=_snapshot(0),
                snapshot_after=_snapshot(0),
                message_start_index=0,
                message_end_index=1,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "b", "x": 2, "y": 2},
                teleport_cycle_ids=[1],
                radar_cycle_id=None,
                move_cycle_id=None,
                pickup_cycle_id=None,
                status="no_fuel_visible",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=1400,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=500,
                landed_signal_received=True,
                landed_x=2,
                landed_y=2,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                phase_overlaps=[],
                decision_basis=None,
                snapshot_before=_snapshot(0),
                snapshot_after=_snapshot(0),
                message_start_index=1,
                message_end_index=2,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "c", "x": 3, "y": 3},
                teleport_cycle_ids=[1],
                radar_cycle_id=None,
                move_cycle_id=None,
                pickup_cycle_id=None,
                status="radar_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=None,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=450,
                landed_signal_received=True,
                landed_x=3,
                landed_y=3,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                phase_overlaps=[],
                decision_basis=None,
                snapshot_before=_snapshot(0),
                snapshot_after=_snapshot(0),
                message_start_index=2,
                message_end_index=3,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "d", "x": 4, "y": 4},
                teleport_cycle_ids=[1],
                radar_cycle_id=None,
                move_cycle_id=None,
                pickup_cycle_id=None,
                status="map_sync_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=None,
                teleport_started_ms=None,
                radar_started_ms=None,
                radar_sync_timestamp_ms=None,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=500,
                landed_signal_received=False,
                landed_x=4,
                landed_y=4,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                phase_overlaps=[],
                decision_basis=None,
                snapshot_before=_snapshot(0),
                snapshot_after=_snapshot(0),
                message_start_index=3,
                message_end_index=4,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "e", "x": 5, "y": 5},
                teleport_cycle_ids=[1],
                radar_cycle_id=None,
                move_cycle_id=None,
                pickup_cycle_id=None,
                status="teleport_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=None,
                radar_sync_timestamp_ms=None,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=450,
                landed_signal_received=False,
                landed_x=5,
                landed_y=5,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                phase_overlaps=[],
                decision_basis=None,
                snapshot_before=_snapshot(0),
                snapshot_after=_snapshot(0),
                message_start_index=4,
                message_end_index=5,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "f", "x": 6, "y": 6},
                teleport_cycle_ids=[1],
                radar_cycle_id=None,
                move_cycle_id=None,
                pickup_cycle_id=None,
                status="pickup_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=1400,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=1500,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=450,
                landed_signal_received=True,
                landed_x=6,
                landed_y=6,
                fuel_target_x=7,
                fuel_target_y=6,
                fuel_target_volume=200,
                phase_overlaps=[],
                decision_basis=None,
                snapshot_before=_snapshot(0),
                snapshot_after=_snapshot(0),
                message_start_index=5,
                message_end_index=6,
            ),
        ],
    )

    summary = format_fuel_probe_summary(session)

    assert "attempts=6" in summary
    assert "target_pickups=3" in summary
    assert "picked_up_fuel=1" in summary
    assert "no_fuel_visible=1" in summary
    assert "radar_timeout=1" in summary
    assert "map_sync_timeout=1" in summary
    assert "teleport_timeout=1" in summary
    assert "pickup_timeout=1" in summary
    assert "session_to_initial_sync_ms=300" in summary
    assert "initial_sync_to_command_ready_ms=60" in summary


def _run_probe_single_target_scenario(
    *,
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
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
    radar_sync_result: int | None,
    fuel_target: ContainerStateDict | None,
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> FuelProbeAttemptResultDict:
    """Run one configured single-target probe scenario.

    Runs the REAL targeting helpers (``_find_visible_fuel_target``,
    ``_visible_fuel_requires_reposition``, ``_find_visible_fuel_landing_tile``)
    against a harness whose world state holds the configured fuel container
    and terrain set up to produce the desired branch:

    * reposition scenarios get a rock-wall at x=102 that spans the viewport
      height — the real BFS finds no detour, ``requires_reposition`` returns
      True, and ``find_landing_tile`` returns the container coord (since the
      container's tile is GROUND).
    * all other scenarios get fully-passable terrain.
    * ``no_fuel_visible`` simply omits the container; real finder returns None.

    Teleport-outcome and pickup-outcome are still callbacks because they drive
    the state machine's terminal status — those are the leaves the test is
    exercising. Everything between the test and those leaves is real code.
    """
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = _build_wait_results(status, map_sync_result, radar_sync_result)

    wait_for_world_sync = _make_world_sync_waiter(wait_results)

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        return wait_for_world_sync(page, provider, started_ms, timeout_ms)

    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.wait_for_radar_sync = _wait_for_world_sync
    fuel_probe_module._wait_for_teleport_outcome = _make_teleport_outcome_callback(teleport_status)
    fuel_probe_module._wait_for_pickup_outcome = _make_pickup_outcome_callback(pickup_status)

    is_reposition_scenario = status in {
        "reposition_map_sync_timeout",
        "reposition_teleport_timeout",
    }

    def _reposition_blocking_terrain() -> TerrainMapProtocol:
        return ground_terrain(rock_wall(102, range(92, 108)))

    terrain_provider: Callable[[], TerrainMapProtocol | None] = (
        _reposition_blocking_terrain if is_reposition_scenario else ground_terrain
    )
    fuel_probe_module.get_terrain_map = terrain_provider
    fuel_targeting_module.get_terrain_map = terrain_provider

    if fuel_target is not None:
        target_key = coord_key(fuel_target["x"], fuel_target["y"])
        probe._world_state["containers"][target_key] = fuel_target

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=250,
        teleport_strategy="sync_before_teleport",
    )

    assert result["status"] == status
    assert result["target"] == target
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 0
    assert probe._fake_page.waits[-1] == 250.0
    if status in {"picked_up_fuel", "pickup_timeout"}:
        assert probe.move_calls == [(101, 100)]
    else:
        assert probe.move_calls == []
    return result


@pytest.mark.parametrize(
    (
        "status",
        "map_sync_result",
        "teleport_status",
        "radar_sync_result",
        "fuel_target",
        "pickup_status",
    ),
    [
        ("map_sync_timeout", None, None, None, None, None),
        ("teleport_timeout", 1200, "teleport_timeout", None, None, None),
        ("radar_timeout", 1200, "landed_exact", None, None, None),
        (
            "reposition_map_sync_timeout",
            1200,
            "landed_exact",
            1600,
            make_container_state(105, 100, True, 300),
            None,
        ),
        (
            "reposition_teleport_timeout",
            1200,
            "reposition_teleport_timeout",
            1600,
            make_container_state(105, 100, True, 300),
            None,
        ),
        ("no_fuel_visible", 1200, "landed_exact", 1600, None, None),
        (
            "picked_up_fuel",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            "picked_up_fuel",
        ),
        (
            "pickup_timeout",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            "pickup_timeout",
        ),
    ],
)
def test_probe_single_target_records_terminal_statuses(
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
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
    radar_sync_result: int | None,
    fuel_target: ContainerStateDict | None,
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> None:
    """Single-target probe records all terminal outcomes."""
    _run_probe_single_target_scenario(
        status=status,
        map_sync_result=map_sync_result,
        teleport_status=teleport_status,
        radar_sync_result=radar_sync_result,
        fuel_target=fuel_target,
        pickup_status=pickup_status,
    )


def test_probe_single_target_rejects_impossible_map_sync_timeout_teleport_outcome() -> None:
    """Fuel probe rejects a teleport outcome that reports map-sync timeout after sync success."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
    action_hooks.wait_for_radar_sync = lambda page, provider, started_ms, timeout_ms: 1200

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
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="map_sync_timeout",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=700,
            fuel_after=650,
            world_timestamp_before=1000,
            world_timestamp_after=1450,
            landed_signal_received=False,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome

    with pytest.raises(
        TeleportProbeError,
        match="teleport outcome reported impossible map_sync_timeout",
    ):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_rejects_missing_tracked_teleport_result() -> None:
    """Fuel probe rejects a tracked attempt that never produced a teleport result."""
    from tankpit_bot.action_lab import fuel_probe as fuel_probe_runtime

    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    original_attempt_runner = fuel_probe_runtime.run_tracked_teleport_attempt

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return TeleportPageSnapshotDict(
            phase=phase,
            timestamp_ms=1000,
            client_present=True,
            map_visible=False,
            client_state=1,
            client_busy=False,
            pending_actions=0,
            heartbeat_age_ms=1,
            last_page_client_send_age_ms=2,
            last_bot_send_age_ms=3,
            ws_ready_state=1,
            current_send_label=None,
            sent_frame_meta_queue_length=0,
            self_fields={},
            world_fields={},
            map_fields={},
            world_collections={},
        )

    def _run_attempt(
        page: action_session.WaitPageProtocol,
        probe: TeleportAttemptProbeProtocol,
        target: TeleportTargetDict,
        *,
        cdp: CDPSessionProtocol | None,
        attempt_label: str,
        fuel_before: int,
        world_timestamp_before: int,
        send_acquisition_command: Callable[[], bool],
        acquisition_command_name: str,
        capture_before_map_open: bool,
        wait_for_acquisition_sync: bool,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
        unexpected_result_error: type[Exception],
        unexpected_result_message: str,
        reset_to_idle_before_start: bool = True,
    ) -> TrackedTeleportAttempt:
        _ = (
            page,
            probe,
            target,
            cdp,
            attempt_label,
            fuel_before,
            world_timestamp_before,
            send_acquisition_command,
            acquisition_command_name,
            capture_before_map_open,
            wait_for_acquisition_sync,
            acquisition_timeout_ms,
            teleport_timeout_ms,
            wait_for_outcome,
            dispatch_failure_error,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
            unavailable_error,
            unavailable_message,
            unexpected_result_error,
            unexpected_result_message,
            reset_to_idle_before_start,
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=1, started_ms=1000),
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1200,
            page_snapshots=[],
            capture_page_snapshot=_capture_page_snapshot,
            teleport_result=None,
            teleport_started_ms=None,
        )

    fuel_probe_runtime.run_tracked_teleport_attempt = _run_attempt
    try:
        with pytest.raises(FuelProbeError, match="fuel attempt ended before teleport dispatch"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
                teleport_strategy="sync_before_teleport",
            )
    finally:
        fuel_probe_runtime.run_tracked_teleport_attempt = original_attempt_runner


def test_resolve_fuel_target_after_radar_rejects_missing_tracked_reposition_result() -> None:
    """Fuel target resolution rejects a tracked reposition without a teleport result."""
    from tankpit_bot.action_lab import fuel_target_phase

    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)
    page = ClockAdvancingPage(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    fuel_target = make_container_state(101, 100, True, 300)
    original_attempt_runner = fuel_target_phase.run_reposition_attempt

    def _requires_reposition(
        probe: fuel_target_phase.FuelTargetPhaseProbeProtocol,
        fuel_target: ContainerStateDict,
    ) -> bool:
        _ = (probe, fuel_target)
        return True

    def _landing_tile(
        probe: fuel_target_phase.FuelTargetPhaseProbeProtocol,
        fuel_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (probe, fuel_target)
        return (102, 100)

    def _make_reposition_target(target_x: int, target_y: int) -> TeleportTargetDict:
        return TeleportTargetDict(
            label=f"fuel_reposition_{target_x}_{target_y}",
            x=target_x,
            y=target_y,
        )

    def _teleport_strategy_requires_map_sync(
        strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
    ) -> bool:
        return strategy == "sync_before_teleport"

    def _wait_for_teleport_outcome_adapter(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
    ) -> TeleportAttemptResultDict:
        _ = (page, provider)
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=kwargs["teleport_cycle_id"],
            status="landed_exact",
            map_open_started_ms=kwargs["map_open_started_ms"],
            map_sync_timestamp_ms=kwargs["map_sync_timestamp_ms"],
            teleport_started_ms=kwargs["teleport_started_ms"],
            completion_timestamp_ms=2200,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=kwargs["fuel_before"],
            fuel_after=840,
            world_timestamp_before=kwargs["world_timestamp_before"],
            world_timestamp_after=2150,
            landed_signal_received=True,
            landed_x=102,
            landed_y=100,
            message_start_index=kwargs["message_start_index"],
            message_end_index=kwargs["message_start_index"],
            page_snapshots=kwargs["page_snapshots"],
        )

    wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol = _wait_for_teleport_outcome_adapter

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return TeleportPageSnapshotDict(
            phase=phase,
            timestamp_ms=2000,
            client_present=True,
            map_visible=False,
            client_state=1,
            client_busy=False,
            pending_actions=0,
            heartbeat_age_ms=1,
            last_page_client_send_age_ms=2,
            last_bot_send_age_ms=3,
            ws_ready_state=1,
            current_send_label=None,
            sent_frame_meta_queue_length=0,
            self_fields={},
            world_fields={},
            map_fields={},
            world_collections={},
        )

    def _run_attempt(
        page: action_session.WaitPageProtocol,
        probe: TeleportAttemptProbeProtocol,
        target: TeleportTargetDict,
        *,
        cdp: CDPSessionProtocol | None,
        attempt_label: str,
        fuel_before: int,
        world_timestamp_before: int,
        send_acquisition_command: Callable[[], bool],
        acquisition_command_name: str,
        capture_before_map_open: bool,
        wait_for_acquisition_sync: bool,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
        unexpected_result_error: type[Exception],
        unexpected_result_message: str,
        reset_to_idle_before_start: bool = True,
    ) -> TrackedTeleportAttempt:
        _ = (
            page,
            probe,
            target,
            cdp,
            attempt_label,
            fuel_before,
            world_timestamp_before,
            send_acquisition_command,
            acquisition_command_name,
            capture_before_map_open,
            wait_for_acquisition_sync,
            acquisition_timeout_ms,
            teleport_timeout_ms,
            wait_for_outcome,
            dispatch_failure_error,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
            unavailable_error,
            unavailable_message,
            unexpected_result_error,
            unexpected_result_message,
            reset_to_idle_before_start,
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=3, started_ms=2000),
            acquisition_started_ms=2000,
            acquisition_sync_timestamp_ms=2200,
            page_snapshots=[],
            capture_page_snapshot=_capture_page_snapshot,
            teleport_result=None,
            teleport_started_ms=None,
        )

    fuel_target_phase.run_reposition_attempt = _run_attempt
    try:
        with pytest.raises(FuelProbeError, match="fuel reposition ended before teleport dispatch"):
            fuel_target_phase.resolve_fuel_target_after_radar(
                page,
                probe,
                cdp=probe._cdp,
                target=target,
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1200,
                teleport_started_ms=1300,
                radar_started_ms=1600,
                radar_sync_timestamp_ms=1700,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                fuel_before=900,
                teleport_result=TeleportAttemptResultDict(
                    target=target,
                    teleport_cycle_id=1,
                    status="landed_exact",
                    map_open_started_ms=1000,
                    map_sync_timestamp_ms=1200,
                    teleport_started_ms=1300,
                    completion_timestamp_ms=1500,
                    map_sync_elapsed_ms=200,
                    teleport_elapsed_ms=200,
                    fuel_before=900,
                    fuel_after=840,
                    world_timestamp_before=950,
                    world_timestamp_after=1450,
                    landed_signal_received=True,
                    landed_x=124,
                    landed_y=100,
                    message_start_index=0,
                    message_end_index=0,
                    page_snapshots=[],
                ),
                message_start_index=0,
                teleport_cycle_ids=[1],
                radar_cycle_id=2,
                teleport_strategy="sync_before_teleport",
                snapshot_before=_snapshot(1000),
                capture_snapshot=lambda: _snapshot(1900),
                terrain_provider=lambda: None,
                find_visible_target=lambda current_probe, allow_unreachable: fuel_target,
                requires_reposition=_requires_reposition,
                find_landing_tile=_landing_tile,
                get_phase_overlaps=probe._get_attempt_phase_overlaps,
                build_no_fuel_visible_result=probe._build_no_fuel_visible_result,
                build_reposition_map_sync_timeout_result=(
                    probe._build_reposition_map_sync_timeout_result
                ),
                build_reposition_teleport_timeout_result=(
                    probe._build_reposition_teleport_timeout_result
                ),
                make_reposition_target=_make_reposition_target,
                wait_for_teleport_outcome=wait_for_teleport_outcome,
                teleport_strategy_requires_map_sync=_teleport_strategy_requires_map_sync,
                no_landing_tile_error=FuelProbeError,
                dispatch_failure_error=FuelProbeError,
                unavailable_error=FuelProbeError,
                unexpected_result_error=TeleportProbeError,
                unavailable_message="cdp session is unavailable",
                no_landing_tile_message="visible fuel target has no teleport landing tile",
                impossible_result_message=(
                    "teleport outcome reported impossible map_sync_timeout during fuel reposition"
                ),
                acquisition_dispatch_failure_message=(
                    "map_open command dispatch failed during fuel reposition"
                ),
                teleport_dispatch_failure_message=(
                    "teleport command dispatch failed during fuel reposition"
                ),
            )
    finally:
        fuel_target_phase.run_reposition_attempt = original_attempt_runner


def test_probe_single_target_repositions_for_blocked_visible_fuel() -> None:
    """Single-target fuel probe can reposition to a blocked visible fuel container."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = [1200, 1600, 1800]

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.wait_for_radar_sync = _wait_for_world_sync

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
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        if target["label"].startswith("fuel_reposition_"):
            landed_x = 102
            landed_y = 100
            fuel_after = 620
        else:
            landed_x = 124
            landed_y = 100
            fuel_after = 640
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=landed_x,
            landed_y=landed_y,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _find_target(
        current_probe: FuelProbe,
        allow_unreachable: bool,
    ) -> ContainerStateDict | None:
        _ = (current_probe, allow_unreachable)
        fuel_target = make_container_state(101, 100, True, 300)
        current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
        return fuel_target

    def _requires_reposition(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> bool:
        _ = (current_probe, current_target)
        return True

    def _find_landing(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (current_probe, current_target)
        return (102, 100)

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
        return ("picked_up_fuel", 2000, 900)

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome
    fuel_probe_module._find_visible_fuel_target = _find_target
    fuel_probe_module._visible_fuel_requires_reposition = _requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = _find_landing
    fuel_probe_module._wait_for_pickup_outcome = _pickup_outcome

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
    )

    assert result["status"] == "picked_up_fuel"
    assert result["reposition_map_open_started_ms"] == 1000
    assert result["reposition_map_sync_timestamp_ms"] is None
    assert result["reposition_teleport_started_ms"] == 1000
    assert result["landed_x"] == 102
    assert result["landed_y"] == 100
    assert result["pickup_started_ms"] == 1000
    assert probe.move_calls == [(101, 100)]


def test_probe_single_target_skips_move_when_pickup_already_completed() -> None:
    """Single-target probe does not enqueue move after an immediate fuel pickup."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = [1200, 1600]

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

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
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=640,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    drain_calls = 0

    def _pickup_before_move(provider: BufferedMessageSourceProtocol) -> int:
        nonlocal drain_calls
        _ = provider
        drain_calls += 1
        if drain_calls < 2:
            return 0
        probe.get_world_state()["self_state"] = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        probe.get_world_state()["containers"].pop(coord_key(101, 100), None)
        return 1

    fuel_target = make_container_state(101, 100, True, 300)
    probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
    fuel_probe_module.get_terrain_map = ground_terrain
    fuel_targeting_module.get_terrain_map = ground_terrain
    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.wait_for_radar_sync = _wait_for_world_sync
    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome
    action_hooks.drain_buffered_messages = _pickup_before_move

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
    )

    assert result["status"] == "picked_up_fuel"
    assert result["fuel_after"] == 900
    assert probe.move_calls == []


def test_finalize_attempt_delay_skips_wait_for_zero_delay() -> None:
    """Fuel probe does not wait when settle delay is disabled."""
    clock = ReplayClock(1000)
    probe = _ProbeHarness(clock)

    probe._finalize_attempt_delay(probe._fake_page, settle_delay_ms=0)

    assert probe._fake_page.waits == []


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


def test_probe_single_target_raises_when_dispatch_fails() -> None:
    """Single-target probe raises on command dispatch failures."""
    from tankpit_bot.sniffer.world_state import register_room_image, set_selected_room

    original_path_exists = core_hooks.path_exists
    original_load_terrain_map = core_hooks.load_terrain_map
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    try:
        register_room_image("1", "field01.gif")
        set_selected_room("1")
        core_hooks.path_exists = lambda path: True
        core_hooks.load_terrain_map = lambda path: InMemoryTerrainMap()

        probe = _ProbeHarness(clock)
        probe.map_open_result = False
        with pytest.raises(FuelProbeError, match="map_open command dispatch failed"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )

        action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
        action_hooks.wait_for_radar_sync = lambda page, provider, started_ms, timeout_ms: 1200
        probe = _ProbeHarness(clock)
        probe.teleport_result = False
        with pytest.raises(FuelProbeError, match="teleport command dispatch failed"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )

        probe = _ProbeHarness(clock)

        def _landed_teleport_outcome(
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
                [
                    Literal[
                        "before_map_open",
                        "before_teleport",
                        "after_map_data",
                        "landed",
                        "timeout",
                    ]
                ],
                TeleportPageSnapshotDict,
            ],
        ) -> TeleportAttemptResultDict:
            _ = (
                page,
                provider,
                message_start_index,
                timeout_ms,
                page_snapshots,
                capture_page_snapshot,
            )
            return TeleportAttemptResultDict(
                target=target,
                teleport_cycle_id=teleport_cycle_id,
                status="landed_exact",
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                completion_timestamp_ms=1500,
                map_sync_elapsed_ms=200,
                teleport_elapsed_ms=200,
                fuel_before=fuel_before,
                fuel_after=650,
                world_timestamp_before=world_timestamp_before,
                world_timestamp_after=1450,
                landed_signal_received=True,
                landed_x=124,
                landed_y=100,
                message_start_index=0,
                message_end_index=0,
                page_snapshots=[],
            )

        fuel_probe_module._wait_for_teleport_outcome = _landed_teleport_outcome
        probe.radar_result = False
        with pytest.raises(FuelProbeError, match="radar command dispatch failed"):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )

        probe = _ProbeHarness(clock)

        def _find_target(
            current_probe: FuelProbe,
            allow_unreachable: bool,
        ) -> ContainerStateDict | None:
            _ = (current_probe, allow_unreachable)
            fuel_target = make_container_state(101, 100, True, 300)
            current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
            return fuel_target

        fuel_probe_module._find_visible_fuel_target = _find_target
        probe.move_result = False
        with pytest.raises(
            FuelProbeError,
            match="move_to command dispatch failed during fuel collection",
        ):
            probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )
    finally:
        core_hooks.path_exists = original_path_exists
        core_hooks.load_terrain_map = original_load_terrain_map


def test_execute_probe_raises_for_invalid_limits_and_missing_playwright() -> None:
    """Fuel probe execute validates pickup limits and Playwright presence."""
    probe = _ProbeHarness(ReplayClock(1000))
    with pytest.raises(ValueError, match="target_pickups must be positive"):
        probe.execute_probe(
            target_pickups=0,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        probe.execute_probe(
            target_pickups=1,
            max_attempts=0,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    with pytest.raises(ValueError, match="max_attempts must be at least target_pickups"):
        probe.execute_probe(
            target_pickups=2,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                target_pickups=1,
                max_attempts=1,
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_collects_attempts_and_requires_terrain() -> None:
    """Fuel probe execute collects attempts and rejects missing terrain."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    session_browser = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = session_browser.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_124_100", "x": 124, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=1500,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=900,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            fuel_target_x=125,
            fuel_target_y=100,
            fuel_target_volume=300,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=0,
            message_end_index=1,
        )
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {
            (115, 99),
            (115, 100),
            (115, 101),
            (116, 99),
            (116, 100),
            (116, 101),
            (117, 99),
            (117, 100),
            (117, 101),
        }
    )

    session = probe.execute_probe(
        target_pickups=1,
        max_attempts=1,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 1
    assert session["spawn_x"] == 100
    assert session["target_pickups"] == 1
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert session_browser.browser_type.launches == [False]

    fuel_probe_module.get_terrain_map = lambda: None
    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        probe.execute_probe(
            target_pickups=1,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )


def test_execute_probe_continues_after_pickup_until_target_pickups_reached() -> None:
    """Fuel probe execute keeps probing after a pickup until target pickups are met."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    session_browser = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = session_browser.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_116_100", "x": 116, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=1500,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=850,
            landed_signal_received=True,
            landed_x=116,
            landed_y=100,
            fuel_target_x=117,
            fuel_target_y=100,
            fuel_target_volume=150,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=0,
            message_end_index=1,
        ),
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_117_100", "x": 117, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1700,
            map_sync_timestamp_ms=1800,
            teleport_started_ms=1900,
            radar_started_ms=2000,
            radar_sync_timestamp_ms=2100,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=2200,
            completion_timestamp_ms=2300,
            fuel_before=850,
            fuel_after=1000,
            landed_signal_received=True,
            landed_x=117,
            landed_y=100,
            fuel_target_x=118,
            fuel_target_y=100,
            fuel_target_volume=150,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=2,
            message_end_index=3,
        ),
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {(x, y) for x in range(0, 201) for y in range(0, 201)}
    )

    session = probe.execute_probe(
        target_pickups=2,
        max_attempts=3,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert [attempt["status"] for attempt in session["attempts"]] == [
        "picked_up_fuel",
        "picked_up_fuel",
    ]
    assert session["target_pickups"] == 2


def test_execute_probe_continues_after_miss_until_pickup_succeeds() -> None:
    """Fuel probe execute keeps probing after a miss until a later pickup succeeds."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    session_browser = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = session_browser.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_116_100", "x": 116, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="no_fuel_visible",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=None,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=650,
            landed_signal_received=True,
            landed_x=116,
            landed_y=100,
            fuel_target_x=None,
            fuel_target_y=None,
            fuel_target_volume=None,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=0,
            message_end_index=1,
        ),
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_117_100", "x": 117, "y": 100},
            teleport_cycle_ids=[1],
            radar_cycle_id=None,
            move_cycle_id=None,
            pickup_cycle_id=None,
            status="picked_up_fuel",
            map_open_started_ms=1700,
            map_sync_timestamp_ms=1800,
            teleport_started_ms=1900,
            radar_started_ms=2000,
            radar_sync_timestamp_ms=2100,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=2200,
            completion_timestamp_ms=2300,
            fuel_before=650,
            fuel_after=900,
            landed_signal_received=True,
            landed_x=117,
            landed_y=100,
            fuel_target_x=118,
            fuel_target_y=100,
            fuel_target_volume=250,
            phase_overlaps=[],
            decision_basis=None,
            snapshot_before=_snapshot(0),
            snapshot_after=_snapshot(0),
            message_start_index=2,
            message_end_index=3,
        ),
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {(x, y) for x in range(0, 201) for y in range(0, 201)}
    )

    session = probe.execute_probe(
        target_pickups=1,
        max_attempts=3,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert [attempt["status"] for attempt in session["attempts"]] == [
        "no_fuel_visible",
        "picked_up_fuel",
    ]
    assert session["target_pickups"] == 1


def test_run_fuel_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    """Fuel probe runner writes both summary JSON and raw capture output."""
    original_probe_class = fuel_probe_module.FuelProbe
    fuel_probe_module.FuelProbe = _FakeFuelProbe
    try:
        session = run_fuel_probe(
            "https://tankpit.com/play",
            "fuel_probe.json",
            target_pickups=3,
            max_attempts=3,
        )
    finally:
        fuel_probe_module.FuelProbe = original_probe_class

    written = fake_fs.read_text(Path("fuel_probe.json"))
    decoded = decode_fuel_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("fuel_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session == decoded
    assert session["capture_session_path"] == "fuel_probe.capture_session.json"
    assert capture_decoded["session_id"] == "fuel-session"
