"""Focused branch tests for the live fuel probe harness."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Literal, Protocol

import pytest
from tests.action_lab.test_fuel_probe import _Clock, _ProbeHarness

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbe,
    FuelProbeError,
    _find_visible_fuel_landing_tile,
    _visible_fuel_requires_reposition,
    format_fuel_probe_summary,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
)
from tankpit_bot.action_lab.fuel_targeting import FuelTargetingError
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportStartupTimingDict,
    TeleportTargetDict,
)
from tankpit_bot.state import ContainerStateDict, make_container_state


class _WaitForTeleportOutcomeProtocol(Protocol):
    """Callable protocol for teleport outcome waiting."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict: ...


class _FuelProbeModuleProtocol(Protocol):
    """Typed access to patchable fuel probe globals."""

    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol
    _find_visible_fuel_target: Callable[[FuelProbe, bool], ContainerStateDict | None]
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


_fuel_module_import = __import__("tankpit_bot.action_lab.fuel_probe", fromlist=["fuel_probe"])
fuel_probe_module: _FuelProbeModuleProtocol = _fuel_module_import


class _SequenceProbeHarness(_ProbeHarness):
    """Probe harness with per-call command dispatch outcomes."""

    def __init__(
        self,
        clock: _Clock,
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


class _DisappearingFuelProbe(_ProbeHarness):
    """Probe harness that simulates impossible fuel-target loss after radar."""

    def _resolve_fuel_target_after_radar(
        self,
        *,
        page: action_session.WaitPageProtocol,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
    ) -> tuple[
        ContainerStateDict | None,
        TeleportAttemptResultDict,
        FuelProbeAttemptResultDict | None,
        int | None,
        int | None,
        int | None,
    ]:
        _ = (
            page,
            target,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            radar_started_ms,
            radar_sync_timestamp_ms,
            map_sync_timeout_ms,
            teleport_timeout_ms,
            fuel_before,
            message_start_index,
        )
        return (None, teleport_result, None, None, None, None)


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore patched fuel probe hooks after each test."""
    original_get_time = action_hooks.get_current_time_ms
    original_wait_sync = action_session.wait_for_world_sync
    original_wait_outcome = fuel_probe_module._wait_for_teleport_outcome
    original_find_visible = fuel_probe_module._find_visible_fuel_target
    original_requires_reposition = fuel_probe_module._visible_fuel_requires_reposition
    original_find_landing = fuel_probe_module._find_visible_fuel_landing_tile
    original_targeting_requires_reposition = fuel_probe_module.visible_fuel_requires_reposition
    original_targeting_find_landing = fuel_probe_module.find_visible_fuel_landing_tile
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_session.wait_for_world_sync = original_wait_sync
    fuel_probe_module._wait_for_teleport_outcome = original_wait_outcome
    fuel_probe_module._find_visible_fuel_target = original_find_visible
    fuel_probe_module._visible_fuel_requires_reposition = original_requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = original_find_landing
    fuel_probe_module.visible_fuel_requires_reposition = original_targeting_requires_reposition
    fuel_probe_module.find_visible_fuel_landing_tile = original_targeting_find_landing


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
        "message_start_index": 0,
        "message_end_index": 1,
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
    fuel_probe_module._find_visible_fuel_target = (
        lambda probe, allow_unreachable: make_container_state(101, 100, True, 300)
    )
    fuel_probe_module._visible_fuel_requires_reposition = lambda probe, fuel_target: True
    fuel_probe_module._find_visible_fuel_landing_tile = lambda probe, fuel_target: (102, 100)


def test_format_fuel_probe_summary_counts_reposition_statuses() -> None:
    """Fuel summary includes reposition timeout counters explicitly."""
    summary = format_fuel_probe_summary(
        _session_with_statuses(
            [
                "picked_up_fuel",
                "reposition_map_sync_timeout",
                "reposition_teleport_timeout",
            ]
        )
    )

    assert "reposition_map_sync_timeout=1" in summary
    assert "reposition_teleport_timeout=1" in summary
    assert "target_pickups=2" in summary


def test_targeting_wrappers_convert_targeting_errors() -> None:
    """Fuel-probe targeting wrappers convert shared targeting errors."""
    probe = _ProbeHarness(_Clock(1000))
    target = make_container_state(101, 100, True, 300)

    def _raise_requires_reposition(
        current_probe: FuelProbe,
        fuel_target: ContainerStateDict,
    ) -> bool:
        _ = (current_probe, fuel_target)
        raise FuelTargetingError("terrain map is unavailable")

    def _raise_find_landing(
        current_probe: FuelProbe,
        fuel_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (current_probe, fuel_target)
        raise FuelTargetingError("self state is unavailable")

    fuel_probe_module.visible_fuel_requires_reposition = _raise_requires_reposition
    fuel_probe_module.find_visible_fuel_landing_tile = _raise_find_landing

    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        _visible_fuel_requires_reposition(probe, target)

    with pytest.raises(FuelProbeError, match="self state is unavailable"):
        _find_visible_fuel_landing_tile(probe, target)


def test_probe_single_target_raises_when_reposition_has_no_landing_tile() -> None:
    """Fuel probe rejects blocked visible fuel without a teleport landing tile."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, timeout_ms)
        return TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=620,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    _set_common_probe_hooks(_teleport_outcome)
    fuel_probe_module._find_visible_fuel_landing_tile = lambda probe, fuel_target: None

    with pytest.raises(
        FuelProbeError,
        match="visible fuel target has no teleport landing tile",
    ):
        _ProbeHarness(clock)._probe_single_fuel_target(
            target=_target(),
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_raises_when_reposition_map_open_dispatch_fails() -> None:
    """Fuel probe rejects blocked-fuel reposition when map-open dispatch fails."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
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
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, timeout_ms)
        return TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=620,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    action_session.wait_for_world_sync = _wait_for_world_sync
    _set_common_probe_hooks(_teleport_outcome)

    with pytest.raises(
        FuelProbeError,
        match="map_open command dispatch failed during fuel reposition",
    ):
        _SequenceProbeHarness(
            clock,
            open_map_results=[True, False],
            teleport_results=[True],
        )._probe_single_fuel_target(
            target=_target(),
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_raises_when_reposition_teleport_dispatch_fails() -> None:
    """Fuel probe rejects blocked-fuel reposition when teleport dispatch fails."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    wait_results = [1200, 1600, 1800]

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
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, timeout_ms)
        return TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=620,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    action_session.wait_for_world_sync = _wait_for_world_sync
    _set_common_probe_hooks(_teleport_outcome)

    with pytest.raises(
        FuelProbeError,
        match="teleport command dispatch failed during fuel reposition",
    ):
        _SequenceProbeHarness(
            clock,
            open_map_results=[True, True],
            teleport_results=[True, False],
        )._probe_single_fuel_target(
            target=_target(),
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_raises_when_reposition_teleport_reports_map_sync_timeout() -> None:
    """Fuel probe rejects impossible map-sync timeout after reposition teleport dispatch."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    wait_results = [1200, 1600, 1800]

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
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, timeout_ms)
        status: Literal[
            "landed_exact",
            "landed_offset",
            "map_sync_timeout",
            "teleport_timeout",
        ] = "map_sync_timeout" if target["label"].startswith("fuel_reposition_") else "landed_exact"
        return TeleportAttemptResultDict(
            target=target,
            status=status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=620,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=status == "landed_exact",
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    action_session.wait_for_world_sync = _wait_for_world_sync
    _set_common_probe_hooks(_teleport_outcome)

    with pytest.raises(
        TeleportProbeError,
        match="teleport outcome reported impossible map_sync_timeout during fuel reposition",
    ):
        _SequenceProbeHarness(
            clock,
            open_map_results=[True, True],
            teleport_results=[True, True],
        )._probe_single_fuel_target(
            target=_target(),
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_raises_when_visible_fuel_disappears_after_radar() -> None:
    """Fuel probe rejects impossible loss of visible fuel after target resolution."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, timeout_ms)
        return TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=620,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome

    with pytest.raises(FuelProbeError, match="visible fuel target disappeared unexpectedly"):
        _DisappearingFuelProbe(clock)._probe_single_fuel_target(
            target=_target(),
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )
