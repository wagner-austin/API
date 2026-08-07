"""Tests for fuel-probe branch coverage: targeting wrappers.

``test_fuel_probe_branches.py`` was 617 lines; the reposition branches
are now a sibling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Literal,
)

import pytest
from tests.action_lab._fuel_branches_harness import (
    _session_with_statuses,
    _set_common_probe_hooks,
    _target,
    fuel_probe_module,
)
from tests.action_lab._fuel_probe_harness import (
    _ProbeHarness,
    _terrain,
)
from tests.action_lab._replay_page import ReplayClock

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import fuel_collection_phase
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseOverlapDict
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbe,
    FuelProbeError,
    _find_visible_fuel_landing_tile,
    _visible_fuel_requires_reposition,
    format_fuel_probe_summary,
)
from tankpit_bot.action_lab.fuel_target_phase import (
    BuildNoFuelVisibleResultProtocol,
    BuildRepositionMapSyncTimeoutResultProtocol,
    BuildRepositionTeleportTimeoutResultProtocol,
    FuelTargetPhaseProbeProtocol,
    FuelTargetResolution,
)
from tankpit_bot.action_lab.fuel_targeting import FuelTargetingError
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.state import (
    ContainerStateDict,
    make_container_state,
)


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
    probe = _ProbeHarness(ReplayClock(1000))
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
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
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
            page_snapshots=[],
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


def test_probe_single_target_raises_when_visible_fuel_disappears_after_radar() -> None:
    """Fuel probe rejects impossible loss of visible fuel after target resolution."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
    action_hooks.wait_for_radar_sync = lambda page, provider, started_ms, timeout_ms: 1200
    original_resolve_fuel_target_phase = fuel_collection_phase.resolve_fuel_target_phase
    fuel_probe_module.get_terrain_map = lambda: _terrain({(124, 100), (101, 100), (102, 100)})

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
            page_snapshots=[],
        )

    _set_common_probe_hooks(_teleport_outcome)

    def _resolve_fuel_target_phase(
        page: action_session.WaitPageProtocol,
        probe: FuelTargetPhaseProbeProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        snapshot_before: PageClientSnapshotDict,
        capture_snapshot: Callable[[], PageClientSnapshotDict],
        terrain_provider: Callable[[], TerrainMapProtocol | None],
        find_visible_target: Callable[
            [FuelTargetPhaseProbeProtocol],
            ContainerStateDict | None,
        ],
        requires_reposition: Callable[
            [FuelTargetPhaseProbeProtocol, ContainerStateDict],
            bool,
        ],
        find_landing_tile: Callable[
            [FuelTargetPhaseProbeProtocol, ContainerStateDict],
            tuple[int, int] | None,
        ],
        get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
        build_no_fuel_visible_result: BuildNoFuelVisibleResultProtocol,
        build_reposition_map_sync_timeout_result: BuildRepositionMapSyncTimeoutResultProtocol,
        build_reposition_teleport_timeout_result: BuildRepositionTeleportTimeoutResultProtocol,
        make_reposition_target: Callable[[int, int], TeleportTargetDict],
        wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol,
        teleport_strategy_requires_map_sync: Callable[
            [Literal["sync_before_teleport", "immediate_after_map_open"]],
            bool,
        ],
        dispatch_failure_error: type[Exception],
        unexpected_result_error: type[Exception],
        no_landing_tile_error: type[Exception],
        unavailable_error: type[Exception],
        unavailable_message: str,
        no_landing_tile_message: str,
        impossible_result_message: str,
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
    ) -> FuelTargetResolution:
        _ = (
            page,
            probe,
            cdp,
            target,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            radar_started_ms,
            radar_sync_timestamp_ms,
            map_sync_timeout_ms,
            teleport_timeout_ms,
            fuel_before,
            teleport_result,
            message_start_index,
            teleport_cycle_ids,
            radar_cycle_id,
            teleport_strategy,
            snapshot_before,
            capture_snapshot,
            terrain_provider,
            find_visible_target,
            requires_reposition,
            find_landing_tile,
            get_phase_overlaps,
            build_no_fuel_visible_result,
            build_reposition_map_sync_timeout_result,
            build_reposition_teleport_timeout_result,
            make_reposition_target,
            wait_for_teleport_outcome,
            teleport_strategy_requires_map_sync,
            dispatch_failure_error,
            unexpected_result_error,
            no_landing_tile_error,
            unavailable_error,
            unavailable_message,
            no_landing_tile_message,
            impossible_result_message,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
        )
        return FuelTargetResolution(
            fuel_target=None,
            teleport_result=teleport_result,
            terminal_result=None,
            decision_basis=None,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )

    fuel_collection_phase.resolve_fuel_target_phase = _resolve_fuel_target_phase
    try:
        with pytest.raises(FuelProbeError, match="visible fuel target disappeared unexpectedly"):
            _ProbeHarness(clock)._probe_single_fuel_target(
                target=_target(),
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=0,
            )
    finally:
        fuel_collection_phase.resolve_fuel_target_phase = original_resolve_fuel_target_phase
