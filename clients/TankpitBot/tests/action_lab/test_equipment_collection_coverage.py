"""Tests for equipment target resolution and collection outcomes.

``test_equipment_collection_coverage.py`` was 1,003 lines; the
reposition paths are now a sibling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from tests.action_lab._equipment_collection_harness import (
    _TARGET,
    _TP_RESULT,
    _build_no_vis,
    _build_pickup,
    _build_radar_timeout,
    _build_repo_map,
    _build_repo_tp,
    _no_find,
    _no_land,
    _no_repo,
    _Page,
    _Probe,
    _resolve,
    _sync_policy,
    _waiter,
)

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.action_lab import (
    equipment_collection_phase as ecp_module,
)
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseOverlapDict,
)
from tankpit_bot.action_lab.equipment_collection_phase import (
    run_tracked_equipment_collection_phase,
)
from tankpit_bot.action_lab.equipment_target_phase import (
    BuildEquipmentRepositionMapSyncTimeoutResultProtocol,
    BuildEquipmentRepositionTeleportTimeoutResultProtocol,
    BuildNoEquipmentVisibleResultProtocol,
    EquipmentTargetPhaseProbeProtocol,
    resolve_equipment_target_after_radar,
)
from tankpit_bot.action_lab.session import (
    WaitPageProtocol,
)
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
)


def test_resolve_no_equipment_visible() -> None:
    """Line 427: find_visible_target returns None."""
    result = _resolve(find_vis=False)
    if result is None:
        pytest.fail("expected terminal result")
    assert result["status"] == "no_equipment_visible"


def test_resolve_equipment_no_reposition() -> None:
    """Line 449: equipment found, no reposition needed."""
    assert _resolve(find_vis=True) is None


def test_collection_propagates_terminal_result() -> None:
    """Line 375: no equipment visible via real resolve -> terminal returned."""
    original_radar = ecp_module.run_radar_phase
    original_resolve = ecp_module.resolve_equipment_target_phase

    def fake_radar(
        page: WaitPageProtocol,
        probe: ecp_module.EquipmentCollectionPhaseProbeProtocol,
        *,
        attempt_label: str,
        timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[ActionPhaseCycleDict, int, int | None]:
        """Fake radar that always succeeds."""
        return (
            ActionPhaseCycleDict(
                phase="radar",
                cycle_id=99,
                started_ms=1300,
            ),
            1300,
            1400,
        )

    ecp_module.run_radar_phase = fake_radar
    ecp_module.resolve_equipment_target_phase = resolve_equipment_target_after_radar
    try:
        result = run_tracked_equipment_collection_phase(
            page=_Page(),
            probe=_Probe(),
            cdp=None,
            target=_TARGET,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            map_sync_timeout_ms=30000,
            teleport_timeout_ms=30000,
            radar_timeout_ms=30000,
            pickup_timeout_ms=10000,
            inventory_count_before=0,
            teleport_result=_TP_RESULT,
            message_start_index=0,
            teleport_cycle_ids=[1],
            teleport_strategy="immediate_after_map_open",
            terrain_provider=lambda: None,
            find_visible_target=_no_find,
            requires_reposition=_no_repo,
            find_landing_tile=_no_land,
            get_phase_overlaps=lambda: [],
            build_radar_timeout_result=_build_radar_timeout,
            build_no_equipment_visible_result=_build_no_vis,
            build_reposition_map_sync_timeout_result=_build_repo_map,
            build_reposition_teleport_timeout_result=_build_repo_tp,
            run_pickup_attempt=_build_pickup,
            make_reposition_target=lambda x, y: _TARGET,
            wait_for_teleport_outcome=_waiter,
            teleport_strategy_requires_map_sync=_sync_policy,
            dispatch_failure_error=RuntimeError,
            unexpected_result_error=RuntimeError,
            unexpected_missing_target_error=RuntimeError,
            no_landing_tile_error=RuntimeError,
            unavailable_error=RuntimeError,
            unavailable_message="u",
            no_landing_tile_message="nl",
            impossible_result_message="i",
            acquisition_dispatch_failure_message="m",
            teleport_dispatch_failure_message="t",
            unexpected_missing_target_message="missing",
        )
        assert result["status"] == "no_equipment_visible"
    finally:
        ecp_module.run_radar_phase = original_radar
        ecp_module.resolve_equipment_target_phase = original_resolve


def test_collection_impossible_missing_target() -> None:
    """equipment_collection_phase.py line 377."""
    from tankpit_bot.action_lab.equipment_target_phase import EquipmentTargetResolution

    original_radar = ecp_module.run_radar_phase
    original_resolve = ecp_module.resolve_equipment_target_phase

    def fake_radar(
        page: WaitPageProtocol,
        probe: ecp_module.EquipmentCollectionPhaseProbeProtocol,
        *,
        attempt_label: str,
        timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[ActionPhaseCycleDict, int, int | None]:
        return (
            ActionPhaseCycleDict(phase="radar", cycle_id=99, started_ms=1300),
            1300,
            1400,
        )

    def fake_resolve(
        page: WaitPageProtocol,
        probe: EquipmentTargetPhaseProbeProtocol,
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
        inventory_count_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
        teleport_cycle_ids: list[int],
        radar_cycle_id: int,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        terrain_provider: Callable[[], TerrainMapProtocol | None],
        find_visible_target: Callable[
            [EquipmentTargetPhaseProbeProtocol],
            ContainerStateDict | None,
        ],
        requires_reposition: Callable[
            [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
            bool,
        ],
        find_landing_tile: Callable[
            [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
            tuple[int, int] | None,
        ],
        get_phase_overlaps: Callable[[], list[ActionPhaseOverlapDict]],
        build_no_equipment_visible_result: BuildNoEquipmentVisibleResultProtocol,
        build_reposition_map_sync_timeout_result: (
            BuildEquipmentRepositionMapSyncTimeoutResultProtocol
        ),
        build_reposition_teleport_timeout_result: (
            BuildEquipmentRepositionTeleportTimeoutResultProtocol
        ),
        make_reposition_target: Callable[[int, int], TeleportTargetDict],
        wait_for_teleport_outcome: TeleportOutcomeWaiterProtocol,
        teleport_strategy_requires_map_sync: Callable[
            [Literal["sync_before_teleport", "immediate_after_map_open"]],
            bool,
        ],
        no_landing_tile_error: type[Exception],
        dispatch_failure_error: type[Exception],
        unavailable_error: type[Exception],
        unexpected_result_error: type[Exception],
        unavailable_message: str,
        no_landing_tile_message: str,
        impossible_result_message: str,
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
    ) -> EquipmentTargetResolution:
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
            inventory_count_before,
            teleport_result,
            message_start_index,
            teleport_cycle_ids,
            radar_cycle_id,
            teleport_strategy,
            terrain_provider,
            find_visible_target,
            requires_reposition,
            find_landing_tile,
            get_phase_overlaps,
            build_no_equipment_visible_result,
            build_reposition_map_sync_timeout_result,
            build_reposition_teleport_timeout_result,
            make_reposition_target,
            wait_for_teleport_outcome,
            teleport_strategy_requires_map_sync,
            no_landing_tile_error,
            dispatch_failure_error,
            unavailable_error,
            unexpected_result_error,
            unavailable_message,
            no_landing_tile_message,
            impossible_result_message,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
        )
        return EquipmentTargetResolution(
            equipment_target=None,
            teleport_result=_TP_RESULT,
            terminal_result=None,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
        )

    ecp_module.run_radar_phase = fake_radar
    ecp_module.resolve_equipment_target_phase = fake_resolve
    try:
        with pytest.raises(RuntimeError, match="missing"):
            run_tracked_equipment_collection_phase(
                page=_Page(),
                probe=_Probe(),
                cdp=None,
                target=_TARGET,
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                map_sync_timeout_ms=30000,
                teleport_timeout_ms=30000,
                radar_timeout_ms=30000,
                pickup_timeout_ms=10000,
                inventory_count_before=0,
                teleport_result=_TP_RESULT,
                message_start_index=0,
                teleport_cycle_ids=[1],
                teleport_strategy="immediate_after_map_open",
                terrain_provider=lambda: None,
                find_visible_target=_no_find,
                requires_reposition=_no_repo,
                find_landing_tile=_no_land,
                get_phase_overlaps=lambda: [],
                build_radar_timeout_result=_build_radar_timeout,
                build_no_equipment_visible_result=_build_no_vis,
                build_reposition_map_sync_timeout_result=_build_repo_map,
                build_reposition_teleport_timeout_result=_build_repo_tp,
                run_pickup_attempt=_build_pickup,
                make_reposition_target=lambda x, y: _TARGET,
                wait_for_teleport_outcome=_waiter,
                teleport_strategy_requires_map_sync=_sync_policy,
                dispatch_failure_error=RuntimeError,
                unexpected_result_error=RuntimeError,
                unexpected_missing_target_error=RuntimeError,
                no_landing_tile_error=RuntimeError,
                unavailable_error=RuntimeError,
                unavailable_message="u",
                no_landing_tile_message="nl",
                impossible_result_message="i",
                acquisition_dispatch_failure_message="m",
                teleport_dispatch_failure_message="t",
                unexpected_missing_target_message="missing",
            )
    finally:
        ecp_module.run_radar_phase = original_radar
        ecp_module.resolve_equipment_target_phase = original_resolve
