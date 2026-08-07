"""Tests for equipment-attempt terminal outcomes."""

from __future__ import annotations

import pytest
from tests.action_lab._equipment_attempt_harness import (
    _ATTEMPT,
    _TARGET,
    _build_map_sync_timeout,
    _build_no_vis,
    _build_radar_timeout,
    _build_repo_map,
    _build_repo_tp,
    _build_tp_timeout,
    _fake_tp_landed,
    _fake_tp_missing_dispatch,
    _fake_tp_sync_timeout,
    _no_land,
    _no_repo,
    _noop_finalize,
    _Probe,
    _sync_policy,
    _waiter,
    _world,
    run_attempt,
)

from tankpit_bot.action_lab.equipment_collection_phase import (
    run_tracked_equipment_collection_phase,
)
from tankpit_bot.action_lab.equipment_probe_attempt import (
    run_single_equipment_target_attempt,
)
from tankpit_bot.action_lab.equipment_probe_types import EquipmentProbeAttemptResultDict
from tankpit_bot.action_lab.equipment_target_phase import (
    EquipmentTargetPhaseProbeProtocol,
)
from tankpit_bot.action_lab.session import (
    BufferedWorldStateProviderProtocol,
    WaitPageProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol
from tankpit_bot.state.types import ContainerStateDict


class TestEquipmentAttemptOutcomes:
    """Tests for equipment-attempt terminal outcomes."""

    def setup_method(self) -> None:
        get_world_service().world_state = _world()
        update_inventory_from_protocol(
            get_world_service(),
            [0, 0, 0, 0, 0],
            [False] * 5,
        )

    def test_map_sync_timeout(self) -> None:
        """Lines 366-376."""
        result = run_attempt(_fake_tp_sync_timeout)
        assert result["status"] == "no_equipment_visible"

    def test_resolved_target_runs_pickup_attempt(self) -> None:
        """A landed teleport + visible reachable target reaches the pickup phase."""
        from tankpit_bot.action_lab import _test_hooks as action_hooks
        from tankpit_bot.state import make_container_state

        container = make_container_state(101, 100, False, 0, timestamp_ms=2000)
        pickup_calls: list[ContainerStateDict] = []

        def _find_container(p: EquipmentTargetPhaseProbeProtocol) -> ContainerStateDict | None:
            _ = p
            return container

        def _recording_pickup(
            *,
            page: WaitPageProtocol,
            target: TeleportTargetDict,
            map_open_started_ms: int,
            map_sync_timestamp_ms: int | None,
            teleport_started_ms: int,
            radar_started_ms: int,
            radar_sync_timestamp_ms: int,
            reposition_map_open_started_ms: int | None,
            reposition_map_sync_timestamp_ms: int | None,
            reposition_teleport_started_ms: int | None,
            pickup_timeout_ms: int,
            inventory_count_before: int,
            teleport_result: TeleportAttemptResultDict,
            equipment_target: ContainerStateDict,
            message_start_index: int,
            teleport_cycle_ids: list[int],
            radar_cycle_id: int,
        ) -> EquipmentProbeAttemptResultDict:
            pickup_calls.append(equipment_target)
            return _ATTEMPT

        def _radar_sync(
            page: WaitPageProtocol,
            provider: BufferedWorldStateProviderProtocol,
            started_ms: int,
            timeout_ms: int,
        ) -> int | None:
            _ = (page, provider, started_ms, timeout_ms)
            return 1400

        action_hooks.wait_for_radar_sync = _radar_sync

        result = run_single_equipment_target_attempt(
            probe=_Probe(),
            target=_TARGET,
            map_sync_timeout_ms=30000,
            teleport_timeout_ms=30000,
            radar_timeout_ms=30000,
            pickup_timeout_ms=10000,
            settle_delay_ms=0,
            teleport_strategy="sync_before_teleport",
            cdp=None,
            wait_for_teleport_outcome=_waiter,
            run_tracked_teleport_attempt=_fake_tp_landed,
            run_tracked_equipment_collection_phase=run_tracked_equipment_collection_phase,
            build_map_sync_timeout_result=_build_map_sync_timeout,
            build_teleport_timeout_result=_build_tp_timeout,
            finalize_attempt_delay=_noop_finalize,
            terrain_provider=lambda: None,
            find_visible_target=_find_container,
            requires_reposition=_no_repo,
            find_landing_tile=_no_land,
            get_phase_overlaps=lambda: [],
            build_radar_timeout_result=_build_radar_timeout,
            build_no_equipment_visible_result=_build_no_vis,
            build_reposition_map_sync_timeout_result=_build_repo_map,
            build_reposition_teleport_timeout_result=_build_repo_tp,
            run_pickup_attempt=_recording_pickup,
            make_reposition_target=lambda x, y: _TARGET,
            teleport_strategy_requires_map_sync=_sync_policy,
            dispatch_failure_error=RuntimeError,
            unavailable_error=RuntimeError,
            unexpected_result_error=RuntimeError,
            unexpected_missing_target_error=RuntimeError,
            no_landing_tile_error=RuntimeError,
            missing_dispatch_error=RuntimeError,
            acquisition_dispatch_failure_message="m",
            teleport_dispatch_failure_message="t",
            reposition_acquisition_dispatch_failure_message="rm",
            reposition_teleport_dispatch_failure_message="rt",
            unavailable_message="u",
            impossible_map_sync_timeout_message="i",
            reposition_impossible_result_message="ri",
            reposition_missing_target_message="rmt",
            no_landing_tile_message="nl",
            missing_dispatch_message="missing dispatch",
        )

        assert result is _ATTEMPT
        assert pickup_calls == [container]

    def test_missing_dispatch_raises(self) -> None:
        """Line 381."""
        with pytest.raises(RuntimeError, match="missing dispatch"):
            run_attempt(
                _fake_tp_missing_dispatch,
                strategy="immediate_after_map_open",
            )
