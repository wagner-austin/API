"""Coverage for equipment target resolution and collection phase.

Exercises resolve_equipment_target_after_radar directly (lines 427, 449)
and run_tracked_equipment_collection_phase via module-level hook swap
for the terminal_result propagation path (line 375).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from typing_extensions import Unpack

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.action_lab import (
    equipment_collection_phase as ecp_module,
)
from tankpit_bot.action_lab import equipment_target_phase as _etp_mod
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseOverlapDict,
)
from tankpit_bot.action_lab.equipment_collection_phase import (
    run_tracked_equipment_collection_phase,
)
from tankpit_bot.action_lab.equipment_probe_types import (
    EquipmentProbeAttemptResultDict,
)
from tankpit_bot.action_lab.equipment_target_phase import (
    BlockedEquipmentRepositionResult,
    BuildEquipmentRepositionMapSyncTimeoutResultProtocol,
    BuildEquipmentRepositionTeleportTimeoutResultProtocol,
    BuildNoEquipmentVisibleResultProtocol,
    EquipmentTargetPhaseProbeProtocol,
    _run_blocked_equipment_reposition,
    resolve_equipment_target_after_radar,
)
from tankpit_bot.action_lab.session import (
    BufferedWorldStateProviderProtocol,
    WaitPageProtocol,
)
from tankpit_bot.action_lab.teleport_attempt import TrackedTeleportAttempt
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterKwargs,
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.state import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.types import CapturedMessage

_TARGET = TeleportTargetDict(label="t", x=10, y=20)
_SELF = make_self_state(
    tank_id=1,
    x=100,
    y=100,
    team=2,
    rank=1,
    fuel=700,
    leaderboard_position=1,
)


def _world() -> WorldStateDict:
    """Build world."""
    b = make_empty_world_state()
    return WorldStateDict(
        self_state=_SELF,
        tanks=b["tanks"],
        containers=b["containers"],
        mines=b["mines"],
        terrain=b["terrain"],
        viewport=ViewportStateDict(left=92, top=92, width=16, height=16),
        scanned_tiles=b["scanned_tiles"],
        timestamp_ms=2000,
    )


_ATTEMPT = EquipmentProbeAttemptResultDict(
    target=_TARGET,
    teleport_cycle_ids=[1],
    radar_cycle_id=None,
    move_cycle_id=None,
    pickup_cycle_id=None,
    status="no_equipment_visible",
    map_open_started_ms=1000,
    map_sync_timestamp_ms=None,
    teleport_started_ms=None,
    radar_started_ms=None,
    radar_sync_timestamp_ms=None,
    reposition_map_open_started_ms=None,
    reposition_map_sync_timestamp_ms=None,
    reposition_teleport_started_ms=None,
    pickup_started_ms=None,
    completion_timestamp_ms=2000,
    inventory_count_before=0,
    inventory_count_after=None,
    landed_signal_received=False,
    landed_x=None,
    landed_y=None,
    equipment_target_x=None,
    equipment_target_y=None,
    phase_overlaps=[],
    message_start_index=0,
    message_end_index=0,
)

_TP_RESULT = TeleportAttemptResultDict(
    target=_TARGET,
    teleport_cycle_id=1,
    status="landed_exact",
    map_open_started_ms=1000,
    map_sync_timestamp_ms=1100,
    teleport_started_ms=1200,
    completion_timestamp_ms=1500,
    map_sync_elapsed_ms=100,
    teleport_elapsed_ms=300,
    fuel_before=700,
    fuel_after=690,
    world_timestamp_before=1100,
    world_timestamp_after=1450,
    landed_signal_received=True,
    landed_x=10,
    landed_y=20,
    message_start_index=0,
    message_end_index=0,
    page_snapshots=[],
)


class _Page:
    """Minimal page."""

    def wait_for_timeout(self, timeout: float) -> None:
        """No-op."""

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


class _Probe:
    """Minimal probe."""

    def __init__(self) -> None:
        self._messages: list[CapturedMessage] = []
        self._w = _world()
        self._cid = 0
        self._cdp_message_buffer: list[str] = []

    @property
    def messages(self) -> list[CapturedMessage]:
        """Messages."""
        return self._messages

    @property
    def magic(self) -> str | None:
        """Magic."""
        return None

    def open_map(self) -> bool:
        """Open map."""
        return True

    def use_radar(self) -> bool:
        """Radar."""
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        """Teleport."""
        return True

    def get_world_state(self) -> WorldStateDict:
        """World."""
        return self._w

    def get_self_state(self) -> SelfStateDict | None:
        """Self."""
        return self._w["self_state"]

    def _require_self_state(self) -> SelfStateDict:
        """Require self."""
        return _SELF

    def _start_action_phase(
        self,
        phase: Literal["teleport", "radar", "move", "pickup"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start phase."""
        self._cid += 1
        return ActionPhaseCycleDict(
            phase=phase,
            cycle_id=self._cid,
            started_ms=1000,
        )

    def _end_action_phase(
        self,
        cycle: ActionPhaseCycleDict,
    ) -> None:
        """End phase."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset."""


def _no_find(
    p: EquipmentTargetPhaseProbeProtocol,
) -> ContainerStateDict | None:
    """Return no equipment."""
    return None


def _found(
    p: EquipmentTargetPhaseProbeProtocol,
) -> ContainerStateDict | None:
    """Return equipment at (10, 20)."""
    return make_container_state(
        x=10,
        y=20,
        is_fuel=False,
        volume=0,
        timestamp_ms=1000,
    )


def _no_repo(
    p: EquipmentTargetPhaseProbeProtocol,
    c: ContainerStateDict,
) -> bool:
    """No reposition."""
    return False


def _no_land(
    p: EquipmentTargetPhaseProbeProtocol,
    c: ContainerStateDict,
) -> tuple[int, int] | None:
    """No landing."""
    return None


def _build_no_vis(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    inventory_count_before: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """No-equipment-visible builder."""
    return _ATTEMPT


def _build_repo_map(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    reposition_map_open_started_ms: int,
    inventory_count_before: int,
    teleport_result: TeleportAttemptResultDict,
    equipment_target: ContainerStateDict,
    message_start_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """Reposition map sync timeout builder."""
    return _ATTEMPT


def _build_repo_tp(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    radar_sync_timestamp_ms: int,
    reposition_map_open_started_ms: int,
    reposition_map_sync_timestamp_ms: int | None,
    reposition_teleport_started_ms: int,
    inventory_count_before: int,
    teleport_result: TeleportAttemptResultDict,
    equipment_target: ContainerStateDict,
    message_start_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """Reposition teleport timeout builder."""
    return _ATTEMPT


def _build_radar_timeout(
    *,
    target: TeleportTargetDict,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    radar_started_ms: int,
    inventory_count_before: int,
    teleport_result: TeleportAttemptResultDict,
    message_start_index: int,
    teleport_cycle_ids: list[int],
    radar_cycle_id: int,
    phase_overlaps: list[ActionPhaseOverlapDict],
) -> EquipmentProbeAttemptResultDict:
    """Radar timeout builder."""
    return _ATTEMPT


def _build_pickup(
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
    """Pickup builder (never called in terminal_result path)."""
    return _ATTEMPT


def _waiter(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    target: TeleportTargetDict,
    **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
) -> TeleportAttemptResultDict:
    """Unreachable waiter."""
    raise AssertionError("should not be called")


def _sync_policy(
    s: Literal["sync_before_teleport", "immediate_after_map_open"],
) -> bool:
    """Sync policy."""
    return s == "sync_before_teleport"


def _resolve(
    find_vis: bool = False,
) -> EquipmentProbeAttemptResultDict | None:
    """Call resolve and return terminal_result if any."""

    def _fv(
        p: EquipmentTargetPhaseProbeProtocol,
    ) -> ContainerStateDict | None:
        return _found(p) if find_vis else None

    r = resolve_equipment_target_after_radar(
        page=_Page(),
        probe=_Probe(),
        cdp=None,
        target=_TARGET,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        map_sync_timeout_ms=30000,
        teleport_timeout_ms=30000,
        inventory_count_before=0,
        teleport_result=_TP_RESULT,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        teleport_strategy="immediate_after_map_open",
        terrain_provider=lambda: None,
        find_visible_target=_fv,
        requires_reposition=_no_repo,
        find_landing_tile=_no_land,
        get_phase_overlaps=lambda: [],
        build_no_equipment_visible_result=_build_no_vis,
        build_reposition_map_sync_timeout_result=_build_repo_map,
        build_reposition_teleport_timeout_result=_build_repo_tp,
        make_reposition_target=lambda x, y: _TARGET,
        wait_for_teleport_outcome=_waiter,
        teleport_strategy_requires_map_sync=_sync_policy,
        no_landing_tile_error=RuntimeError,
        dispatch_failure_error=RuntimeError,
        unavailable_error=RuntimeError,
        unexpected_result_error=RuntimeError,
        unavailable_message="u",
        no_landing_tile_message="no landing",
        impossible_result_message="i",
        acquisition_dispatch_failure_message="m",
        teleport_dispatch_failure_message="t",
    )
    return r.terminal_result


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


# =========================================================================
# Additional coverage — lines 227, 261-262, 287, 289, 377, 494->496
# =========================================================================


_TP_TIMEOUT = TeleportAttemptResultDict(
    target=_TARGET,
    teleport_cycle_id=1,
    status="teleport_timeout",
    map_open_started_ms=1000,
    map_sync_timestamp_ms=1100,
    teleport_started_ms=1200,
    completion_timestamp_ms=1500,
    map_sync_elapsed_ms=100,
    teleport_elapsed_ms=300,
    fuel_before=700,
    fuel_after=690,
    world_timestamp_before=1100,
    world_timestamp_after=1450,
    landed_signal_received=False,
    landed_x=None,
    landed_y=None,
    message_start_index=0,
    message_end_index=0,
    page_snapshots=[],
)


def _has_land(
    p: EquipmentTargetPhaseProbeProtocol,
    c: ContainerStateDict,
) -> tuple[int, int] | None:
    """Landing tile available."""
    return (15, 25)


def _yes_repo(
    p: EquipmentTargetPhaseProbeProtocol,
    c: ContainerStateDict,
) -> bool:
    """Reposition required."""
    return True


_CYCLE = ActionPhaseCycleDict(phase="teleport", cycle_id=1, started_ms=1000)


def _make_tracked(
    *,
    sync_ts: int | None = 1100,
    tp_result: TeleportAttemptResultDict | None = _TP_RESULT,
    tp_started: int | None = 1200,
) -> TrackedTeleportAttempt:
    """Build a TrackedTeleportAttempt for reposition stubs."""
    from tankpit_bot.action_lab.types import TeleportPageSnapshotDict

    def _snap(
        label: Literal[
            "before_map_open",
            "before_teleport",
            "after_map_data",
            "landed",
            "timeout",
        ],
    ) -> TeleportPageSnapshotDict:
        return TeleportPageSnapshotDict(
            phase=label,
            timestamp_ms=1000,
            client_present=True,
            map_visible=False,
            client_state=0,
            client_busy=False,
            pending_actions=0,
            heartbeat_age_ms=0,
            last_page_client_send_age_ms=0,
            last_bot_send_age_ms=0,
            ws_ready_state=1,
            current_send_label=None,
            sent_frame_meta_queue_length=0,
            self_fields={},
            world_fields={},
            map_fields={},
            world_collections={},
        )

    return TrackedTeleportAttempt(
        message_start_index=0,
        teleport_cycle=_CYCLE,
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=sync_ts,
        page_snapshots=[],
        capture_page_snapshot=_snap,
        teleport_result=tp_result,
        teleport_started_ms=tp_started,
    )


def _common_reposition_call(
    *,
    find_landing: Callable[
        [EquipmentTargetPhaseProbeProtocol, ContainerStateDict],
        tuple[int, int] | None,
    ] = _has_land,
    strategy: Literal[
        "sync_before_teleport", "immediate_after_map_open"
    ] = "immediate_after_map_open",
) -> BlockedEquipmentRepositionResult:
    """Call _run_blocked_equipment_reposition with common args."""
    return _run_blocked_equipment_reposition(
        page=_Page(),
        probe=_Probe(),
        cdp=None,
        target=_TARGET,
        equipment_target=make_container_state(
            x=10,
            y=20,
            is_fuel=False,
            volume=0,
            timestamp_ms=1000,
        ),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        map_sync_timeout_ms=30000,
        teleport_timeout_ms=30000,
        inventory_count_before=0,
        teleport_result=_TP_RESULT,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        teleport_strategy=strategy,
        wait_for_teleport_outcome=_waiter,
        teleport_strategy_requires_map_sync=_sync_policy,
        find_landing_tile=find_landing,
        get_phase_overlaps=lambda: [],
        build_reposition_map_sync_timeout_result=_build_repo_map,
        build_reposition_teleport_timeout_result=_build_repo_tp,
        make_reposition_target=lambda x, y: _TARGET,
        dispatch_failure_error=RuntimeError,
        unavailable_error=RuntimeError,
        unexpected_result_error=RuntimeError,
        no_landing_tile_error=RuntimeError,
        unavailable_message="u",
        no_landing_tile_message="no landing",
        impossible_result_message="i",
        acquisition_dispatch_failure_message="m",
        teleport_dispatch_failure_message="t",
    )


def test_no_landing_tile_raises() -> None:
    """equipment_target_phase.py line 227."""
    with pytest.raises(RuntimeError, match="no landing"):
        _common_reposition_call(find_landing=_no_land)


def test_reposition_map_sync_timeout() -> None:
    """equipment_target_phase.py lines 261-262."""
    original = _etp_mod.run_equipment_reposition_attempt
    _etp_mod.run_equipment_reposition_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=None, tp_result=None, tp_started=None
    )
    try:
        result = _common_reposition_call(strategy="sync_before_teleport")
        if result.terminal_result is None:
            raise AssertionError("expected terminal result")
    finally:
        _etp_mod.run_equipment_reposition_attempt = original


def test_reposition_dispatch_failure_raises() -> None:
    """equipment_target_phase.py line 287."""
    original = _etp_mod.run_equipment_reposition_attempt
    _etp_mod.run_equipment_reposition_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=2100, tp_result=None, tp_started=None
    )
    try:
        with pytest.raises(RuntimeError):
            _common_reposition_call()
    finally:
        _etp_mod.run_equipment_reposition_attempt = original


def test_reposition_teleport_timeout() -> None:
    """equipment_target_phase.py line 289."""
    original = _etp_mod.run_equipment_reposition_attempt
    _etp_mod.run_equipment_reposition_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=2100, tp_result=_TP_TIMEOUT, tp_started=2200
    )
    try:
        result = _common_reposition_call()
        if result.terminal_result is None:
            raise AssertionError("expected terminal result")
    finally:
        _etp_mod.run_equipment_reposition_attempt = original


def test_reposition_success_propagates_teleport() -> None:
    """equipment_target_phase.py branch 494->496."""
    original = _etp_mod.run_equipment_reposition_attempt
    _etp_mod.run_equipment_reposition_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=2100, tp_result=_TP_RESULT, tp_started=2200
    )
    try:
        r = resolve_equipment_target_after_radar(
            page=_Page(),
            probe=_Probe(),
            cdp=None,
            target=_TARGET,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            map_sync_timeout_ms=30000,
            teleport_timeout_ms=30000,
            inventory_count_before=0,
            teleport_result=_TP_RESULT,
            message_start_index=0,
            teleport_cycle_ids=[1],
            radar_cycle_id=2,
            teleport_strategy="immediate_after_map_open",
            terrain_provider=lambda: None,
            find_visible_target=_found,
            requires_reposition=_yes_repo,
            find_landing_tile=_has_land,
            get_phase_overlaps=lambda: [],
            build_no_equipment_visible_result=_build_no_vis,
            build_reposition_map_sync_timeout_result=_build_repo_map,
            build_reposition_teleport_timeout_result=_build_repo_tp,
            make_reposition_target=lambda x, y: _TARGET,
            wait_for_teleport_outcome=_waiter,
            teleport_strategy_requires_map_sync=_sync_policy,
            no_landing_tile_error=RuntimeError,
            dispatch_failure_error=RuntimeError,
            unavailable_error=RuntimeError,
            unexpected_result_error=RuntimeError,
            unavailable_message="u",
            no_landing_tile_message="nl",
            impossible_result_message="i",
            acquisition_dispatch_failure_message="m",
            teleport_dispatch_failure_message="t",
        )
        if r.teleport_result is None:
            raise AssertionError("expected teleport result")
        assert r.teleport_result["status"] == "landed_exact"
    finally:
        _etp_mod.run_equipment_reposition_attempt = original


def test_reposition_timeout_preserves_original_teleport() -> None:
    """equipment_target_phase.py branch 494->496 False path."""
    original = _etp_mod.run_equipment_reposition_attempt
    _etp_mod.run_equipment_reposition_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=2100, tp_result=_TP_TIMEOUT, tp_started=2200
    )
    try:
        r = resolve_equipment_target_after_radar(
            page=_Page(),
            probe=_Probe(),
            cdp=None,
            target=_TARGET,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            map_sync_timeout_ms=30000,
            teleport_timeout_ms=30000,
            inventory_count_before=0,
            teleport_result=_TP_RESULT,
            message_start_index=0,
            teleport_cycle_ids=[1],
            radar_cycle_id=2,
            teleport_strategy="immediate_after_map_open",
            terrain_provider=lambda: None,
            find_visible_target=_found,
            requires_reposition=_yes_repo,
            find_landing_tile=_has_land,
            get_phase_overlaps=lambda: [],
            build_no_equipment_visible_result=_build_no_vis,
            build_reposition_map_sync_timeout_result=_build_repo_map,
            build_reposition_teleport_timeout_result=_build_repo_tp,
            make_reposition_target=lambda x, y: _TARGET,
            wait_for_teleport_outcome=_waiter,
            teleport_strategy_requires_map_sync=_sync_policy,
            no_landing_tile_error=RuntimeError,
            dispatch_failure_error=RuntimeError,
            unavailable_error=RuntimeError,
            unexpected_result_error=RuntimeError,
            unavailable_message="u",
            no_landing_tile_message="nl",
            impossible_result_message="i",
            acquisition_dispatch_failure_message="m",
            teleport_dispatch_failure_message="t",
        )
        assert r.teleport_result["status"] == "landed_exact"
        if r.terminal_result is None:
            raise AssertionError("expected terminal result from timeout")
    finally:
        _etp_mod.run_equipment_reposition_attempt = original


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
