"""Tests for the equipment reposition path.

Every terminal outcome a reposition attempt can reach, including the
timeout that must preserve the original teleport result.
"""

from __future__ import annotations

import pytest
from tests.action_lab._equipment_collection_harness import (
    _TARGET,
    _TP_RESULT,
    _TP_TIMEOUT,
    _build_no_vis,
    _build_repo_map,
    _build_repo_tp,
    _common_reposition_call,
    _found,
    _has_land,
    _make_tracked,
    _no_land,
    _Page,
    _Probe,
    _sync_policy,
    _waiter,
    _yes_repo,
)
from tests.action_lab._teleport_seams import equipment_target_phase_module

from tankpit_bot.action_lab.equipment_target_phase import (
    resolve_equipment_target_after_radar,
)


def test_no_landing_tile_raises() -> None:
    """equipment_target_phase.py line 227."""
    with pytest.raises(RuntimeError, match="no landing"):
        _common_reposition_call(find_landing=_no_land)


def test_reposition_map_sync_timeout() -> None:
    """equipment_target_phase.py lines 261-262."""
    original = equipment_target_phase_module.run_tracked_teleport_attempt
    equipment_target_phase_module.run_tracked_teleport_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=None, tp_result=None, tp_started=None
    )
    try:
        result = _common_reposition_call(strategy="sync_before_teleport")
        if result.terminal_result is None:
            raise AssertionError("expected terminal result")
    finally:
        equipment_target_phase_module.run_tracked_teleport_attempt = original


def test_reposition_dispatch_failure_raises() -> None:
    """equipment_target_phase.py line 287."""
    original = equipment_target_phase_module.run_tracked_teleport_attempt
    equipment_target_phase_module.run_tracked_teleport_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=2100, tp_result=None, tp_started=None
    )
    try:
        with pytest.raises(RuntimeError):
            _common_reposition_call()
    finally:
        equipment_target_phase_module.run_tracked_teleport_attempt = original


def test_reposition_teleport_timeout() -> None:
    """equipment_target_phase.py line 289."""
    original = equipment_target_phase_module.run_tracked_teleport_attempt
    equipment_target_phase_module.run_tracked_teleport_attempt = lambda *_a, **_kw: _make_tracked(
        sync_ts=2100, tp_result=_TP_TIMEOUT, tp_started=2200
    )
    try:
        result = _common_reposition_call()
        if result.terminal_result is None:
            raise AssertionError("expected terminal result")
    finally:
        equipment_target_phase_module.run_tracked_teleport_attempt = original


def test_reposition_success_propagates_teleport() -> None:
    """equipment_target_phase.py branch 494->496."""
    original = equipment_target_phase_module.run_tracked_teleport_attempt
    equipment_target_phase_module.run_tracked_teleport_attempt = lambda *_a, **_kw: _make_tracked(
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
        equipment_target_phase_module.run_tracked_teleport_attempt = original


def test_reposition_timeout_preserves_original_teleport() -> None:
    """equipment_target_phase.py branch 494->496 False path."""
    original = equipment_target_phase_module.run_tracked_teleport_attempt
    equipment_target_phase_module.run_tracked_teleport_attempt = lambda *_a, **_kw: _make_tracked(
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
        equipment_target_phase_module.run_tracked_teleport_attempt = original
