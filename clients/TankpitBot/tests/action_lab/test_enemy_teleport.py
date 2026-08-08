"""Tests for ``probe_single_enemy_attempt``.

Every terminal outcome one attempt can reach. ``test_enemy_teleport.py``
was 1,391 lines; targeting, settle, and execution are now siblings.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import (
    Literal,
)

import pytest
from tests.action_lab._enemy_teleport_harness import (
    _enemy,
    _make_world,
    _ProbeHarness,
    enemy_module,
    enemy_targeting_module,
)
from tests.action_lab._replay_page import ReplayClock

from tankpit_bot._test_hooks import (
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_teleport import EnemyTeleportProbe
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_self_state,
)


def test_probe_single_enemy_attempt_returns_acquisition_timeout() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: None

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        heartbeat_interval_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "acquisition_timeout"
    assert result["teleport_started_ms"] is None
    assert probe.teleport_calls == []


def test_probe_single_enemy_attempt_returns_no_enemy() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _missing_enemy(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return None

    enemy_targeting_module._require_fresh_enemy_threat = _missing_enemy

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        heartbeat_interval_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "no_enemy"


def test_probe_single_enemy_attempt_returns_no_landing_tile() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _no_landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
        ws: WorldService,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms, ws)
        return (-1, -1)

    enemy_targeting_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _no_landing

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        heartbeat_interval_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "no_landing_tile"
    assert result["enemy"] == _enemy()


def test_probe_single_enemy_attempt_raises_when_teleport_dispatch_fails() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe.teleport_result = False
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
        ws: WorldService,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms, ws)
        return (119, 130)

    enemy_targeting_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing

    with pytest.raises(TeleportProbeError, match="teleport command dispatch failed"):
        probe._probe_single_enemy_attempt(
            acquisition_strategy="nearest_enemy",
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
            heartbeat_interval_ms=0,
            excluded_tank_ids=frozenset(),
        )


def test_probe_single_enemy_attempt_records_teleport_timeout() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
        ws: WorldService,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms, ws)
        return (119, 130)

    def _timeout_result(
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
            [Literal["after_map_data", "landed", "timeout"]],
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
            status="teleport_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=850,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=False,
            landed_x=100,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _enemy_after(
        probe: EnemyTeleportProbe,
        tank_id: int,
    ) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return _enemy()

    enemy_targeting_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing
    enemy_module._wait_for_teleport_outcome = _timeout_result
    enemy_targeting_module._enemy_by_id = _enemy_after

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        heartbeat_interval_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "teleport_timeout"
    assert result["enemy_still_visible"] is True


def test_probe_single_enemy_attempt_settles_after_landed_result() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._self_state = make_self_state(
        tank_id=1,
        x=119,
        y=130,
        team=2,
        rank=1,
        fuel=820,
        leaderboard_position=1,
    )
    probe._world_state = _make_world(1450, 119, 130, 820)
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
        ws: WorldService,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms, ws)
        return (119, 130)

    def _landed_result(
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
            [Literal["after_map_data", "landed", "timeout"]],
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
            fuel_after=820,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=119,
            landed_y=130,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _enemy_after(
        probe: EnemyTeleportProbe,
        tank_id: int,
    ) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return _enemy(x=120, y=130)

    enemy_targeting_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing
    enemy_module._wait_for_teleport_outcome = _landed_result
    enemy_targeting_module._enemy_by_id = _enemy_after

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=250,
        heartbeat_interval_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == "landed_adjacent"
    assert probe._fake_page.waits[-1] == 250.0

    # The LANDED-path settle must heartbeat too — the first heartbeat
    # watch run went silent because only the non-teleport path dwelled
    # (live catch, 2026-07-24): a successful watch is exactly the
    # landed path, so pin it here.
    probe.move_calls = []
    heartbeat_result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=3000,
        heartbeat_interval_ms=1500,
        excluded_tank_ids=frozenset(),
    )
    assert heartbeat_result["status"] == "landed_adjacent"
    assert len(probe.move_calls) == 2
    assert probe._fake_page.waits[-2:] == [1500.0, 1500.0]


@pytest.mark.parametrize(
    ("enemy_after", "expected_status"),
    [
        (_enemy(x=120, y=130), "landed_adjacent"),
        (_enemy(x=123, y=130), "landed_not_adjacent"),
    ],
)
def test_probe_single_enemy_attempt_records_landed_outcome(
    enemy_after: EnemyThreatDict,
    expected_status: str,
) -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._self_state = make_self_state(
        tank_id=1,
        x=119,
        y=130,
        team=2,
        rank=1,
        fuel=820,
        leaderboard_position=1,
    )
    probe._world_state = _make_world(1450, 119, 130, 820)
    action_hooks.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _enemy_found(
        probe: EnemyTeleportProbe,
        started_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        _ = (probe, started_ms, excluded_tank_ids)
        return _enemy()

    def _landing(
        world: WorldStateDict,
        self_state: SelfStateDict,
        target: EnemyThreatDict,
        terrain: TerrainMapProtocol | None,
        now_ms: int,
        ws: WorldService,
    ) -> tuple[int, int]:
        _ = (world, self_state, target, terrain, now_ms, ws)
        return (119, 130)

    def _landed_result(
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
            [Literal["after_map_data", "landed", "timeout"]],
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
            fuel_after=820,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=119,
            landed_y=130,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    def _enemy_after(
        probe: EnemyTeleportProbe,
        tank_id: int,
    ) -> EnemyThreatDict | None:
        _ = (probe, tank_id)
        return enemy_after

    enemy_targeting_module._require_fresh_enemy_threat = _enemy_found
    enemy_module.choose_combat_landing_tile = _landing
    enemy_module._wait_for_teleport_outcome = _landed_result
    enemy_targeting_module._enemy_by_id = _enemy_after

    result = probe._probe_single_enemy_attempt(
        acquisition_strategy="nearest_enemy",
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        heartbeat_interval_ms=0,
        excluded_tank_ids=frozenset(),
    )

    assert result["status"] == expected_status
    assert result["enemy_distance_after"] == (
        abs(119 - enemy_after["x"]) + abs(130 - enemy_after["y"])
    )
