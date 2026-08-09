"""Tests for the post-teleport settle dwell and its heartbeat."""

from __future__ import annotations

from tests.action_lab._enemy_teleport_harness import (
    _ProbeHarness,
    _snapshot,
)
from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_page import ReplayClock

from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.sniffer.world_service import WorldService


def test_finish_non_teleport_attempt_resets_state_and_settles() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    result = probe._finish_non_teleport_attempt(
        page=probe._fake_page,
        cdp=StubSnapshotCDPSession(),
        acquisition_strategy="nearest_enemy",
        status="no_enemy",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100,
        fuel_before=900,
        world_timestamp_before=950,
        enemy=None,
        landing_target=None,
        message_start_index=4,
        settle_delay_ms=250,
        heartbeat_interval_ms=0,
        snapshot_before=_snapshot(900),
    )

    assert result["status"] == "no_enemy"
    assert result["message_start_index"] == 4
    assert result["message_end_index"] == 0
    assert probe.get_state() == "IDLE"
    assert probe._fake_page.waits[-1] == 250.0


def test_settle_dwell_heartbeat_walk_shuffles_in_place() -> None:
    """A positive heartbeat splits the dwell into interval steps, each
    led by a drain plus a 1-tile walk shuffle (east, back west, ...) —
    the 2026-07-24 decisive run confirmed real actions hold the push
    stream open, and the per-beat drain keeps the shuffle origin at
    the tank's true position instead of the frozen landing tile."""
    probe = _ProbeHarness()
    drains = 0

    def _counting_drain(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        nonlocal drains
        drains += 1
        return 0

    action_hooks.drain_buffered_messages = _counting_drain
    probe._settle_dwell(probe._fake_page, 4000, 1500)
    assert probe.inventory_calls == 0
    assert probe.move_calls == [(101, 100), (99, 100), (101, 100)]
    assert drains == 3
    assert probe._fake_page.waits[-3:] == [1500.0, 1500.0, 1000.0]


def test_settle_dwell_heartbeat_falls_back_to_inventory_without_self_state() -> None:
    """No self position -> the walk cannot aim; the query fallback fires."""
    probe = _ProbeHarness()
    probe._self_state = None
    probe._settle_dwell(probe._fake_page, 3000, 1500)
    assert probe.move_calls == []
    assert probe.inventory_calls == 2


def test_settle_dwell_without_heartbeat_is_one_silent_wait() -> None:
    probe = _ProbeHarness()
    probe._settle_dwell(probe._fake_page, 4000, 0)
    assert probe.inventory_calls == 0
    assert probe._fake_page.waits[-1] == 4000.0


def test_settle_dwell_zero_settle_is_a_no_op() -> None:
    """A non-positive settle dwells not at all, on either heartbeat path.

    Both heartbeat branches are exercised because only the no-heartbeat
    one can observe the guard: with a positive heartbeat the loop's own
    ``remaining > 0`` test declines the dwell anyway, so that path passed
    whether the guard existed or not (mutation survivor, 2026-08-08).
    Drop the guard and the no-heartbeat branch waits for ``0.0`` -- and a
    negative settle waits for a negative timeout.
    """
    probe = _ProbeHarness()
    baseline_waits = len(probe._fake_page.waits)
    probe._settle_dwell(probe._fake_page, 0, 1500)
    probe._settle_dwell(probe._fake_page, 0, 0)
    probe._settle_dwell(probe._fake_page, -250, 0)
    assert probe.inventory_calls == 0
    assert probe.move_calls == []
    assert len(probe._fake_page.waits) == baseline_waits
