"""Tests for combat probe live execution paths via DI harnesses."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from tests.action_lab._combat_probe_harness import (
    _ExecuteHarness,
    _ProbeHarness,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.combat_probe import (
    _current_enemy_by_id,
    _enemy_from_world_state,
    _find_fresh_enemy,
    _wait_for_shot_feedback,
)
from tankpit_bot.action_lab.combat_probe_types import (
    CombatEngagementDict,
)
from tankpit_bot.bot.ai.world_types import make_enemy_threat
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.state import (
    make_tank_state,
)


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_sync_playwright = core_hooks.sync_playwright
    yield
    action_hooks.get_current_time_ms = original_get_time
    core_hooks.sync_playwright = original_sync_playwright
    reset_world_state()


def test_find_fresh_enemy_returns_closest_fresh() -> None:
    """_find_fresh_enemy returns closest enemy confirmed after started_ms."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=109,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-31",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
        "51": make_tank_state(
            tank_id=51,
            x=102,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-32",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
    }

    result = _find_fresh_enemy(probe, 1000, frozenset())

    if result is None:
        pytest.fail("expected closest fresh enemy")
    assert result["tank_id"] == 51


def test_find_fresh_enemy_returns_none_without_self_state() -> None:
    probe = _ProbeHarness()
    probe._self_state = None
    assert _find_fresh_enemy(probe, 0, frozenset()) is None


def test_find_fresh_enemy_excludes_ids() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=102,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-33",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
        "51": make_tank_state(
            tank_id=51,
            x=105,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-34",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
    }

    result = _find_fresh_enemy(probe, 1000, frozenset({50}))

    if result is None:
        pytest.fail("expected enemy after exclusion")
    assert result["tank_id"] == 51


def test_current_enemy_by_id_finds_match() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-35",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }

    result = _current_enemy_by_id(probe, 50)
    if result is None:
        pytest.fail("expected enemy by id")
    assert result["tank_id"] == 50


def test_current_enemy_by_id_returns_none_when_missing() -> None:
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    assert _current_enemy_by_id(probe, 99) is None


def test_current_enemy_by_id_returns_none_without_self_state() -> None:
    probe = _ProbeHarness()
    probe._self_state = None
    assert _current_enemy_by_id(probe, 50) is None


def test_enemy_from_world_state_returns_position() -> None:
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=130,
            y=140,
            team=1,
            rank=1,
            damage_state=0,
            name="red-31",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }

    result = _enemy_from_world_state(probe, 50)
    assert result == (130, 140)


def test_enemy_from_world_state_returns_none_at_origin() -> None:
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=0,
            y=0,
            team=1,
            rank=1,
            damage_state=0,
            name="dead",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }

    assert _enemy_from_world_state(probe, 50) is None


def test_enemy_from_world_state_returns_none_when_absent() -> None:
    probe = _ProbeHarness()
    assert _enemy_from_world_state(probe, 999) is None


def test_wait_for_shot_feedback_returns_hit() -> None:
    """Simulates a confirmed hit by pre-setting world service flags."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    ws = get_world_service()
    ws.got_our_shot_response = True
    ws.got_confirmed_hit = True

    page = ClockAdvancingPage(clock)
    got_response, was_hit = _wait_for_shot_feedback(page, probe)

    assert got_response is True
    assert was_hit is True


def test_wait_for_shot_feedback_returns_miss() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    ws = get_world_service()
    ws.got_our_shot_response = True
    ws.got_confirmed_hit = False

    page = ClockAdvancingPage(clock)
    got_response, was_hit = _wait_for_shot_feedback(page, probe)

    assert got_response is True
    assert was_hit is False


def test_wait_for_shot_feedback_returns_timeout() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()

    page = ClockAdvancingPage(clock)
    got_response, was_hit = _wait_for_shot_feedback(page, probe)

    assert got_response is False
    assert was_hit is False


def test_engage_single_target_records_shots() -> None:
    """Single engagement records per-shot data and detects kill."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    ws = get_world_service()

    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-35",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }
    enemy = make_enemy_threat(
        tank_id=50,
        x=101,
        y=100,
        distance=1,
        damage_state=0,
        rank=1,
        team=1,
        name="red-35",
        is_bot=False,
        timestamp_ms=1000,
    )

    ws.got_our_shot_response = True
    ws.got_confirmed_hit = True

    def _on_wait() -> None:
        ws.got_our_shot_response = True
        ws.got_confirmed_hit = True
        ws.killed_tank_ids.add(50)

    probe._fake_page = ClockAdvancingPage(clock, on_wait=_on_wait)

    result = probe._engage_single_target(enemy, max_shots=5)

    assert result["target_id"] == 50
    assert result["total_hits"] >= 1
    assert result["kill_confirmed"] is True
    assert result["shots"][0]["result"] == "hit"


def test_engage_single_target_detects_flee() -> None:
    """Engagement detects target movement as flee."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()
    ws = get_world_service()

    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="runner",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }
    enemy = make_enemy_threat(
        tank_id=50,
        x=101,
        y=100,
        distance=1,
        damage_state=0,
        rank=1,
        team=1,
        name="runner",
        is_bot=False,
        timestamp_ms=1000,
    )

    shot_count = 0

    def _on_wait() -> None:
        nonlocal shot_count
        shot_count += 1
        ws.got_our_shot_response = True
        ws.got_confirmed_hit = True
        if shot_count == 1:
            probe._world_state["tanks"]["50"] = make_tank_state(
                tank_id=50,
                x=105,
                y=100,
                team=1,
                rank=1,
                damage_state=2,
                name="runner",
                is_bot=False,
                is_self=False,
                timestamp_ms=1000,
            )
        elif shot_count >= 2:
            ws.killed_tank_ids.add(50)

    probe._fake_page = ClockAdvancingPage(clock, on_wait=_on_wait)
    ws.got_our_shot_response = True
    ws.got_confirmed_hit = True

    result = probe._engage_single_target(enemy, max_shots=5)

    assert result["target_fled"] is True
    assert result["kill_confirmed"] is True


def test_engage_exits_on_target_gone_from_world_state() -> None:
    """Engagement exits when target disappears from world state entirely."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()

    enemy = make_enemy_threat(
        tank_id=50,
        x=101,
        y=100,
        distance=1,
        damage_state=0,
        rank=1,
        team=1,
        name="gone",
        is_bot=False,
        timestamp_ms=1000,
    )

    result = probe._engage_single_target(enemy, max_shots=5)

    assert result["kill_confirmed"] is True
    assert len(result["shots"]) == 0


def test_engage_exits_on_timeout() -> None:
    """Engagement stops on shot feedback timeout."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness()

    probe._world_state["tanks"] = {
        "50": make_tank_state(
            tank_id=50,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="silent",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }
    enemy = make_enemy_threat(
        tank_id=50,
        x=101,
        y=100,
        distance=1,
        damage_state=0,
        rank=1,
        team=1,
        name="silent",
        is_bot=False,
        timestamp_ms=1000,
    )

    probe._fake_page = ClockAdvancingPage(clock)

    result = probe._engage_single_target(enemy, max_shots=5)

    assert result["total_timeouts"] == 1
    assert len(result["shots"]) == 1
    assert result["shots"][0]["result"] == "timeout"


def _make_engagement(
    target_id: int = 50,
    target_name: str = "target",
) -> CombatEngagementDict:
    return CombatEngagementDict(
        target_id=target_id,
        target_name=target_name,
        initial_target_x=101,
        initial_target_y=100,
        initial_distance=1,
        landed_x=100,
        landed_y=100,
        shots=[],
        total_hits=0,
        total_misses=0,
        total_timeouts=0,
        kill_confirmed=False,
        target_fled=False,
        final_target_x=101,
        final_target_y=100,
        final_distance=1,
    )


def test_execute_probe_raises_on_zero_engagements() -> None:
    harness = _ExecuteHarness()
    with pytest.raises(ValueError, match="max_engagements must be positive"):
        harness.execute_probe(
            max_engagements=0,
            max_shots_per_engagement=10,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=5000,
            teleport_timeout_ms=10000,
        )
