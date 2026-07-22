"""Step (e) part 1: the divergence-zero soak and its negative control.

The Phase 3 fuel/ammo books run INSIDE the production ``WorldService``
the seam feeds, so every seam round already keeps the double-entry
accounts. The soak makes their verdict explicit: a multi-round sim
session under a stepped clock must judge at least one accounting
window and record ZERO physics divergences — in the book counters AND
in the captured ``events.jsonl`` stream.

The negative control proves the detector has teeth: a deliberately
corrupted fuel sync delivered through the same production ingestion
path MUST fire ``physics_divergence``. A soak that can't fail is not
evidence.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.bot.tick_loop import _tick_once
from tankpit_bot.protocol.types import TankStatusSyncDict
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.sim.opponent import decide_opponent
from tankpit_bot.sim.session import deliver_batch
from tankpit_bot.sniffer.world_state import get_world_service
from tests.conftest import FakeFileSystem
from tests.sim.seam import (
    RICH_CONTAINERS,
    SEAM_CLIENT_ID,
    SEAM_ENEMY_ID,
    SeamClock,
    boot_seam,
)


def _captured_events(fake_fs: FakeFileSystem, latest_events_path: str) -> list[JSONObject]:
    """Parse the captured events.jsonl stream into records.

    Args:
        fake_fs: The fake file system holding the artifacts.
        latest_events_path: Path of the latest-events JSONL artifact.

    Returns:
        The decoded event records, in emission order.
    """
    files = fake_fs.get_written_files()
    lines = files[latest_events_path].strip().splitlines()
    return [narrow_json_to_dict(load_json_str(line)) for line in lines]


def _divergence_events(records: list[JSONObject]) -> list[JSONObject]:
    """Filter the physics-divergence diagnostics from an event stream.

    Args:
        records: Decoded event records.

    Returns:
        The records whose ``diagnostic_kind`` is ``physics_divergence``.
    """
    return [r for r in records if r.get("diagnostic_kind") == "physics_divergence"]


def test_seam_soak_is_divergence_free(fake_fs: FakeFileSystem) -> None:
    """30 production rounds against the sim judge as physics-clean.

    Positive controls first — the session must have actually played
    (commands crossed the seam, the fuel book judged at least one
    window, the ammo book anchored at least one snapshot, events
    flowed) — and then the verdict: zero divergences in both book
    counters and zero ``physics_divergence`` events in the captured
    stream.
    """
    artifacts = configure_bot_runtime_logging("20260722-000001")
    clock = SeamClock(100_000)
    original_clock: Callable[[], int] = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    try:
        bot, server, link, _table = boot_seam(enemy_fuel=4000, containers=RICH_CONTAINERS)
        for _ in range(30):
            _tick_once(bot)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
            clock.advance(1000)
        _tick_once(bot)
    finally:
        _test_hooks.get_current_time_ms = original_clock
    ws = get_world_service()
    assert len(link.sent_commands) >= 10
    assert ws.fuel_book["windows"] >= 1
    assert ws.ammo_book["snapshots"] >= 1
    records = _captured_events(fake_fs, artifacts["latest_events_path"])
    assert records != []
    assert ws.fuel_book["divergences"] == 0
    assert ws.ammo_book["divergences"] == 0
    assert _divergence_events(records) == []


def test_fighting_soak_is_divergence_free(fake_fs: FakeFileSystem) -> None:
    """A soak under RETURN FIRE still judges as physics-clean.

    The scripted opponent (``sim.opponent``) shoots and dodges while
    the production bot plays — the first sim session where the
    client-side channels a passive world never touches actually run:
    incoming 0x53 echoes, armor absorption on the ammo book, and the
    fuel book's enemy-hit feasibility entries. Positive controls
    demand the fight happened (the enemy fired, the client's shields
    or fuel actually paid); the verdict is still zero divergences in
    both books and zero ``physics_divergence`` events.
    """
    artifacts = configure_bot_runtime_logging("20260722-000003")
    clock = SeamClock(100_000)
    original_clock: Callable[[], int] = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    try:
        bot, server, link, _table = boot_seam(
            enemy_fuel=6000,
            containers=RICH_CONTAINERS,
            enemy_counts=(5, 25, 0, 25, 5),
        )
        start_shields = server.world["tanks"][SEAM_CLIENT_ID]["counts"][0]
        for _ in range(24):
            _tick_once(bot)
            opponent_command = decide_opponent(server.world, SEAM_ENEMY_ID, SEAM_CLIENT_ID)
            if opponent_command is not None:
                server.queue_command(SEAM_ENEMY_ID, opponent_command)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
            clock.advance(1000)
        _tick_once(bot)
    finally:
        _test_hooks.get_current_time_ms = original_clock
    ws = get_world_service()
    truth = server.world["tanks"][SEAM_CLIENT_ID]
    assert ws.ammo_book["enemy_shots"] > 0
    assert truth["counts"][0] <= start_shields
    assert ws.fuel_book["windows"] >= 1
    assert ws.fuel_book["divergences"] == 0
    assert ws.ammo_book["divergences"] == 0
    records = _captured_events(fake_fs, artifacts["latest_events_path"])
    assert _divergence_events(records) == []


def test_detector_fires_on_corrupted_fuel_sync(fake_fs: FakeFileSystem) -> None:
    """A corrupted fuel sync through the real ingestion path MUST fire.

    After a short clean warmup, the sim delivers a self fuel sync
    whose value is a lie (+700 with no announcing gain), followed by a
    quiet reading of the same value so the fuel book closes the block.
    The book must count a divergence and the ``physics_divergence``
    diagnostic must land in the captured event stream — proof the soak
    above is a real verdict, not a detector that can't fail.
    """
    artifacts = configure_bot_runtime_logging("20260722-000002")
    clock = SeamClock(100_000)
    original_clock: Callable[[], int] = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    try:
        bot, server, link, _table = boot_seam()
        for _ in range(4):
            _tick_once(bot)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
            clock.advance(1000)
        truth = server.world["tanks"][SEAM_CLIENT_ID]
        corrupted = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=truth["team"],
            tank_id=SEAM_CLIENT_ID,
            damage_state=truth["damage_state"],
            rank=truth["rank"],
            lb_score=0,
            promo_state=0,
            fuel=truth["fuel"] + 700,
        )
        deliver_batch(bot._cdp_message_buffer, [corrupted, corrupted], link)
        clock.advance(1000)
        _tick_once(bot)
    finally:
        _test_hooks.get_current_time_ms = original_clock
    ws = get_world_service()
    assert ws.fuel_book["divergences"] >= 1
    records = _captured_events(fake_fs, artifacts["latest_events_path"])
    assert _divergence_events(records) != []
