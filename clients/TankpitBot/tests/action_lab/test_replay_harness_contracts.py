"""Explicit seam/contract tests for the action-lab replay harness.

The :mod:`tests.action_lab._replay_harness` module is a load-bearing
test infrastructure piece -- if the page-client snapshot it derives
from world state, the page substitute it hands the probe, or the
frame-batch cursor it uses to feed recorded payloads ever drift from
the production interfaces they target, dozens of replay tests fail in
hard-to-diagnose ways.

These tests pin the harness's seams **explicitly** at the unit level:

* The derived snapshot satisfies the real
  :func:`tankpit_bot.browser.page_client_snapshot.decode_page_client_snapshot`.
* The page substitute satisfies
  :class:`tankpit_bot._test_hooks.PageProtocol`.
* The frame cursor honors its declared cursor semantics.
* The replay clock advances monotonically.

Each test runs in microseconds and is purely contract-shaped; the
end-to-end replay tests in :mod:`tests.action_lab.test_replay_movement_probe`
exercise the same surfaces in integration.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject
from tests.action_lab._replay_cdp import (
    WorldStateDerivedCDP,
    build_world_derived_snapshot,
)
from tests.action_lab._replay_page import (
    FrameBatchSource,
    ReplayClock,
    ReplayPage,
)

from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
    decode_page_client_snapshot,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)


def _seed_self_state(ws: WorldService, x: int, y: int, fuel: int) -> WorldStateDict:
    """Seed ``ws`` with a known self_state.

    Replaces the service's self_state entry so the derived snapshot has
    something concrete to project.
    """
    world = ws.get_world_state()
    world["self_state"] = make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=2,
        rank=3,
        fuel=fuel,
        leaderboard_position=5,
    )
    return world


def test_world_derived_snapshot_decodes_through_production_decoder() -> None:
    """The harness's derived snapshot satisfies the real snapshot decoder.

    The page-client snapshot decoder is what the production probe uses
    to validate ``Runtime.evaluate`` responses; the replay harness must
    produce a JSON object that survives the same decoder unchanged.
    """
    ws = WorldService()
    _seed_self_state(ws, x=131, y=126, fuel=1100)

    raw = build_world_derived_snapshot(ws)
    decoded: PageClientSnapshotDict = decode_page_client_snapshot(raw)

    assert decoded["client_present"] is True
    assert decoded["self_fields"] == {"x": 131, "y": 126, "fuel": 1100}


def test_world_derived_snapshot_handles_missing_self_state() -> None:
    """A pre-sync world state still produces a valid snapshot."""
    ws = WorldService()
    world = ws.get_world_state()
    world["self_state"] = None

    raw = build_world_derived_snapshot(ws)
    decoded = decode_page_client_snapshot(raw)

    assert decoded["client_present"] is True
    assert decoded["self_fields"] == {}


def test_world_state_derived_cdp_returns_snapshot_payload() -> None:
    """``WorldStateDerivedCDP.send`` returns the same payload shape live CDP returns."""
    ws = WorldService()
    _seed_self_state(ws, x=146, y=110, fuel=934)
    cdp = WorldStateDerivedCDP(ws)

    response = cdp.send("Runtime.evaluate", {"expression": "irrelevant"})

    result = response.get("result")
    if not isinstance(result, dict):
        pytest.fail("CDP response missing result envelope")
    value = result.get("value")
    if not isinstance(value, dict):
        pytest.fail("CDP response value is not a JSON object")
    decoded = decode_page_client_snapshot(value)
    assert decoded["self_fields"]["x"] == 146


def test_world_state_derived_cdp_ignores_non_runtime_evaluate_methods() -> None:
    """Non-evaluate CDP methods return an empty value (harness compat)."""
    cdp = WorldStateDerivedCDP(WorldService())

    response = cdp.send("Network.enable", None)

    assert response == {"result": {"value": None}}


def test_world_state_derived_cdp_on_and_detach_are_noops() -> None:
    """The harness CDP substitute records no handlers and never detaches."""
    cdp = WorldStateDerivedCDP(WorldService())
    received: list[JSONObject] = []

    def _handler(payload: JSONObject) -> None:
        """Record any payload the substitute might forward (it should not)."""
        received.append(payload)

    cdp.on("Network.webSocketFrameReceived", _handler)
    cdp.detach()

    assert received == []


def test_replay_page_satisfies_page_protocol() -> None:
    """The replay page exposes every method real production code consumes."""
    source = FrameBatchSource(payloads=[], batch_size=1)
    clock = ReplayClock()

    class _SinkBuffer:
        """Minimal target satisfying the harness's CDP-buffer protocol."""

        def __init__(self) -> None:
            self._cdp_message_buffer: list[str] = []
            self.xor_table: bytes | None = None

    sink = _SinkBuffer()
    page: PageProtocol = ReplayPage(sink, source, clock)

    assert page.url == "https://tankpit.com/play"
    page.keyboard.press("a")
    page.keyboard.type("hello")
    page.goto("https://example.com")
    page.wait_for_timeout(100.0)
    page.wait_for_event("close")
    page.wait_for_function("true")
    page.close()
    assert page.evaluate("1+1") is None


def test_replay_page_wait_advances_clock_and_feeds_frames() -> None:
    """``wait_for_timeout`` mirrors the production poll-and-drain step."""
    source = FrameBatchSource(payloads=["AAA", "BBB", "CCC"], batch_size=1)
    clock = ReplayClock()

    class _SinkBuffer:
        """Minimal target satisfying the harness's CDP-buffer protocol."""

        def __init__(self) -> None:
            self._cdp_message_buffer: list[str] = []
            self.xor_table: bytes | None = None

    sink = _SinkBuffer()
    page = ReplayPage(sink, source, clock)

    page.wait_for_timeout(100.0)
    page.wait_for_timeout(100.0)

    assert clock.now_ms == 200
    assert sink._cdp_message_buffer == ["AAA", "BBB"]
    assert page.frames_fed == 2
    assert page.waits_ms == [100.0, 100.0]


def test_replay_page_wait_past_exhaustion_still_advances_clock() -> None:
    """When the recorded frames run out, the clock still ticks toward timeout."""
    source = FrameBatchSource(payloads=["AAA"], batch_size=1)
    clock = ReplayClock()

    class _SinkBuffer:
        """Minimal target satisfying the harness's CDP-buffer protocol."""

        def __init__(self) -> None:
            self._cdp_message_buffer: list[str] = []
            self.xor_table: bytes | None = None

    sink = _SinkBuffer()
    page = ReplayPage(sink, source, clock)

    page.wait_for_timeout(100.0)
    page.wait_for_timeout(100.0)
    page.wait_for_timeout(100.0)

    assert clock.now_ms == 300
    assert sink._cdp_message_buffer == ["AAA"]
    assert page.frames_fed == 1


def test_frame_batch_source_walks_payloads_in_order() -> None:
    """The cursor returns batches in order and exposes the consumed count."""
    source = FrameBatchSource(payloads=["A", "B", "C", "D", "E"], batch_size=2)

    first = source.next_batch()
    second = source.next_batch()
    third = source.next_batch()
    fourth = source.next_batch()

    assert first == ["A", "B"]
    assert second == ["C", "D"]
    assert third == ["E"]
    assert fourth == []
    assert source.consumed == 5


def test_frame_batch_source_empty_payloads_returns_empty_batch() -> None:
    """The cursor handles an empty stream without raising."""
    source = FrameBatchSource(payloads=[], batch_size=5)

    assert source.next_batch() == []
    assert source.consumed == 0


def test_replay_clock_advances_monotonically() -> None:
    """``advance`` is additive; calls without advance return the same value."""
    clock = ReplayClock(now_ms=1000)

    first = clock()
    clock.advance(50)
    second = clock()
    clock.advance(150)
    third = clock()

    assert (first, second, third) == (1000, 1050, 1200)


def test_replay_clock_is_callable_for_hook_substitution() -> None:
    """``__call__`` matches the ``Callable[[], int]`` hook signature."""
    from collections.abc import Callable

    def _consume(time_fn: Callable[[], int]) -> int:
        """Invoke a wall-clock-shaped callable and return its value."""
        return time_fn()

    clock = ReplayClock(now_ms=42)
    assert _consume(clock) == 42


def test_world_derived_snapshot_reflects_world_state_timestamp() -> None:
    """Snapshot timestamp tracks the live world-state singleton."""
    ws = WorldService()
    world = ws.get_world_state()
    world["timestamp_ms"] = 12_345
    _seed_self_state(ws, x=1, y=2, fuel=3)

    raw = build_world_derived_snapshot(ws)

    assert raw["timestamp_ms"] == 12_345


def test_world_derived_snapshot_round_trips_world_state() -> None:
    """A fresh empty world projects to a valid, deterministic snapshot."""
    ws = WorldService()
    world = ws.get_world_state()
    fresh = make_empty_world_state()
    world.update(fresh)

    raw_first = build_world_derived_snapshot(ws)
    raw_second = build_world_derived_snapshot(ws)

    assert raw_first == raw_second
    assert decode_page_client_snapshot(raw_first)["client_present"] is True
