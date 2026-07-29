"""Tests for the bot-service wiring inside the tick loop.

Covers ``_apply_pending_mode_override``, ``_publish_session_status``,
and ``_sync_live_view_demand`` against a real :class:`Bot` with a
real :class:`ModeBridge` / :class:`StatusBus` / :class:`FrameBus` (no
mocks — the primitives are the DUT). The wire is: SPA writes to the
bridge → tick loop drains it → ai_state carries the override → tick
loop publishes a status frame reflecting it; ``/video`` subscribers
on the frame bus toggle the in-page live-view caster.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.tick_loop import (
    _apply_pending_mode_override,
    _publish_session_status,
    _sync_live_view_demand,
)
from tankpit_bot.service.frame_bus import FrameBus
from tankpit_bot.service.mode_bridge import ModeBridge
from tankpit_bot.service.status_bus import StatusBus


class _RecordingCDP:
    """CDP-session fake that records live-view evaluate sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, JSONObject | None]] = []
        self.handlers: dict[str, Callable[[JSONObject], None]] = {}

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        self.sent.append((method, params))
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        self.handlers[event] = handler

    def detach(self) -> None:
        raise AssertionError("the demand sync never detaches the session")


class TestApplyPendingModeOverride:
    """Contract for draining the mode bridge into the AI state."""

    def test_empty_bridge_leaves_manual_mode_untouched(self) -> None:
        """An empty bridge does not disturb ``ai_state["manual_mode"]``."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        assert bot._ai_state["manual_mode"] is None

        _apply_pending_mode_override(bot)

        assert bot._ai_state["manual_mode"] is None

    def test_hunt_pin_translates_to_ai_mode(self) -> None:
        """A submitted ``"HUNT"`` becomes ``manual_mode = "HUNT"``."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        bridge.submit("HUNT")

        _apply_pending_mode_override(bot)

        assert bot._ai_state["manual_mode"] == "HUNT"

    def test_collect_pin_translates_to_ai_mode(self) -> None:
        """A submitted ``"COLLECT"`` becomes ``manual_mode = "COLLECT"``."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        bridge.submit("COLLECT")

        _apply_pending_mode_override(bot)

        assert bot._ai_state["manual_mode"] == "COLLECT"

    def test_unset_pin_translates_to_ai_mode(self) -> None:
        """A submitted ``"UNSET"`` becomes ``manual_mode = "UNSET"``."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        bridge.submit("UNSET")

        _apply_pending_mode_override(bot)

        assert bot._ai_state["manual_mode"] == "UNSET"

    def test_auto_pin_restores_none(self) -> None:
        """A submitted ``"AUTO"`` clears ``manual_mode`` back to ``None``."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        # Start with a pinned HUNT — proves AUTO reverses it.
        bridge.submit("HUNT")
        _apply_pending_mode_override(bot)
        assert bot._ai_state["manual_mode"] == "HUNT"

        bridge.submit("AUTO")
        _apply_pending_mode_override(bot)

        assert bot._ai_state["manual_mode"] is None

    def test_drain_consumes_the_pending_override(self) -> None:
        """The drained override does not fire again on the next tick."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        bridge.submit("HUNT")
        _apply_pending_mode_override(bot)
        assert bot._ai_state["manual_mode"] == "HUNT"

        # Second call, bridge is empty — must not reset to None.
        _apply_pending_mode_override(bot)

        assert bot._ai_state["manual_mode"] == "HUNT"


class TestPublishSessionStatus:
    """Contract for building and publishing the session status frame."""

    def test_publish_reaches_subscriber_with_current_state(self) -> None:
        """The subscriber receives a frame reflecting ``bot._ai_state``."""
        bus = StatusBus()
        bot = Bot("https://test.tankpit.com/", status_bus=bus)
        subscriber = bus.subscribe()
        bot._start_timestamp_ms = 500

        _publish_session_status(bot)

        frame = subscriber.next_frame(timeout=0.5)
        if frame is None:
            raise AssertionError("expected a published status frame")
        assert frame["running"] is True
        assert frame["session_started_ms"] == 500
        assert frame["manual_mode"] == "AUTO"
        assert frame["active_mode"] == "UNSET"
        assert frame["active_mode_state"] == ""
        assert frame["stats"]["kills"] == 0
        assert frame["stats"]["hits"] == 0
        assert frame["stats"]["misses"] == 0
        assert frame["stats"]["radars_used"] == 0
        assert frame["stats"]["teleports"] == 0

    def test_publish_reflects_manual_hunt_pin(self) -> None:
        """A HUNT-pinned bot publishes ``manual_mode = "HUNT"``."""
        from tankpit_bot.bot.ai.types import AIStateDict

        bus = StatusBus()
        bot = Bot("https://test.tankpit.com/", status_bus=bus)
        bot._ai_state = AIStateDict(**{**bot._ai_state, "manual_mode": "HUNT"})
        subscriber = bus.subscribe()
        bot._start_timestamp_ms = 900

        _publish_session_status(bot)

        frame = subscriber.next_frame(timeout=0.5)
        if frame is None:
            raise AssertionError("expected a published status frame")
        assert frame["manual_mode"] == "HUNT"

    def test_publish_reflects_live_counters(self) -> None:
        """Advanced counters flow through to the published stats."""
        from tankpit_bot.bot.ai.types import AIStateDict

        bus = StatusBus()
        bot = Bot("https://test.tankpit.com/", status_bus=bus)
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "session_kill_count": 3,
                "session_hit_count": 12,
                "session_miss_count": 5,
                "live_radars_used": 7,
                "live_teleports": 9,
            }
        )
        subscriber = bus.subscribe()
        bot._start_timestamp_ms = 0

        _publish_session_status(bot)

        frame = subscriber.next_frame(timeout=0.5)
        if frame is None:
            raise AssertionError("expected a published status frame")
        assert frame["stats"]["kills"] == 3
        assert frame["stats"]["hits"] == 12
        assert frame["stats"]["misses"] == 5
        assert frame["stats"]["radars_used"] == 7
        assert frame["stats"]["teleports"] == 9

    def test_publish_with_no_subscribers_is_a_noop(self) -> None:
        """Publishing without subscribers does not raise."""
        bus = StatusBus()
        bot = Bot("https://test.tankpit.com/", status_bus=bus)
        bot._start_timestamp_ms = 0

        _publish_session_status(bot)  # must not raise

        assert bus.subscriber_count() == 0


class TestBotServiceDefaults:
    """Contract for the default ``ModeBridge`` / ``StatusBus`` on ``Bot``."""

    def test_bot_gets_a_default_mode_bridge_when_none_supplied(self) -> None:
        """A standalone ``make bot`` bot gets an inert bridge."""
        bot = Bot("https://test.tankpit.com/")
        # Bridge is empty and stays empty — no HTTP handler is writing.
        assert bot._mode_bridge.drain() is None
        assert bot._mode_bridge.peek() is None

    def test_bot_gets_a_default_status_bus_when_none_supplied(self) -> None:
        """A standalone ``make bot`` bot gets an empty bus."""
        bot = Bot("https://test.tankpit.com/")
        assert bot._status_bus.subscriber_count() == 0

    def test_supplied_mode_bridge_is_used_directly(self) -> None:
        """The Bot uses the injected bridge as its own reference."""
        bridge = ModeBridge()
        bot = Bot("https://test.tankpit.com/", mode_bridge=bridge)
        assert bot._mode_bridge is bridge

    def test_supplied_status_bus_is_used_directly(self) -> None:
        """The Bot uses the injected bus as its own reference."""
        bus = StatusBus()
        bot = Bot("https://test.tankpit.com/", status_bus=bus)
        assert bot._status_bus is bus

    def test_bot_gets_a_default_frame_bus_when_none_supplied(self) -> None:
        """A standalone ``make bot`` bot gets an empty frame bus."""
        bot = Bot("https://test.tankpit.com/")
        assert bot._frame_bus.subscriber_count() == 0

    def test_supplied_frame_bus_is_used_directly(self) -> None:
        """The Bot uses the injected frame bus as its own reference."""
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", frame_bus=frames)
        assert bot._frame_bus is frames


class TestSyncLiveViewDemand:
    """Demand-driven in-page caster toggling at the tick boundary."""

    def test_noop_before_cdp_attach(self) -> None:
        """No CDP session yet → nothing happens even with demand."""
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", frame_bus=frames)
        frames.subscribe()

        _sync_live_view_demand(bot)

        assert bot._live_view.active is False

    def test_viewer_demand_installs_the_caster(self) -> None:
        """A frame-bus subscriber makes the next tick install the caster."""
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", frame_bus=frames)
        cdp = _RecordingCDP()
        bot._cdp = cdp
        frames.subscribe()

        _sync_live_view_demand(bot)

        assert bot._live_view.active is True
        methods = [method for method, _ in cdp.sent]
        assert methods == ["Runtime.addBinding", "Runtime.evaluate"]

    def test_no_demand_with_inactive_caster_stays_inactive(self) -> None:
        """Zero subscribers and no caster → nothing is sent."""
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", frame_bus=frames)
        cdp = _RecordingCDP()
        bot._cdp = cdp

        _sync_live_view_demand(bot)

        assert bot._live_view.active is False
        assert cdp.sent == []

    def test_last_viewer_leaving_stops_the_caster(self) -> None:
        """Demand dropping to zero stops the caster at the next tick."""
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", frame_bus=frames)
        cdp = _RecordingCDP()
        bot._cdp = cdp
        subscriber = frames.subscribe()
        _sync_live_view_demand(bot)
        assert bot._live_view.active is True

        frames.unsubscribe(subscriber)
        _sync_live_view_demand(bot)

        assert bot._live_view.active is False
        assert len(cdp.sent) == 3  # addBinding + caster install + stop

    def test_sustained_demand_reensures_every_tick(self) -> None:
        """Continuing demand re-evaluates the idempotent snippet per tick.

        The repetition is the navigation self-heal: quit-to-lobby or
        a re-login wipes injected JS, and the next demanded tick
        reinstalls the caster without any navigation detection.
        """
        frames = FrameBus()
        bot = Bot("https://test.tankpit.com/", frame_bus=frames)
        cdp = _RecordingCDP()
        bot._cdp = cdp
        frames.subscribe()

        _sync_live_view_demand(bot)
        _sync_live_view_demand(bot)
        _sync_live_view_demand(bot)

        assert bot._live_view.active is True
        assert len(cdp.sent) == 4  # one addBinding + three caster installs
