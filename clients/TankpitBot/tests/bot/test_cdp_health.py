"""Tests for the page-client health gate."""

from __future__ import annotations

from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _update_fuel_total,
)
from tests.bot._cdp_harness import _snapshot_for_health
from tests.conftest import FakeEnv


class TestPageClientHealthGate:
    """TestPageClientHealthGate tests."""

    def test_healthy_snapshot_returns_true(self) -> None:
        from tankpit_bot.bot.tick_body import _is_page_client_healthy

        assert _is_page_client_healthy(_snapshot_for_health()) is True

    def test_client_not_present_returns_false(self) -> None:
        from tankpit_bot.bot.tick_body import _is_page_client_healthy

        assert _is_page_client_healthy(_snapshot_for_health(client_present=False)) is False

    def test_websocket_closed_returns_false(self) -> None:
        from tankpit_bot.bot.tick_body import _is_page_client_healthy

        assert _is_page_client_healthy(_snapshot_for_health(ws_ready_state=3)) is False

    def test_websocket_state_none_returns_false(self) -> None:
        from tankpit_bot.bot.tick_body import _is_page_client_healthy

        assert _is_page_client_healthy(_snapshot_for_health(ws_ready_state=None)) is False

    def test_stale_heartbeat_does_not_block_dispatch(self) -> None:
        """A 30s-old transport heartbeat must NOT veto the tick.

        Regression guard for the live-run freeze (run 20260609-233736):
        ``activeGame.va.j`` only refreshes ~every 30s, so treating its
        age as staleness froze the bot ~25 of every 30 seconds while
        the WebSocket was OPEN and world updates were flowing.
        """
        from tankpit_bot.bot.tick_body import _is_page_client_healthy

        snapshot = _snapshot_for_health()
        stale = PageClientSnapshotDict(**{**snapshot, "heartbeat_age_ms": 30_000})

        assert _is_page_client_healthy(stale) is True

    def test_tick_once_returns_early_when_snapshot_unhealthy(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Tick exits before dispatch when the live snapshot reports WS closed.

        Drives the in-loop short-circuit at the snapshot-health check:
        the bot decided to dispatch radar, but the page-client snapshot
        shows the WebSocket isn't OPEN, so the tick returns before
        ``executor.execute`` runs.
        """
        from collections.abc import Callable

        from platform_core.json_utils import JSONObject

        from tankpit_bot._test_hooks import CDPSessionProtocol
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.sniffer.world_state import update_world_state_from_position

        class _UnhealthySnapshotCDP(CDPSessionProtocol):
            """CDP returning a healthy-shaped snapshot with ws_ready_state=3 (CLOSED)."""

            def __init__(self) -> None:
                self.sent_methods: list[str] = []

            def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
                _ = params
                self.sent_methods.append(method)
                if method == "Runtime.evaluate":
                    return {
                        "result": {
                            "value": {
                                "timestamp_ms": 1000,
                                "client_present": True,
                                "map_visible": False,
                                "client_state": 1,
                                "client_busy": False,
                                "pending_actions": 0,
                                "heartbeat_age_ms": 50,
                                "last_page_client_send_age_ms": 100,
                                "last_bot_send_age_ms": 100,
                                "ws_ready_state": 3,
                                "current_send_label": None,
                                "sent_frame_meta_queue_length": 0,
                                "self_fields": {},
                                "world_fields": {},
                                "map_fields": {},
                                "world_collections": {},
                            }
                        }
                    }
                return {}

            def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
                _ = (event, handler)

            def detach(self) -> None:
                return

        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        unhealthy_cdp = _UnhealthySnapshotCDP()
        bot._cdp = unhealthy_cdp
        original_state = bot.get_state()

        _tick_once(bot)

        runtime_calls = [m for m in unhealthy_cdp.sent_methods if m == "Runtime.evaluate"]
        assert runtime_calls == ["Runtime.evaluate"]
        assert bot.get_state() == original_state
