"""Tests for the Bot's AI integration over a CDP session."""

from __future__ import annotations

from tankpit_bot.browser import _test_hooks as browser_hooks
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _update_fuel_total,
)
from tankpit_bot.sniffer.world_state_inventory import (
    update_inventory_from_protocol,
)
from tankpit_bot.state.types import (
    make_self_state,
    make_viewport_state,
)
from tests.bot._cdp_harness import _make_snapshot
from tests.conftest import FakeEnv


class TestBotAIIntegration:
    """TestBotAIIntegration tests."""

    def test_tick_once_no_self_state(self, fake_env: FakeEnv) -> None:
        """_tick_once returns early when self_state is None."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once

        bot = Bot("https://test.tankpit.com/", headless=True)
        _tick_once(bot)
        # AI state unchanged — no self_state to act on
        assert bot._ai_state["mode"] == "UNSET"

    def test_tick_once_returns_if_self_state_disappears_after_sync(self, fake_env: FakeEnv) -> None:
        """_tick_once aborts when the refreshed world loses self_state mid-tick."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.state.types import WorldStateDict

        class FlakyWorldBot(Bot):
            """Bot whose world state disappears on the second read."""

            def __init__(self, target_url: str, *, headless: bool) -> None:
                super().__init__(target_url, headless=headless)
                self._world_reads = 0
                self._state_data["state"] = "IDLE"

            def get_world_state(self) -> WorldStateDict:
                """Return populated state once, then lose self.

                Read 1 feeds the first self check; the disappearance
                must hit the post-sync re-read to cover the mid-tick
                abort branch.
                """
                self._world_reads += 1
                if self._world_reads <= 1:
                    return WorldStateDict(
                        self_state=make_self_state(
                            tank_id=1,
                            x=100,
                            y=100,
                            team=1,
                            rank=1,
                            fuel=800,
                            leaderboard_position=1,
                        ),
                        tanks={},
                        containers={},
                        mines={},
                        terrain={},
                        viewport=make_viewport_state(left=91, top=91, width=18, height=18),
                        scanned_tiles={},
                        timestamp_ms=0,
                    )
                return WorldStateDict(
                    self_state=None,
                    tanks={},
                    containers={},
                    mines={},
                    terrain={},
                    viewport=make_viewport_state(left=91, top=91, width=18, height=18),
                    scanned_tiles={},
                    timestamp_ms=0,
                )

            def _update_state_from_world(self) -> None:
                """Keep the bot in IDLE for this targeted tick-loop test."""

        bot = FlakyWorldBot("https://test.tankpit.com/", headless=True)

        _tick_once(bot)

        assert bot._world_reads == 2

    def test_tick_once_enforces_autoscroll_once_on_first_spawned_tick(
        self, fake_env: FakeEnv
    ) -> None:
        """The first tick with a spawned tank runs the toggle dance once.

        The enforcement rides the tick loop because the world service
        is pull-fed -- the 23:08/23:16 launches proved a pre-loop wait
        starves forever on a state nothing was draining yet. The
        two-read flaky world fires the enforcement on read 1 (spawned)
        and aborts on read 2, keeping the tick shallow; the second
        tick must not re-enforce.
        """
        from tankpit_bot._test_hooks import CDPSessionProtocol, PageWaitProtocol
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.state.types import WorldStateDict
        from tankpit_bot.types.message import CapturedMessage
        from tests.fakes.cdp import FakeCDPSession
        from tests.fakes.page import FakePage

        class SpawnedOnceBot(Bot):
            """Bot whose world is spawned on read 1, gone on read 2."""

            def __init__(self, target_url: str, *, headless: bool) -> None:
                super().__init__(target_url, headless=headless)
                self._world_reads = 0
                self._state_data["state"] = "IDLE"

            def get_world_state(self) -> WorldStateDict:
                self._world_reads += 1
                if self._world_reads <= 1:
                    return WorldStateDict(
                        self_state=make_self_state(
                            tank_id=1,
                            x=100,
                            y=100,
                            team=1,
                            rank=1,
                            fuel=800,
                            leaderboard_position=1,
                        ),
                        tanks={},
                        containers={},
                        mines={},
                        terrain={},
                        viewport=make_viewport_state(left=91, top=91, width=18, height=18),
                        scanned_tiles={},
                        timestamp_ms=0,
                    )
                return WorldStateDict(
                    self_state=None,
                    tanks={},
                    containers={},
                    mines={},
                    terrain={},
                    viewport=make_viewport_state(left=91, top=91, width=18, height=18),
                    scanned_tiles={},
                    timestamp_ms=0,
                )

            def _update_state_from_world(self) -> None:
                """Keep the bot in IDLE for this targeted tick-loop test."""

        calls: list[int] = []

        def _recorder(
            page: PageWaitProtocol,
            cdp: CDPSessionProtocol,
            messages: list[CapturedMessage],
            ws: WorldService,
        ) -> None:
            del page, cdp, messages, ws
            calls.append(1)

        original = browser_hooks.ensure_autoscroll_off
        browser_hooks.ensure_autoscroll_off = _recorder
        try:
            bot = SpawnedOnceBot("https://test.tankpit.com/", headless=True)
            session = FakeCDPSession()
            bot._page = FakePage(session)
            bot._cdp = session

            _tick_once(bot)
            assert calls == [1]
            assert bot._autoscroll_enforced is True

            bot._world_reads = 0
            _tick_once(bot)
            assert calls == [1]
        finally:
            browser_hooks.ensure_autoscroll_off = original

    def test_tick_once_waits_for_position_before_planning(self, fake_env: FakeEnv) -> None:
        """_tick_once does not execute AI commands while state is WAITING_FOR_POSITION."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(209, 79)
        _update_fuel_total(ws, 345)
        update_inventory_from_protocol(ws, [25, 25, 25, 25, 11], [False, True, False, False, True])

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._magic = "test_magic"

        _tick_once(bot)

        assert bot.get_state() == "WAITING_FOR_POSITION"
        assert fake_cdp._sent_methods == []

    def test_tick_once_nothing_to_do_opens_map(self, fake_env: FakeEnv) -> None:
        """_tick_once opens map when nothing to do (no enemies, no containers)."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        _update_fuel_total(ws, 1200)
        update_inventory_from_protocol(ws, [30, 30, 30, 30, 30], [True] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        # Set last_scan_ms so radar doesn't fire first
        from tankpit_bot.bot.ai.types import AIStateDict

        bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": 1})
        _tick_once(bot)
        # No enemies, no containers → map_open to find enemies
        assert bot._ai_state["mode"] == "HUNT"
        assert bot._ai_state["last_map_open_ms"] > 0

    def test_tick_once_hunt_with_enemy(self, fake_env: FakeEnv) -> None:
        """_tick_once dispatches HUNT when enemy is nearby."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.state.tank_mutations import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation, make_tank_state
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        _update_fuel_total(ws, 1200)
        update_inventory_from_protocol(ws, [30, 30, 30, 30, 30], [True] * 5)

        # Add a close, damaged enemy to trigger HUNT mode

        ws.world_state = apply_tank_observation(
            ws.world_state,
            make_tank_observation(
                tank_id=10,
                timestamp_ms=get_current_time_ms(),
                is_wire_sourced=True,
                storage_source="viewport",
                position=(110, 100),
                team=1,
                rank=0,
                name="enemy",
                is_bot=True,
            ),
        )
        # Set damage to critical so homing gets enabled
        tank = make_tank_state(
            tank_id=10,
            x=110,
            y=100,
            team=1,
            rank=0,
            damage_state=3,
            name="enemy",
            is_bot=True,
            is_self=False,
            timestamp_ms=get_current_time_ms(),
        )
        new_tanks = dict(ws.world_state["tanks"])
        new_tanks["10"] = tank
        ws.world_state["tanks"] = new_tanks

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        _tick_once(bot)
        assert bot._ai_state["mode"] == "HUNT"

    def test_tick_once_updates_ai_state(self, fake_env: FakeEnv) -> None:
        """_tick_once persists updated AI state after tick."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        _update_fuel_total(ws, 1200)
        update_inventory_from_protocol(ws, [30, 30, 30, 30, 30], [True] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        from tankpit_bot.bot.ai.types import AIStateDict

        bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": 1})
        _tick_once(bot)
        # decide() persists updated AI state — last_map_open_ms should be set
        assert bot._ai_state["last_map_open_ms"] > 0

    def test_dispatch_command_move(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches move to move_to."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_move_command
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        dispatch_command(bot, make_move_command(100, 100), _make_snapshot())
        assert bot.get_state() == "MOVING"

    def test_dispatch_command_pickup_fuel(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches pickup_fuel to pickup_fuel_to."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_pickup_fuel_command
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        dispatch_command(bot, make_pickup_fuel_command(100, 100), _make_snapshot())
        assert bot.get_state() == "COLLECTING"

    def test_dispatch_command_shoot(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches shoot to shoot_at."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_shoot_command
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        dispatch_command(bot, make_shoot_command(55, 53), _make_snapshot())  # Within viewport
        assert bot.get_state() == "COMBAT"

    def test_dispatch_command_radar(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches radar to use_radar."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_radar_command
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        dispatch_command(bot, make_radar_command(), _make_snapshot())
        assert bot.get_state() == "SCANNING"

    def test_dispatch_command_teleport(self, fake_env: FakeEnv) -> None:
        """With the map open, dispatch_command sends the teleport directly."""

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_teleport_command
        from tankpit_bot.sniffer.world_state_containers import (
            update_world_state_from_fuel_total,
        )
        from tests.fakes import FakeCDPSession, FakePage

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        # The hop must be affordable or the executor's refusal
        # prediction (physics/supervisor.py) suppresses the send:
        # (50,50) -> (150,150) costs 848 against 1100 fuel.
        update_world_state_from_fuel_total(ws, 1100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        # The previous tick's dispatch WAS the open (the deferral
        # contract): the overlay alone no longer certifies the map.
        ws.last_wire_command_name = "map_open"
        dispatch_command(bot, make_teleport_command(150, 150), _make_snapshot(map_visible=True))
        assert bot.get_state() == "TELEPORTING"

    def test_dispatch_command_teleport_defers_until_map_open(self, fake_env: FakeEnv) -> None:
        """With the map closed, the tick opens the map instead of teleporting.

        A teleport dispatched in the same tick as the wire map_open is
        silently dropped by the server (run 20260610-024x: 4/15 lost vs
        0/21 with the map already open), so the executor never sends
        both in one tick.
        """

        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_teleport_command
        from tests.fakes import FakeCDPSession, FakePage

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"

        achieved = dispatch_command(
            bot, make_teleport_command(200, 200), _make_snapshot(map_visible=False)
        )

        assert achieved is True
        assert bot.get_state() == "IDLE"
        assert bot._state_data["in_flight_action"]["kind"] == "map_open"
