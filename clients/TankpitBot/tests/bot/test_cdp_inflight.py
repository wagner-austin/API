"""Tests for the in-flight action guards.

Blocked-walk and blocked-collection clearing, the map-open sync hold,
and the scope completion hold. The stall timeouts are
:mod:`tests.bot.test_cdp_inflight_stalls`.
"""

from __future__ import annotations

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _update_fuel_total,
)
from tankpit_bot.sniffer.world_state_inventory import (
    update_inventory_from_protocol,
)
from tankpit_bot.sniffer.world_state_radar import (
    update_world_state_from_radar as _update_radar,
)
from tests.bot._cdp_harness import _sba
from tests.conftest import FakeEnv


class TestBotInFlightGuards:
    """In-flight action guards, stall timeouts, and blocked-state clearing."""

    def test_clear_blocked_walk_resets_state(self, fake_env: FakeEnv) -> None:
        """Blocked walking clears MOVING so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_walk
        from tests.fakes import InMemoryTerrainMap

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 800)
        ws.terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        action = bot._state_data["in_flight_action"]

        cleared = _clear_blocked_walk(bot, action)

        assert cleared is True
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_blocked_walk(self, fake_env: FakeEnv) -> None:
        """Blocked walking returns False from the in-flight gate after clearing state."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tests.fakes import InMemoryTerrainMap

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 800)
        ws.terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_clear_blocked_walk_returns_false_without_self_state(self, fake_env: FakeEnv) -> None:
        """Blocked-walk helper does nothing when self position is unknown."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_walk

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)

        action = bot._state_data["in_flight_action"]
        assert _clear_blocked_walk(bot, action) is False
        assert bot.get_state() == "MOVING"

    def test_clear_blocked_collection_resets_state(self, fake_env: FakeEnv) -> None:
        """Blocked collection clears COLLECTING so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_collection
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tests.fakes import InMemoryTerrainMap

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        _update_radar(ws, containers, mines, [])
        ws.terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)
        action = bot._state_data["in_flight_action"]

        cleared = _clear_blocked_collection(bot, action)

        assert cleared is True
        assert bot.get_state() == "IDLE"

    def test_clear_blocked_collection_returns_false_when_viewport_path_exists(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Reachable collection remains in flight when the viewport path is valid."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_collection
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tests.fakes import InMemoryTerrainMap

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 580)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=12, y=10, volume=-1)]
        mines: list[RadarMineDict] = []
        _update_radar(ws, containers, mines, [])
        ws.terrain_map = InMemoryTerrainMap()

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 12, 10)
        action = bot._state_data["in_flight_action"]

        cleared = _clear_blocked_collection(bot, action)

        assert cleared is False
        assert bot.get_state() == "COLLECTING"

    def test_has_in_flight_action_clears_blocked_collection(self, fake_env: FakeEnv) -> None:
        """Blocked collection returns False from the in-flight gate after clearing state."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tests.fakes import InMemoryTerrainMap

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        _update_radar(ws, containers, mines, [])
        ws.terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_false_for_shoot_kind(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Shoot actions are not blocking for replanning."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "COMBAT", "shoot", 50, 50)

        assert has_in_flight_action(bot) is False

    def test_has_in_flight_action_waits_for_pending_map_open_sync(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """map_open waits until at least one fresh world sync arrives."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.state.types import WorldStateDict

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0, started_ms=started_ms)
        ws.world_state = WorldStateDict(**{**ws.world_state, "timestamp_ms": started_ms})

        assert has_in_flight_action(bot) is True
        assert bot._state_data["in_flight_action"]["kind"] == "map_open"

    def test_has_in_flight_action_holds_map_open_until_map_data_processed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """map_open does not clear on incidental syncs; only on MAP_DATA arrival.

        Asserts the authoritative-signal contract: bumping ``timestamp_ms``
        through an unrelated ``TankStatus`` / ``ViewportUpdate`` must NOT
        clear the pending map_open. Only :func:`mark_map_data_processed`
        -- called by the dispatcher when a MAP_DATA blob is decoded -- is
        the legitimate completion signal.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.state.types import WorldStateDict

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0, started_ms=started_ms)
        # Bump world_state timestamp the way an unrelated sync would --
        # this MUST NOT clear the action; the old proxy gate would have
        # fired here.
        ws.world_state = WorldStateDict(**{**ws.world_state, "timestamp_ms": started_ms + 1})

        assert has_in_flight_action(bot) is True
        kind_before_signal = bot._state_data["in_flight_action"]["kind"]
        assert kind_before_signal == "map_open"

        # Now mark the authoritative MAP_DATA signal; the wait should clear.
        ws.mark_map_data_processed()

        assert has_in_flight_action(bot) is False
        kind_after_signal = bot._state_data["in_flight_action"]["kind"]
        assert kind_after_signal == "none"

    def test_clear_blocked_collection_returns_false_when_adjacent(self, fake_env: FakeEnv) -> None:
        """Adjacent collection remains viable even if the target tile itself is blocked."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_collection
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tests.fakes import InMemoryTerrainMap

        ws = WorldService()
        ws.update_world_state_from_position(14, 10)
        _update_fuel_total(ws, 400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        _update_radar(ws, containers, mines, [])
        ws.terrain_map = InMemoryTerrainMap({(15, 10): "#"})

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        action = bot._state_data["in_flight_action"]
        assert _clear_blocked_collection(bot, action) is False
        assert bot.get_state() == "COLLECTING"

    def test_clear_blocked_collection_returns_false_without_self_state(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Blocked-collection helper does nothing when self position is unknown."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_collection

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        action = bot._state_data["in_flight_action"]
        assert _clear_blocked_collection(bot, action) is False
        assert bot.get_state() == "COLLECTING"

    def test_tick_once_waits_for_pending_scan(self, fake_env: FakeEnv) -> None:
        """_tick_once does not fire new commands while radar results are pending."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        _update_fuel_total(ws, 300)
        update_inventory_from_protocol(ws, [0, 0, 0, 0, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "SCANNING"


class TestScopeInFlight:
    """The tracked pan: hold until 0x5A, stall as the drop's only exit.

    Regression pins for the scope-pending radar drop
    ([[viewport-shift-protocol]], 2026-08-20): while a pan is in
    flight the tick loop must not plan — the hold is what makes
    dispatching radar or map_open into the unsettled window
    unrepresentable.
    """

    def test_has_in_flight_action_holds_scope_until_viewport_update(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A pending pan holds the tick loop until its 0x5A is ingested.

        Incidental world syncs must NOT clear it — only
        ``mark_viewport_update_processed`` (the dispatcher's 0x5A
        ingestion) is the completion signal, mirroring map_open's
        MAP_DATA contract.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.state.types import WorldStateDict

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "scope", 0, 0, started_ms=started_ms)
        ws.world_state = WorldStateDict(**{**ws.world_state, "timestamp_ms": started_ms + 1})

        assert has_in_flight_action(bot) is True
        kind_before_signal = bot._state_data["in_flight_action"]["kind"]
        assert kind_before_signal == "scope"

        ws.mark_viewport_update_processed()

        assert has_in_flight_action(bot) is False
        kind_after_signal = bot._state_data["in_flight_action"]["kind"]
        assert kind_after_signal == "none"
        outcomes = [record["outcome"] for record in ws.ledger.rings["scope"]]
        assert outcomes == ["confirmed"]

    def test_stalled_scope_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A dropped pan's only exit is the stall timeout, typed as scope.

        The server silently drops pans in rare windows (no charge, no
        0x5A — archive max healthy confirm is 8 s, timeout is 10 s);
        the stall clears the hold so the bot replans, and the ledger
        books a ``scope:stall_timeout``.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "IDLE", "scope", 0, 0)
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot._state_data["in_flight_action"]["kind"] == "none"
        outcomes = [record["outcome"] for record in ws.ledger.rings["scope"]]
        assert outcomes == ["stall_timeout"]
