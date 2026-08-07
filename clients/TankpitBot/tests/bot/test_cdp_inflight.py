"""Tests for the in-flight action guards.

Blocked-walk and blocked-collection clearing, the stall timeouts, and
the map-open sync hold.
"""

from __future__ import annotations

from tankpit_bot.bot.states import (
    make_in_flight_action,
)
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_state import get_world_service
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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import InMemoryTerrainMap

        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 800)
        get_world_service().terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import InMemoryTerrainMap

        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 800)
        get_world_service().terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import InMemoryTerrainMap

        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        _update_radar(get_world_service(), containers, mines, [])
        get_world_service().terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import InMemoryTerrainMap

        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 580)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=12, y=10, volume=-1)]
        mines: list[RadarMineDict] = []
        _update_radar(get_world_service(), containers, mines, [])
        get_world_service().terrain_map = InMemoryTerrainMap()

        bot = Bot("https://test.tankpit.com/", headless=True)
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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import InMemoryTerrainMap

        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        _update_radar(get_world_service(), containers, mines, [])
        get_world_service().terrain_map = InMemoryTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_move(self, fake_env: FakeEnv) -> None:
        """Stalled movement times out so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_collection(self, fake_env: FakeEnv) -> None:
        """Stalled collection times out so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(64, 64)
        _update_fuel_total(get_world_service(), 800)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=72, y=63, volume=-1)]
        mines: list[RadarMineDict] = []
        _update_radar(get_world_service(), containers, mines, [])

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 72, 63)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_teleport(self, fake_env: FakeEnv) -> None:
        """Stalled teleport times out so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(64, 64)
        _update_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "TELEPORTING", "teleport", 128, 128)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tankpit_bot.state.types import WorldStateDict

        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0, started_ms=started_ms)
        get_world_service().world_state = WorldStateDict(
            **{**get_world_service().world_state, "timestamp_ms": started_ms}
        )

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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tankpit_bot.state.types import WorldStateDict

        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0, started_ms=started_ms)
        # Bump world_state timestamp the way an unrelated sync would --
        # this MUST NOT clear the action; the old proxy gate would have
        # fired here.
        get_world_service().world_state = WorldStateDict(
            **{**get_world_service().world_state, "timestamp_ms": started_ms + 1}
        )

        assert has_in_flight_action(bot) is True
        kind_before_signal = bot._state_data["in_flight_action"]["kind"]
        assert kind_before_signal == "map_open"

        # Now mark the authoritative MAP_DATA signal; the wait should clear.
        get_world_service().mark_map_data_processed()

        assert has_in_flight_action(bot) is False
        kind_after_signal = bot._state_data["in_flight_action"]["kind"]
        assert kind_after_signal == "none"

    def test_stalled_map_open_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A map_open that stalls past timeout clears so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 1400)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_stall_guard_prevents_clear_when_started_ms_is_zero(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A zero started_ms prevents the stall timeout from firing."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.states import InFlightActionDict
        from tankpit_bot.bot.tick_loop_actions import _clear_stalled_action

        bot = Bot("https://test.tankpit.com/", headless=True)
        action: InFlightActionDict = make_in_flight_action(
            "move",
            15,
            10,
            0,
        )

        result = _clear_stalled_action(bot, action)

        assert result is False

    def test_fresh_scan_does_not_trigger_stall_timeout(self, fake_env: FakeEnv) -> None:
        """A recently started scan does not trigger stall timeout."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_stalled_action

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)
        action = bot._state_data["in_flight_action"]

        assert _clear_stalled_action(bot, action) is False
        assert bot.get_state() == "SCANNING"

    def test_stalled_scan_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A scan that stalls past timeout clears so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            is_scan_viewport_failed,
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 1400)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"
        assert is_scan_viewport_failed(0, 0, get_current_time_ms()) is True

    def test_stalled_move_marks_failed_move_target(self, fake_env: FakeEnv) -> None:
        """Stalled move records the destination as a failed move target."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 1400)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 73, 158)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        has_in_flight_action(bot)

        now = get_current_time_ms()
        assert is_move_target_failed(73, 158, now) is True
        assert is_move_target_failed(74, 158, now) is False

    def test_clear_blocked_collection_returns_false_when_adjacent(self, fake_env: FakeEnv) -> None:
        """Adjacent collection remains viable even if the target tile itself is blocked."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_blocked_collection
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import InMemoryTerrainMap

        update_world_state_from_position(14, 10)
        _update_fuel_total(get_world_service(), 400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        _update_radar(get_world_service(), containers, mines, [])
        get_world_service().terrain_map = InMemoryTerrainMap({(15, 10): "#"})

        bot = Bot("https://test.tankpit.com/", headless=True)
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
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        update_world_state_from_position(100, 100)
        _update_fuel_total(get_world_service(), 300)
        update_inventory_from_protocol(get_world_service(), [0, 0, 0, 0, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "SCANNING"
