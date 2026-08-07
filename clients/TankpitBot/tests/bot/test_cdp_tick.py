"""Tests for ``tick_once``: one pass of the bot's decide-dispatch loop."""

from __future__ import annotations

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
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


class TestBotTickOnce:
    """One pass of the decide-dispatch loop."""

    def test_tick_once_low_fuel_radars_for_fuel(self, fake_env: FakeEnv) -> None:
        """_tick_once uses radar when fuel low and no containers visible."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        _update_fuel_total(get_world_service(), 300)
        update_inventory_from_protocol(get_world_service(), [0, 0, 0, 0, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        # last_scan_ms=1, scan_cooldown=5000, now will be >> 5001 → cooldown elapsed
        _tick_once(bot)
        # Fuel < 500, no containers, scan cooldown elapsed → radar
        assert bot._ai_state["last_scan_ms"] > 0

    def test_tick_once_low_fuel_walks_to_edge(self, fake_env: FakeEnv) -> None:
        """_tick_once walks to viewport edge when fuel low and radar on cooldown."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        _update_fuel_total(get_world_service(), 300)
        update_inventory_from_protocol(get_world_service(), [0, 0, 0, 0, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        from tankpit_bot.bot.ai.types import AIStateDict

        # Set recent scan so radar cooldown blocks → forces walk to edge
        bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": get_current_time_ms()})
        _tick_once(bot)
        # Fuel < 500, no containers, radar on cooldown → walk to edge
        # Should have sent a move command (drain + move = 2 CDP calls)
        assert len(fake_cdp._sent_methods) >= 2

    def test_tick_once_waits_for_in_flight_movement(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while a walk is still resolving."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, InMemoryTerrainMap

        reset_world_state()
        update_world_state_from_position(10, 10)
        _update_fuel_total(get_world_service(), 800)
        update_inventory_from_protocol(get_world_service(), [5, 5, 5, 5, 5], [False] * 5)
        get_world_service().terrain_map = InMemoryTerrainMap()

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "MOVING"

    def test_tick_once_waits_for_in_flight_teleport(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while a teleport is still resolving."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(196, 85)
        _update_fuel_total(get_world_service(), 582)
        update_inventory_from_protocol(get_world_service(), [5, 5, 5, 5, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "TELEPORTING", "teleport", 196, 86)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "TELEPORTING"

    def test_tick_once_waits_for_in_flight_collection(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while pickup movement is still resolving."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, InMemoryTerrainMap

        reset_world_state()
        update_world_state_from_position(205, 79)
        _update_fuel_total(get_world_service(), 580)
        update_inventory_from_protocol(get_world_service(), [5, 5, 5, 5, 5], [False] * 5)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=205, y=80, volume=-1)]
        mines: list[RadarMineDict] = []
        _update_radar(get_world_service(), containers, mines, [])
        get_world_service().terrain_map = InMemoryTerrainMap()

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 205, 80)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "COLLECTING"

    def test_tick_once_waits_for_pending_shot_feedback(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while the last shot outcome is still pending."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)
        update_inventory_from_protocol(
            get_world_service(), [0, 10, 0, 0, 0], [False, True, False, False, False]
        )

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = get_current_time_ms()

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "IDLE"

    def test_tick_once_untracked_shoot_dispatches_and_persists(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """An id-targeted shoot at an untracked tank DISPATCHES and persists.

        Executor-side shoot validation was deleted 2026-07-21 (the tick
        is synchronous, so decide-time registry truth cannot go stale
        before dispatch). State persistence is the dispatch contract.
        (The registry keeps departed tanks -- remove_tank is a no-op --
        so untracked ids only occur for genuinely unknown tanks.)
        """
        import tankpit_bot.bot.ai_strategy as ai_strategy_mod
        from tankpit_bot._test_hooks import TerrainMapProtocol
        from tankpit_bot.bot.ai.scoring_types import make_behavior_score
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.combat_feedback import CombatFeedback
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
        from tankpit_bot.bot.types import make_shoot_command
        from tankpit_bot.inventory import InventoryState
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol
        from tankpit_bot.state.types import (
            SelfStateDict,
            WorldStateDict,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        _update_fuel_total(get_world_service(), 800)
        update_inventory_from_protocol(
            get_world_service(), [0, 10, 0, 0, 0], [False, True, False, False, False]
        )

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._magic = "test_magic"
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        original_state = AIStateDict(**bot._ai_state)
        dispatched_state = AIStateDict(
            **{
                **bot._ai_state,
                "last_shoot_ms": 12345,
                "last_shot_target_id": 10,
                "last_shot_target_name": "enemy",
                "combat_target_id": 10,
                "combat_target_x": 101,
                "combat_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_shoot_command(101, 100, 10),
            behavior=make_behavior_score("HUNT", 800, 101, 100, "shoot_target", target_id=10),
            updated_ai_state=dispatched_state,
            desired_equipment=[],
        )

        def fake_decide(
            world: WorldStateDict,
            self_state: SelfStateDict,
            ai_state: AIStateDict,
            inventory: InventoryState,
            timestamp_ms: int,
            terrain: TerrainMapProtocol | None,
            combat_feedback: CombatFeedback = "",
            map_fuel_dots: tuple[tuple[int, int], ...] = (),
        ) -> TickDecisionDict:
            _ = (
                world,
                self_state,
                ai_state,
                inventory,
                timestamp_ms,
                terrain,
                combat_feedback,
                map_fuel_dots,
            )
            return decision

        original_decide = ai_strategy_mod.decide
        try:
            ai_strategy_mod.decide = fake_decide
            _tick_once(bot)
        finally:
            ai_strategy_mod.decide = original_decide

        _ = original_state
        assert bot._ai_state == dispatched_state
        assert bot._ai_state["last_shot_target_id"] == 10
        assert bot._ai_state["combat_target_id"] == 10
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_tick_once_failed_dispatch_does_not_persist_ai_state(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A dispatch failure (CDP-level) must not advance AI state.

        Save-restore fake on executor.execute, mirroring this file's
        ai_strategy.decide idiom: the only remaining non-dispatch path
        is a page/CDP send failure, which has no fake knob.
        """
        import tankpit_bot.bot.ai_strategy as ai_strategy_mod
        from tankpit_bot._test_hooks import TerrainMapProtocol
        from tankpit_bot.bot.ai.scoring_types import make_behavior_score
        from tankpit_bot.bot.ai.types import AIStateDict
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.combat_feedback import CombatFeedback
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
        from tankpit_bot.bot.types import make_shoot_command
        from tankpit_bot.inventory import InventoryState
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol
        from tankpit_bot.state.types import (
            SelfStateDict,
            WorldStateDict,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        _update_fuel_total(get_world_service(), 800)
        update_inventory_from_protocol(
            get_world_service(), [0, 10, 0, 0, 0], [False, True, False, False, False]
        )

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._magic = "test_magic"
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        original_state = AIStateDict(**bot._ai_state)
        failed_state = AIStateDict(
            **{
                **bot._ai_state,
                "last_shoot_ms": 12345,
                "last_shot_target_id": 10,
                "last_shot_target_name": "enemy",
                "combat_target_id": 10,
                "combat_target_x": 101,
                "combat_target_y": 100,
            }
        )
        decision = make_tick_decision(
            command=make_shoot_command(101, 100, 10),
            behavior=make_behavior_score("HUNT", 800, 101, 100, "shoot_target", target_id=10),
            updated_ai_state=failed_state,
            desired_equipment=[],
        )

        def fake_decide(
            world: WorldStateDict,
            self_state: SelfStateDict,
            ai_state: AIStateDict,
            inventory: InventoryState,
            timestamp_ms: int,
            terrain: TerrainMapProtocol | None,
            combat_feedback: CombatFeedback = "",
            map_fuel_dots: tuple[tuple[int, int], ...] = (),
        ) -> TickDecisionDict:
            _ = (
                world,
                self_state,
                ai_state,
                inventory,
                timestamp_ms,
                terrain,
                combat_feedback,
                map_fuel_dots,
            )
            return decision

        import tankpit_bot.bot.executor as executor_mod
        from tankpit_bot._test_hooks import BotProtocol

        def failing_execute(
            bot: BotProtocol,
            decision: TickDecisionDict,
            snapshot: PageClientSnapshotDict,
        ) -> bool:
            _ = (bot, decision, snapshot)
            return False

        original_decide = ai_strategy_mod.decide
        original_execute = executor_mod.execute
        try:
            ai_strategy_mod.decide = fake_decide
            executor_mod.execute = failing_execute
            _tick_once(bot)
        finally:
            ai_strategy_mod.decide = original_decide
            executor_mod.execute = original_execute

        assert bot._ai_state == original_state
        assert bot._ai_state["last_shot_target_id"] == -1
        assert bot._ai_state["combat_target_id"] == -1

    def test_tick_once_dispatches_regular_radar_before_search_hop(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_tick_once scans first when regular radar is available at zero extra stock."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp

        _tick_once(bot)

        # CDP calls: snapshot read + structure survey + radar dispatch +
        # flag-binding arm (first tick on a fresh session) + overlay update.
        assert fake_cdp._sent_methods == ["Runtime.evaluate"] * 3 + [
            "Runtime.addBinding",
            "Runtime.evaluate",
        ]
        assert bot.get_state() == "SCANNING"

    def test_tick_once_critical_equipment_preempts_combat(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_tick_once searches for equipment when dual is below break threshold."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tankpit_bot.state.tank_mutations import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation, make_tank_state
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        _update_fuel_total(get_world_service(), 800)
        update_inventory_from_protocol(
            get_world_service(), [0, 3, 0, 10, 10], [False, True, False, True, True]
        )

        get_world_service().world_state = apply_tank_observation(
            get_world_service().world_state,
            make_tank_observation(
                tank_id=10,
                timestamp_ms=get_current_time_ms(),
                is_wire_sourced=True,
                storage_source="viewport",
                position=(106, 100),
                team=1,
                rank=0,
                name="enemy",
                is_bot=True,
            ),
        )
        enemy = make_tank_state(
            tank_id=10,
            x=106,
            y=100,
            team=1,
            rank=0,
            damage_state=3,
            name="enemy",
            is_bot=True,
            is_self=False,
            timestamp_ms=get_current_time_ms(),
        )
        new_tanks = dict(get_world_service().world_state["tanks"])
        new_tanks["10"] = enemy
        get_world_service().world_state["tanks"] = new_tanks

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        from tankpit_bot.bot.ai.types import AIStateDict

        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
            }
        )

        _tick_once(bot)

        # dual=3 < dual_break_threshold=4 → critical equipment preempts combat
        assert fake_cdp._sent_methods[0] == "Runtime.evaluate"
        assert "Input.dispatchKeyEvent" not in fake_cdp._sent_methods
        assert bot.get_state() == "SCANNING"

    def test_tick_once_replans_with_regular_radar_when_pending_shot_target_was_killed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Killed-target replanning uses regular radar before broader search movement."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _tick_once
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)
        update_inventory_from_protocol(
            get_world_service(), [0, 10, 0, 0, 0], [False, True, False, False, False]
        )

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = get_current_time_ms()
        bot._ai_state["killed_tank_ids"] = {"50": get_current_time_ms()}

        _tick_once(bot)

        # CDP calls: snapshot read + structure survey + radar dispatch +
        # flag-binding arm (first tick on a fresh session) + overlay update.
        assert fake_cdp._sent_methods == ["Runtime.evaluate"] * 3 + [
            "Runtime.addBinding",
            "Runtime.evaluate",
        ]
        assert bot.get_state() == "SCANNING"
