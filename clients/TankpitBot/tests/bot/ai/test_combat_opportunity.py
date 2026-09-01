"""Firefight opportunity fire: finishers, return fire, divert bookkeeping.

User ruling 2026-08-14: "you have a main target ofc, but you should
also return fire to anyone else engaging and take kill shots ... when
someone is in the lowest or second lowest damage state."
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_close import close_target
from tankpit_bot.bot.ai.combat_strategy import engage_target
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.ledger.damage_book import ConfirmedIncomingDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._combat_fixtures import _enemy_threat
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
    seed_confirmed_incoming,
)
from tests.conftest import FakeEnv
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000


def _tank(tank_id: int, x: int, y: int, *, name: str, damage_state: int = 3) -> TankStateDict:
    """Build a live, viewport-fresh enemy tank record."""
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=damage_state,
        timestamp_ms=_NOW,
        last_wire_seen_ms=_NOW,
        last_position_update_ms=_NOW,
        last_viewport_observation_ms=_NOW,
    )


def _locked_ctx(
    tanks: dict[str, TankStateDict],
    *,
    blocked: dict[str, int] | None = None,
    terrain: InMemoryTerrainMap | None = None,
    combat_feedback: CombatFeedback = "",
) -> DecideCtx:
    """Build a ctx engaged on the tank with id 50 (self at (100,100))."""
    ws = WorldService()
    world, self_state = make_world(self_x=100, self_y=100, fuel=800, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": tanks["50"]["x"],
            "combat_target_y": tanks["50"]["y"],
            "last_shot_target_id": 50,
            "blocked_combat_targets": dict(blocked) if blocked else {},
        }
    )
    return DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        _NOW,
        terrain,
        combat_feedback,
        ws=ws,
    )


def _main_target(tanks: dict[str, TankStateDict]) -> tuple[DecideCtx, EnemyThreatDict]:
    """Return a locked ctx plus the main target threat for close_target."""
    ctx = _locked_ctx(tanks)
    main = _enemy_threat(x=tanks["50"]["x"], y=tanks["50"]["y"], name="Main")
    return ctx, main


class TestFinisherDivert:
    """Kill shots on the lowest damage tiers preempt the main target."""

    def test_finisher_divert_holds_the_lock(self) -> None:
        """A visible tier-1 enemy draws the shot; the main lock survives."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 100, 103, name="red-1", damage_state=1),
        }
        ctx, main = _main_target(tanks)

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "opportunity_shot"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 60
        updated = decision["updated_ai_state"]
        assert updated["combat_target_id"] == 50
        assert updated["last_shot_target_id"] == 60

    def test_most_damaged_finisher_wins_over_attacker(self) -> None:
        """Tier 0 beats tier 1; any finisher beats a healthy attacker."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 100, 103, name="red-2", damage_state=1),
            "61": _tank(61, 100, 105, name="red-1", damage_state=0),
            "62": _tank(62, 103, 100, name="red-3"),
        }
        ctx, main = _main_target(tanks)
        ctx.ws.damage_book["confirmed_incoming"].append(
            ConfirmedIncomingDict(timestamp_ms=_NOW - 1000, cost=45, shooter_id=62)
        )

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "opportunity_shot"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 61

    def test_off_window_finisher_is_not_diverted(self) -> None:
        """A divert never moves or shifts: outside the window, no shot."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 100, 130, name="red-1", damage_state=0),
        }
        ctx, main = _main_target(tanks)

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "shoot_target"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 50

    def test_out_of_range_finisher_is_not_diverted(self) -> None:
        """In the window but beyond shot range: the main target keeps the tick."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 107, 104, name="red-1", damage_state=0),
        }
        ctx, main = _main_target(tanks)

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "shoot_target"

    def test_occluded_finisher_is_not_diverted(self) -> None:
        """A rock on the dual line voids the divert (clear-line law)."""
        terrain = InMemoryTerrainMap({(100, 101): InMemoryTerrainMap.ROCK})
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 100, 103, name="red-1", damage_state=0),
        }
        ctx = _locked_ctx(tanks, terrain=terrain)
        main = _enemy_threat(x=101, y=100, name="Main")

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "shoot_target"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 50

    def test_blocked_finisher_is_not_diverted(self) -> None:
        """A blocked id (afterimage/shield cooldown) never draws a divert."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 100, 103, name="red-1", damage_state=0),
        }
        ctx = _locked_ctx(tanks, blocked={"60": _NOW})
        main = _enemy_threat(x=101, y=100, name="Main")

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "shoot_target"


class TestHumanFightDiscipline:
    """No bot draws a divert while the main lock is a human.

    Operator ruling 2026-09-01: "we should not be engaging a bot
    during a human chase firefight." Bots respawn and their return
    fire is noise; every diverted beat is a free beat handed to the
    human. Human-vs-human diverts stay live.
    """

    def test_bot_finisher_never_diverts_a_human_fight(self) -> None:
        """A dying practice bot in view: the shot still goes to the human."""
        tanks = {
            "50": _tank(50, 101, 100, name="Beerus"),
            "60": _tank(60, 100, 103, name="red-1", damage_state=1),
        }
        ctx, main = _main_target(tanks)

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] != "opportunity_shot"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 50

    def test_bot_attacker_never_diverts_a_human_fight(self) -> None:
        """A practice bot hitting us mid-human-fight draws nothing back."""
        tanks = {
            "50": _tank(50, 101, 100, name="Beerus"),
            "60": _tank(60, 103, 100, name="red-3"),
        }
        ctx, main = _main_target(tanks)
        ctx.ws.damage_book["confirmed_incoming"].append(
            ConfirmedIncomingDict(timestamp_ms=_NOW - 1000, cost=45, shooter_id=60)
        )

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] != "opportunity_shot"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 50

    def test_second_human_finisher_still_diverts(self) -> None:
        """Human-vs-human stays live: a dying consented human draws the shot.

        Whis has hit us (consent-by-aggression, the human-combat
        gate), so the human-fight discipline does not bar him — only
        BOTS are barred from diverting a human fight.
        """
        tanks = {
            "50": _tank(50, 101, 100, name="Beerus"),
            "60": _tank(60, 100, 103, name="Whis", damage_state=1),
        }
        ctx, main = _main_target(tanks)
        seed_confirmed_incoming(ctx.ws, 1)

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "opportunity_shot"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 60


class TestReturnFire:
    """Anyone engaging us gets a shot back while they keep hitting."""

    def test_recent_attacker_draws_return_fire(self) -> None:
        """A fuel-confirmed hit inside the window diverts one shot back."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 103, 100, name="red-3"),
        }
        ctx, main = _main_target(tanks)
        ctx.ws.damage_book["confirmed_incoming"].append(
            ConfirmedIncomingDict(timestamp_ms=_NOW - 1000, cost=45, shooter_id=60)
        )

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "opportunity_shot"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 60
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_stale_hit_does_not_divert(self) -> None:
        """An attacker silent past the window stops drawing return fire."""
        tanks = {
            "50": _tank(50, 101, 100, name="blue-9"),
            "60": _tank(60, 103, 100, name="red-3"),
        }
        ctx, main = _main_target(tanks)
        ctx.ws.damage_book["confirmed_incoming"].append(
            ConfirmedIncomingDict(timestamp_ms=_NOW - 7000, cost=45, shooter_id=60)
        )

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "shoot_target"
        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_id"] == 50

    def test_main_target_attacker_is_no_divert(self) -> None:
        """The main target hitting us is the ordinary fight, not a divert."""
        tanks = {"50": _tank(50, 101, 100, name="Main")}
        ctx, main = _main_target(tanks)
        ctx.ws.damage_book["confirmed_incoming"].append(
            ConfirmedIncomingDict(timestamp_ms=_NOW - 1000, cost=45, shooter_id=50)
        )

        decision = close_target(ctx, main)

        assert decision["behavior"]["reason_kind"] == "shoot_target"


class TestDivertFeedbackScoping:
    """A divert's feedback never drives the main lock's consequences."""

    def test_diverted_miss_does_not_chase_the_lock(self) -> None:
        """Feedback keyed to a divert id leaves the lock shot untouched.

        Without the scoping, the diverted miss would run the lock's
        stationary-miss branch and trade a live point-blank fight for
        a map chase.
        """
        tanks = {"50": _tank(50, 101, 100, name="Main")}
        ctx = _locked_ctx(tanks, combat_feedback="miss")
        # The last shot was a DIVERT at id 60; its miss arrives now.
        ctx.ai_state["last_shot_target_id"] = 60
        main = _enemy_threat(x=101, y=100, name="Main")

        decision = engage_target(ctx, main)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["behavior"]["reason_kind"] == "shoot_target"

    def test_lock_miss_still_chases(self) -> None:
        """Feedback keyed to the lock keeps its stationary-miss verdict."""
        tanks = {"50": _tank(50, 101, 100, name="Main")}
        ctx = _locked_ctx(tanks, combat_feedback="miss")
        main = _enemy_threat(x=101, y=100, name="Main")

        decision = engage_target(ctx, main)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_target"


class TestDivertMissBlock:
    """A missed divert target is blocked in the feedback layer."""

    def test_diverted_miss_blocks_the_divert_and_holds_the_lock(self, fake_env: FakeEnv) -> None:
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state_combat import mark_combat_hit

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._ai_state["combat_target_id"] = 50
        bot._ai_state["last_shot_target_id"] = 60
        bot._ai_state["last_shot_target_name"] = "red-1"
        mark_combat_hit(ws, weapon_byte=0, victim_id=-1)

        result = _get_combat_feedback(bot)

        assert result == "miss"
        assert "60" in bot._ai_state["blocked_combat_targets"]
        assert bot._ai_state["combat_target_id"] == 50

    def test_lock_miss_is_not_blocked_here(self, fake_env: FakeEnv) -> None:
        """The engage path owns the lock's miss verdict, not this hook."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_combat_feedback import _get_combat_feedback
        from tankpit_bot.sniffer.world_state_combat import mark_combat_hit

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._ai_state["combat_target_id"] = 50
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Main"
        mark_combat_hit(ws, weapon_byte=0, victim_id=-1)

        result = _get_combat_feedback(bot)

        assert result == "miss"
        assert "50" not in bot._ai_state["blocked_combat_targets"]
