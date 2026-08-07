"""Tests for the human-consent combat contract.

User ruling 2026-07-30 (session 8 killed over it): "to engage in
combat, the human must respond hello or engage the bot first." The
bot teleports a few tiles off the human, says HELLO, and only a chat
response or a first strike consents them into combat.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import (
    GREETING_STANDOFF_TILES,
    choose_greeting_landing_tile,
)
from tankpit_bot.bot.ai.threat_acquisition import find_acquisition_target
from tankpit_bot.bot.ai.threat_primitives import (
    human_combat_consented,
    make_enemy_threat_from_tank,
)
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.ledger.damage_book import record_incoming_shot
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
    make_tank_state,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _human_world(name: str = "Yuppler") -> tuple[WorldStateDict, SelfStateDict]:
    world = make_empty_world_state()
    self_state = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    tank = make_tank_state(
        tank_id=1229,
        x=110,
        y=100,
        team=1,
        rank=1,
        damage_state=0,
        name=name,
        is_bot=False,
        is_self=False,
    )
    tank["timestamp_ms"] = 100000
    tank["last_viewport_observation_ms"] = 100000
    world["tanks"]["1229"] = tank
    return world, self_state


class TestHumanCombatConsented:
    """The consent predicate's two signals."""

    def test_unknown_human_is_not_consented(self) -> None:
        """No chat and no incoming damage means no consent."""
        assert human_combat_consented(1229) is False

    def test_chat_consents(self) -> None:
        """Any non-self-echo chat from their id consents them."""
        get_world_service().chat_seen_tank_ids.add(1229)
        assert human_combat_consented(1229) is True

    def test_first_strike_consents(self) -> None:
        """An incoming shot from their id consents them."""
        record_incoming_shot(get_world_service().damage_book, 1229, "Yuppler", 1, 100000)
        assert human_combat_consented(1229) is True


class TestConsentGates:
    """Unconsented humans never enter targeting; consented ones do."""

    def test_unconsented_human_excluded_from_threats(self) -> None:
        """The viewport threat list drops an unconsented human."""
        world, self_state = _human_world()

        assert analyze_threats(world, self_state, now_ms=100000) == []

    def test_consented_human_enters_threats(self) -> None:
        """A chat response admits the human to the threat list."""
        world, self_state = _human_world()
        get_world_service().chat_seen_tank_ids.add(1229)

        threats = analyze_threats(world, self_state, now_ms=100000)

        assert [t["tank_id"] for t in threats] == [1229]

    def test_unconsented_human_rejected_at_acquisition(self) -> None:
        """Map acquisition refuses the human with human_not_consented."""
        world, self_state = _human_world()

        result = find_acquisition_target(
            world,
            self_state,
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        assert result is None

    def test_attacked_bot_may_acquire_the_attacker(self) -> None:
        """A first strike consents the human into acquisition."""
        world, self_state = _human_world()
        record_incoming_shot(get_world_service().damage_book, 1229, "Yuppler", 1, 100000)

        result = find_acquisition_target(
            world,
            self_state,
            blocked={},
            killed={},
            terrain=None,
            now_ms=100000,
            map_open_cooldown_ms=5000,
            engagement_reserve_fuel=650,
        )

        if result is None:
            raise AssertionError("attacker must be acquirable")
        assert result["tank_id"] == 1229

    def test_practice_bots_need_no_consent(self) -> None:
        """The consent contract never touches practice bots."""
        world, self_state = _human_world(name="red-6")

        threats = analyze_threats(world, self_state, now_ms=100000)

        assert [t["tank_id"] for t in threats] == [1229]


class TestGreetingLandingTile:
    """The a-few-tiles-off greeting landing chooser."""

    def test_lands_in_the_standoff_band_never_adjacent(self) -> None:
        """The landing sits in the greeting band, tie toward self."""
        world, self_state = _human_world()
        target = make_enemy_threat_from_tank(world["tanks"]["1229"], 10)
        terrain = InMemoryTerrainMap()

        landing = choose_greeting_landing_tile(world, self_state, target, terrain, 100000)

        if landing is None:
            raise AssertionError("open terrain must yield a greeting landing")
        ring = abs(landing[0] - 110) + abs(landing[1] - 100)
        assert GREETING_STANDOFF_TILES - 1 <= ring <= GREETING_STANDOFF_TILES + 1
        # Tie toward self: the band tile facing (100,100) wins.
        assert landing == (104, 100)

    def test_unknown_terrain_yields_no_landing(self) -> None:
        """Without terrain no landing can be vouched for."""
        world, self_state = _human_world()
        target = make_enemy_threat_from_tank(world["tanks"]["1229"], 10)

        assert choose_greeting_landing_tile(world, self_state, target, None, 100000) is None

    def test_never_falls_back_inside_the_band(self) -> None:
        """With only near tiles passable the chooser declines entirely."""
        world, self_state = _human_world()
        target = make_enemy_threat_from_tank(world["tanks"]["1229"], 10)
        near_only = {
            (110 + dx, 100 + dy)
            for dx in range(-3, 4)
            for dy in range(-3, 4)
            if abs(dx) + abs(dy) <= 3
        }
        terrain = InMemoryTerrainMap.from_passable_set(near_only)

        assert choose_greeting_landing_tile(world, self_state, target, terrain, 100000) is None


class TestConsentEdgeCoverage:
    """Edge branches of the consent stack's scanners and choosers."""

    def test_rank_protected_human_never_enters_threats_even_consented(self) -> None:
        """The rank window still outranks consent in the threat list."""
        world, self_state = _human_world()
        world["tanks"]["1229"]["rank"] = 0
        get_world_service().chat_seen_tank_ids.add(1229)

        assert analyze_threats(world, self_state, now_ms=100000) == []

    def test_greeting_landing_clips_map_corner(self) -> None:
        """A human near the map corner only considers in-bounds band tiles."""
        world, self_state = _human_world()
        world["tanks"]["1229"]["x"] = 2
        world["tanks"]["1229"]["y"] = 2
        target = make_enemy_threat_from_tank(world["tanks"]["1229"], 10)
        terrain = InMemoryTerrainMap()

        landing = choose_greeting_landing_tile(world, self_state, target, terrain, 100000)

        if landing is None:
            raise AssertionError("corner human must still get a greeting landing")
        assert 0 <= landing[0] <= 255 and 0 <= landing[1] <= 255

    def test_greeting_landing_skips_occupied_band_tiles(self) -> None:
        """A tank standing on the preferred band tile is skipped."""
        world, self_state = _human_world()
        blocker = make_tank_state(
            tank_id=60,
            x=104,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-60",
            is_bot=True,
            is_self=False,
            # A blocking body must be viewport-fresh under the
            # occupancy law -- a stale entry no longer vetoes.
            last_viewport_observation_ms=100000,
        )
        world["tanks"]["60"] = blocker
        target = make_enemy_threat_from_tank(world["tanks"]["1229"], 10)
        terrain = InMemoryTerrainMap()

        landing = choose_greeting_landing_tile(world, self_state, target, terrain, 100000)

        if landing is None:
            raise AssertionError("an occupied band tile must not kill the landing")
        assert landing != (104, 100)
