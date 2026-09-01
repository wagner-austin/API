"""Threat ordering: the fleet assist/spread split and the points floor.

Split from ``test_threats.py`` when it crossed the 600-line ceiling
(2026-09-01). These classes pin the sort components the 2026-09-01
session added: sibling locks focus HUMANS and spread PRACTICE BOTS,
and rank-0 recruits (the measured no-points tier, [[game-rules]])
yield the tick to paying targets.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot.ai._threat_fixtures import _self_at, _tank, _world


class TestFleetAssistRanking:
    """Sibling locks focus humans and spread bots (2026-09-01 split).

    Operator observation, verbatim: "All four bots are locked on the
    same target ... stacked on a practice bot in World rather than
    spreading." A sibling-engaged HUMAN is the session's stake and
    ranks first; a sibling-engaged PRACTICE BOT is a respawner four
    tanks should never queue on, so it ranks LAST.
    """

    def test_fleet_engaged_human_ranks_before_a_nearer_bot(self) -> None:
        """Focus fire on humans: the ally's consented human wins."""
        ws = WorldService()
        ws.fleet_engaged_target_ids = {60: 99000}
        ws.fleet_consented_tank_ids = {60}
        world = _world(
            {
                "50": _tank("50", x=103, y=100, team=2, name="red-1"),
                "60": _tank("60", x=108, y=100, team=2, name="Beerus", is_bot=False),
            }
        )

        threats = analyze_threats(ws, world, _self_at(), now_ms=0)

        assert [threat["tank_id"] for threat in threats] == [60, 50]

    def test_fleet_engaged_bot_spreads_to_the_free_one(self) -> None:
        """A sibling's locked practice bot ranks last, not first."""
        ws = WorldService()
        ws.fleet_engaged_target_ids = {50: 99000}
        world = _world(
            {
                "50": _tank("50", x=103, y=100, team=2, name="red-1"),
                "60": _tank("60", x=108, y=100, team=2, name="red-2"),
            }
        )

        threats = analyze_threats(ws, world, _self_at(), now_ms=0)

        assert [threat["tank_id"] for threat in threats] == [60, 50]

    def test_no_fleet_locks_keeps_distance_order(self) -> None:
        """With an empty assist set the nearest enemy leads."""
        ws = WorldService()
        world = _world(
            {
                "50": _tank("50", x=103, y=100, team=2, name="red-1"),
                "60": _tank("60", x=108, y=100, team=2, name="red-2"),
            }
        )

        threats = analyze_threats(ws, world, _self_at(), now_ms=0)

        assert [threat["tank_id"] for threat in threats] == [50, 60]


class TestPointsFloorRanking:
    """A paying kill outranks a free one; recruits stay huntable.

    The points-floor law, measured 2026-09-01 across 19 World kills
    with zero contradictions ([[game-rules]]): rank-0 victims pay
    "Enemy's rank was too low" to every killer rank, rank 1+ pays
    extra points to every killer rank. The sort component orders, it
    never excludes — a map of nothing but recruits still gets hunted
    for its spoils.
    """

    def test_paying_target_outranks_a_nearer_recruit(self) -> None:
        """The rank-1 bot at distance 8 beats the recruit at distance 3."""
        ws = WorldService()
        world = _world(
            {
                "50": _tank("50", x=103, y=100, team=2, name="red-1", rank=0),
                "60": _tank("60", x=108, y=100, team=2, name="red-2", rank=1),
            }
        )

        threats = analyze_threats(ws, world, _self_at(), now_ms=0)

        assert [threat["tank_id"] for threat in threats] == [60, 50]

    def test_bot_spread_outranks_the_points_preference(self) -> None:
        """A sibling's locked paying bot still yields to the free recruit.

        The spread component is senior to points: queuing on a
        sibling's kill wastes the fleet's beats however well it pays.
        """
        ws = WorldService()
        ws.fleet_engaged_target_ids = {60: 99000}
        world = _world(
            {
                "50": _tank("50", x=103, y=100, team=2, name="red-1", rank=0),
                "60": _tank("60", x=108, y=100, team=2, name="red-2", rank=1),
            }
        )

        threats = analyze_threats(ws, world, _self_at(), now_ms=0)

        assert [threat["tank_id"] for threat in threats] == [50, 60]
