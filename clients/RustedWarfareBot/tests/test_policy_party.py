"""The shared party discipline, exercised on its own.

The holders (raid, hunt, dive) are tested through their own behaviours;
what is pinned here is the contract they share: a draft comes whole from
the gathered or not at all, in id order so two runs of one seed draft
identically; the fastest-first draft the dive uses ranks by the
catalogue's speed and still returns id order; the road home is an
attack-move because it crosses the same ground the road out did; and the
detachment's bookkeeping sends an order once per objective per member
([[policy-raid]], [[issuing-orders]]).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.party import Detachment, draft_fastest, draft_gathered, homeward
from rw_bot.wire.state import Entity
from tests.wire_fixtures import entity


def _tank(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


def _hover(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "hoverTank", x=x, y=y)


def _stats(type_name: str, speed: float) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=350,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE: dict[str, UnitStats] = {
    "c_tank": _stats("c_tank", 1.1),
    "hoverTank": _stats("hoverTank", 2.0),
}

_ANCHOR = entity(1, "commandCenter", x=0.0, y=0.0)


def test_a_draft_is_whole_and_in_id_order() -> None:
    army = (_tank(22), _tank(20), _tank(21))
    assert draft_gathered(army, _ANCHOR, 2) == [20, 21]


def test_too_few_gathered_is_no_draft_at_all() -> None:
    """Half a party is the v1 conveyor; the gathering ground must hold one."""
    army = (_tank(20), _tank(21, x=5000.0))
    assert draft_gathered(army, _ANCHOR, 2) == []


def test_only_units_at_the_anchor_are_gathered() -> None:
    army = (_tank(20), _tank(21), _tank(22, x=5000.0))
    assert draft_gathered(army, _ANCHOR, 3) == []
    assert draft_gathered(army, _ANCHOR, 2) == [20, 21]


def test_the_fastest_draft_takes_speed_first_and_returns_id_order() -> None:
    """Two hovers gathered behind three tanks by id: the hovers go, and the
    party is reported in id order like every other draft."""
    army = (_tank(20), _tank(21), _tank(22), _hover(31), _hover(30))
    assert draft_fastest(army, _ANCHOR, 2, _CATALOGUE) == [30, 31]
    assert draft_fastest(army, _ANCHOR, 3, _CATALOGUE) == [20, 30, 31]


def test_the_fastest_draft_breaks_speed_ties_on_the_lowest_id() -> None:
    army = (_tank(22), _tank(20), _tank(21))
    assert draft_fastest(army, _ANCHOR, 2, _CATALOGUE) == [20, 21]


def test_an_unpriced_type_is_drafted_last_not_never() -> None:
    """The catalogue cannot price it, so it is the slowest thing gathered."""
    army = (entity(20, "mystery"), _tank(21), _hover(22))
    assert draft_fastest(army, _ANCHOR, 2, _CATALOGUE) == [21, 22]
    assert draft_fastest(army, _ANCHOR, 3, _CATALOGUE) == [20, 21, 22]


def test_the_fastest_draft_is_whole_from_the_gathered_or_nothing() -> None:
    army = (_hover(20), _hover(21, x=5000.0))
    assert draft_fastest(army, _ANCHOR, 2, _CATALOGUE) == []


def test_the_road_home_is_an_attack_move_to_the_anchor() -> None:
    orders = homeward([20, 21], _ANCHOR)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [
        (20, 0.0, 0.0),
        (21, 0.0, 0.0),
    ]


def test_a_detachment_opens_empty_at_the_engines_first_group() -> None:
    """Below the engine's first-group size the AI calls a force a trickle."""
    detachment = Detachment()
    assert detachment.size == 3
    assert detachment.party() == frozenset()
    assert detachment.objectives == 0
    assert detachment.marches == 0


def test_an_objective_is_ordered_once_per_member_and_counted_once() -> None:
    detachment = Detachment(size=2)
    assert detachment.muster([20, 21])
    first = detachment.advance(9, 400.0, 0.0)
    assert [(o["unit_id"], o["x"]) for o in first] == [(20, 400.0), (21, 400.0)]
    assert detachment.advance(9, 400.0, 0.0) == ()
    assert detachment.objectives == 1
    assert detachment.marches == 2


def test_a_new_objective_re_sends_every_member() -> None:
    detachment = Detachment(size=2)
    detachment.muster([20, 21])
    detachment.advance(9, 400.0, 0.0)
    again = detachment.advance(7, 90.0, 0.0)
    assert [(o["unit_id"], o["x"]) for o in again] == [(20, 90.0), (21, 90.0)]
    assert detachment.objectives == 2
    assert detachment.marches == 4


def test_a_forgotten_objective_counts_afresh_when_retaken() -> None:
    """The raid's confirmation path: the memory is corrected, the objective
    dropped, and whatever is chosen next -- even the same id -- is new."""
    detachment = Detachment(size=1)
    detachment.muster([20])
    detachment.advance(9, 400.0, 0.0)
    detachment.forget_objective()
    assert len(detachment.advance(9, 400.0, 0.0)) == 1
    assert detachment.objectives == 2


def test_survivors_are_the_living_members_in_id_order() -> None:
    detachment = Detachment(size=3)
    detachment.muster([22, 20, 21])
    assert detachment.survivors((_tank(21), _tank(22), _tank(30))) == [21, 22]
    assert detachment.survivors(()) == []


def test_disbanding_sends_the_survivors_home_and_forgets_everything() -> None:
    detachment = Detachment(size=2)
    detachment.muster([20, 21])
    detachment.advance(9, 400.0, 0.0)
    orders = detachment.disband([21], _ANCHOR)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [(21, 0.0, 0.0)]
    assert detachment.party() == frozenset()
    # A fresh muster on the old objective is a new objective: the sent
    # record went with the party.
    detachment.muster([21])
    assert len(detachment.advance(9, 400.0, 0.0)) == 1
    assert detachment.objectives == 2


def test_mustering_nothing_reports_no_party() -> None:
    detachment = Detachment(size=2)
    assert not detachment.muster([])
    assert detachment.party() == frozenset()
