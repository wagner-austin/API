"""The shared party discipline, exercised on its own.

Both holders (raid and hunt) are tested through their own behaviours;
what is pinned here is the contract they share: a draft comes whole from
the gathered or not at all, in id order so two runs of one seed draft
identically, and the road home is an attack-move because it crosses the
same ground the road out did ([[policy-raid]]).
"""

from __future__ import annotations

from rw_bot.policy.party import draft_gathered, homeward
from rw_bot.wire.state import Entity
from tests.wire_fixtures import entity


def _tank(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


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


def test_the_road_home_is_an_attack_move_to_the_anchor() -> None:
    orders = homeward([20, 21], _ANCHOR)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [
        (20, 0.0, 0.0),
        (21, 0.0, 0.0),
    ]
