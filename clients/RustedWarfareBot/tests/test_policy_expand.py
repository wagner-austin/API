"""Turning goals into an executable plan.

Expansion is pure, so it is tested exhaustively without a game. The headline
cases run against the real archived build tree, because the property that
matters -- that the plan the bot will actually execute is reachable -- is a
property of the engine's tree rather than of a fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.mechanics.build_tree import decode_build_tree
from rw_bot.mechanics.catalogue import UnitStats, decode_catalogue
from rw_bot.policy.expand import ExpansionError, expand

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DUMP = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"

#: What a match starts with, and therefore what expansion may assume.
_OPENING = ("commandCenter", "builder")


def _tree() -> dict[str, frozenset[str]]:
    return decode_build_tree(_DUMP.read_text(encoding="utf-8").splitlines())


def _catalogue() -> dict[str, UnitStats]:
    lines = _CATALOGUE_PATH.read_text(encoding="utf-8", errors="strict").splitlines()
    return {unit["type_name"]: unit for unit in decode_catalogue(lines)}


def _unit(type_name: str, price: int) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=0.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


def test_a_tank_gains_the_factory_that_makes_it() -> None:
    """The whole point: the prerequisite is derived, not written out by hand."""
    assert expand(("c_tank",), _tree(), _OPENING, _catalogue()) == ("landFactory", "c_tank")


def test_two_tanks_share_one_factory() -> None:
    """Availability accumulates, which is why expansion runs over the list.

    Expanding each goal in isolation would insert a factory twice and spend the
    credits for a second one the bot does not need.
    """
    assert expand(("c_tank", "c_tank"), _tree(), _OPENING, _catalogue()) == (
        "landFactory",
        "c_tank",
        "c_tank",
    )


def test_two_different_units_from_one_factory_share_it() -> None:
    assert expand(("hoverTank", "c_artillery"), _tree(), _OPENING, _catalogue()) == (
        "landFactory",
        "hoverTank",
        "c_artillery",
    )


def test_goal_order_survives_expansion() -> None:
    """Prerequisites go in front of what needs them, not in front of everything.

    The economy has to open the plan, so an extractor asked for first must stay
    first even though a later goal drags a factory in.
    """
    assert expand(("extractorT1", "c_tank"), _tree(), _OPENING, _catalogue()) == (
        "extractorT1",
        "landFactory",
        "c_tank",
    )


def test_something_already_buildable_gains_nothing() -> None:
    assert expand(("extractorT1",), _tree(), _OPENING, _catalogue()) == ("extractorT1",)


def test_a_goal_nothing_makes_is_refused_before_the_game_starts() -> None:
    """The laboratory, caught at expansion rather than after 300 samples."""
    with pytest.raises(ExpansionError) as caught:
        expand(("laboratory",), _tree(), _OPENING, _catalogue())
    assert caught.value.code == "RW-EXPAND-001"
    assert "laboratory" in caught.value.message


def test_a_goal_unreachable_from_an_empty_roster_names_its_producers() -> None:
    """Owning nothing makes everything unreachable, and the message says why.

    The build tree has cycles -- a factory makes a builder and a builder makes a
    factory -- so a search that did not track what it was already resolving
    would not terminate here.
    """
    with pytest.raises(ExpansionError) as caught:
        expand(("c_tank",), _tree(), (), _catalogue())
    assert caught.value.code == "RW-EXPAND-001"
    assert "landFactory" in caught.value.message


def test_the_cheaper_producer_is_chosen() -> None:
    tree = {
        "cheap": frozenset({"goal"}),
        "dear": frozenset({"goal"}),
        "have": frozenset({"cheap", "dear"}),
    }
    catalogue = {"cheap": _unit("cheap", 100), "dear": _unit("dear", 900)}
    assert expand(("goal",), tree, ("have",), catalogue) == ("cheap", "goal")


def test_an_unpriced_producer_loses_to_a_priced_one() -> None:
    """The catalogue and the tree are separate dumps with separate coverage."""
    tree = {
        "priced": frozenset({"goal"}),
        "unpriced": frozenset({"goal"}),
        "have": frozenset({"priced", "unpriced"}),
    }
    catalogue = {"priced": _unit("priced", 5000)}
    assert expand(("goal",), tree, ("have",), catalogue) == ("priced", "goal")


def test_a_dead_end_producer_is_abandoned_for_a_reachable_one() -> None:
    """The cheapest producer is not always reachable, so the search backtracks."""
    tree = {
        "cheap": frozenset({"goal"}),
        "dear": frozenset({"goal"}),
        "have": frozenset({"dear"}),
    }
    catalogue = {"cheap": _unit("cheap", 100), "dear": _unit("dear", 900)}
    assert expand(("goal",), tree, ("have",), catalogue) == ("dear", "goal")


def test_a_chain_two_deep_is_built_in_order() -> None:
    tree = {"have": frozenset({"middle"}), "middle": frozenset({"goal"})}
    catalogue = {"middle": _unit("middle", 100)}
    assert expand(("goal",), tree, ("have",), catalogue) == ("middle", "goal")


def test_a_producer_that_makes_itself_does_not_loop() -> None:
    """A self-edge is a cycle of length one and must not be followed."""
    tree = {"goal": frozenset({"goal"})}
    with pytest.raises(ExpansionError) as caught:
        expand(("goal",), tree, (), {})
    assert caught.value.code == "RW-EXPAND-001"


def test_the_expanded_default_plan_is_reachable_from_the_opening_roster() -> None:
    """What the bot will actually run, checked against the real tree.

    A plan that is not reachable costs a whole match to discover, so this is the
    check that has to hold before any live run.
    """
    from scripts.play import DEFAULT_GOALS

    plan = expand(DEFAULT_GOALS, _tree(), _OPENING, _catalogue())
    assert plan == (
        "extractorT1",
        "extractorT1",
        "extractorT1",
        "landFactory",
        "c_tank",
        "c_tank",
    )
