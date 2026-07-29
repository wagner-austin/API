"""Ordering the conversions that raise income, priced the way the engine prices them.

The headline cases run against the **real** archived unit dump rather than a
fixture, because the property under test is a claim about the game's own
numbers: that a conversion costs what the engine says a conversion costs, and
not what the resulting unit costs to build. A fixture can be given any two
prices and would agree with either reading.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.mechanics.catalogue import UnitStats, decode_catalogue
from rw_bot.policy.budget import Budget
from rw_bot.policy.spending import upgrade_income
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, option, sample

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"

#: What the engine prints for the tier one: ``Price: $700`` and
#: ``T2 Upgrade Price: $1400``. Both figures are in the dump, and the gap
#: between them is the whole of this module's regression.
_T1_BUILD_PRICE = 700
_T1_UPGRADE_PRICE = 1400

#: What the engine prints for the tier two. The build price is what the old
#: code claimed for a conversion; the real conversion is 700 credits cheaper.
_T2_BUILD_PRICE = 2100


def _catalogue() -> dict[str, UnitStats]:
    lines = _CATALOGUE_PATH.read_text(encoding="utf-8", errors="strict").splitlines()
    return {unit["type_name"]: unit for unit in decode_catalogue(lines)}


def _offering(*unit_ids: int, credits_held: int) -> Sample:
    """Build a world where each named extractor offers to convert itself."""
    return sample(
        *(entity(unit_id, "extractorT1") for unit_id in unit_ids),
        credits=credits_held,
        options=tuple(
            option(unit_id, "extractorT2", index=index, placed=False)
            for index, unit_id in enumerate(unit_ids)
        ),
    )


def test_the_dump_states_both_prices_and_they_differ() -> None:
    """The premise of the regression, asserted before anything depends on it.

    If the engine ever priced a conversion at the target's build cost the two
    readings would coincide and every other test here would pass under the old
    code as well. They do not coincide: 1,400 against 2,100.
    """
    catalogue = _catalogue()
    assert catalogue["extractorT1"]["price"] == _T1_BUILD_PRICE
    assert catalogue["extractorT1"]["upgrade_prices"] == (_T1_UPGRADE_PRICE,)
    assert catalogue["extractorT2"]["price"] == _T2_BUILD_PRICE


def test_a_conversion_is_claimed_at_the_holders_upgrade_price() -> None:
    """The bug: 2,100 claimed for a purchase the engine charges 1,400 for."""
    world = _offering(400, credits_held=10_000)
    budget = Budget(10_000, 0)
    orders = upgrade_income(world, _catalogue(), budget, set())
    assert len(orders) == 1
    assert budget.spent() == _T1_UPGRADE_PRICE


def test_an_upgrade_affordable_only_at_the_true_price_is_still_ordered() -> None:
    """The credits between the two readings, which is where the fault bit.

    At 1,400 held, the conversion is exactly affordable and the old code
    refused it -- on every tick the balance sat in this band. A budget that
    already turns down 1,185 to 1,685 claims a match cannot spare a refusal it
    invented ([[policy-holding-ground]]).
    """
    world = _offering(400, credits_held=_T1_UPGRADE_PRICE)
    budget = Budget(_T1_UPGRADE_PRICE, 0)
    orders = upgrade_income(world, _catalogue(), budget, set())
    assert len(orders) == 1
    assert orders[0]["type_name"] == "extractorT2"
    assert orders[0]["unit_id"] == 400
    assert budget.spent() == _T1_UPGRADE_PRICE


def test_one_credit_short_of_the_true_price_is_refused() -> None:
    """The gate still exists; it was only ever set at the wrong number."""
    budget = Budget(_T1_UPGRADE_PRICE - 1, 0)
    orders = upgrade_income(
        _offering(400, credits_held=_T1_UPGRADE_PRICE - 1), _catalogue(), budget, set()
    )
    assert orders == ()
    assert budget.spent() == 0


def test_ordering_stops_at_the_first_refusal() -> None:
    """Every conversion costs the same, so a refusal means the budget is out.

    Three extractors offer and two are affordable. Skipping to a third would
    spend credits the second was refused for.
    """
    budget = Budget(2 * _T1_UPGRADE_PRICE, 0)
    ordered: set[tuple[int, str]] = set()
    orders = upgrade_income(
        _offering(400, 401, 402, credits_held=2 * _T1_UPGRADE_PRICE),
        _catalogue(),
        budget,
        ordered,
    )
    assert tuple(order["unit_id"] for order in orders) == (400, 401)
    assert budget.spent() == 2 * _T1_UPGRADE_PRICE
    assert ordered == {(400, "extractorT2"), (401, "extractorT2")}


def test_a_structure_already_told_to_upgrade_is_not_asked_twice() -> None:
    """A conversion never fills ``queued``, so the offer stands while it runs.

    Re-ordering it sent duplicates, and one arriving after the conversion had
    finished was addressed to a unit that could no longer make what it named --
    which crashed the match rather than degrading it
    ([[policy-holding-ground]]).
    """
    budget = Budget(10_000, 0)
    orders = upgrade_income(
        _offering(400, credits_held=10_000), _catalogue(), budget, {(400, "extractorT2")}
    )
    assert orders == ()
    assert budget.spent() == 0


def test_the_same_structure_is_asked_again_for_the_next_tier() -> None:
    """A conversion keeps the engine identity, so the memory is per tier.

    Keyed by the unit alone, a structure that had been told to become a tier
    two could never be told to become a tier three, and the walk would stop
    where it used to stop ([[policy-holding-ground]]).
    """
    world = sample(
        entity(400, "extractorT2"),
        credits=10_000,
        options=(option(400, "extractorT3", placed=False),),
    )
    budget = Budget(10_000, 0)
    orders = upgrade_income(world, _catalogue(), budget, {(400, "extractorT2")})
    assert tuple(order["type_name"] for order in orders) == ("extractorT3",)
    # Priced from the holder: the tier two's own conversion, not the tier
    # three's 6,100 build price.
    assert budget.spent() == 4000


def test_a_holder_the_catalogue_does_not_price_is_skipped() -> None:
    """An unpriced holder is a gap in the dump, not a free upgrade.

    Ordering it would send a conversion whose cost was never claimed, which is
    how two spenders came to spend the same credit ([[policy-budget]]).
    """
    world = sample(
        entity(400, "someModExtractor"),
        credits=10_000,
        options=(option(400, "extractorT2", placed=False),),
    )
    budget = Budget(10_000, 0)
    assert upgrade_income(world, _catalogue(), budget, set()) == ()
    assert budget.spent() == 0


def test_a_holder_that_declares_no_upgrade_is_skipped() -> None:
    """Priced, but with no conversion of its own -- so there is no cost to claim.

    The Command Center is in the catalogue at 3,000 with an empty
    ``upgrade_prices``: 73 of the 90 types declare no upgrade at all. Taking
    ``price`` as a stand-in is exactly the substitution that caused the fault,
    and here it would claim 3,000 for a conversion the engine does not offer.
    """
    catalogue = _catalogue()
    assert catalogue["commandCenter"]["upgrade_prices"] == ()
    world = sample(
        entity(400, "commandCenter"),
        credits=10_000,
        options=(option(400, "extractorT2", placed=False),),
    )
    budget = Budget(10_000, 0)
    assert upgrade_income(world, catalogue, budget, set()) == ()
    assert budget.spent() == 0


def test_the_reserve_is_not_spent_on_an_upgrade() -> None:
    """A conversion is investment, so it may not take what replaces a loss.

    Unprotected like expansion: it pays back over the rest of the match, and
    the army dying now cannot wait for it ([[policy-budget]]).
    """
    budget = Budget(_T1_UPGRADE_PRICE, _T1_UPGRADE_PRICE)
    orders = upgrade_income(
        _offering(400, credits_held=_T1_UPGRADE_PRICE), _catalogue(), budget, set()
    )
    assert orders == ()
    assert budget.spent() == 0
