"""The medic channel: one saving hire at a time until the headcount holds.

The tier-two factory offers the 3,500-credit combat engineer and pressure
never lets 3,500 fit a tick -- measured, `produce:combatEngineer asked 181
got 0` -- so a refused hire withholds its price exactly as the tech unlock
does, and the bound is the doctrine's headcount ([[policy-budget]]).
"""

from __future__ import annotations

from rw_bot.policy.budget import Budget
from rw_bot.policy.medic import Medic
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, option, sample


def _world(*, medics_alive: int, factory_queued: int = 0, offer: bool = True) -> Sample:
    entities = [
        entity(213, "commandCenter"),
        entity(500, "landFactory", queued=factory_queued),
    ]
    for index in range(medics_alive):
        entities.append(entity(600 + index, "combatEngineer", x=50.0 + index))
    options = (option(500, "combatEngineer", key="u_combatEngineer", price=3500),) if offer else ()
    return sample(*entities, credits=4000, options=options)


def test_a_funded_hire_goes_to_the_idle_offering_factory() -> None:
    medic = Medic()
    budget = Budget(10_000, reserve=0)
    orders = medic.hire(_world(medics_alive=0), budget, 2)
    assert [o["unit_id"] for o in orders] == [500]
    assert [o["type_name"] for o in orders] == ["combatEngineer"]
    assert budget.spent() == 3500


def test_a_refused_hire_withholds_its_price() -> None:
    """The saving pattern: later spenders see that much less this tick."""
    medic = Medic()
    budget = Budget(2_000, reserve=0)
    assert medic.hire(_world(medics_alive=0), budget, 1) == ()
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is False


def test_one_hire_outstanding_at_a_time() -> None:
    """A healer mid-production is a healer already paid for."""
    medic = Medic()
    assert len(medic.hire(_world(medics_alive=0), Budget(10_000, reserve=0), 2)) == 1
    # The factory is now mid-queue: the same shortfall must not hire again.
    busy = _world(medics_alive=0, factory_queued=1)
    assert medic.hire(busy, Budget(10_000, reserve=0), 2) == ()
    # Rolled out: the slot clears and the next hire goes through.
    grown = _world(medics_alive=1)
    assert len(medic.hire(grown, Budget(10_000, reserve=0), 2)) == 1


def test_a_full_headcount_hires_nobody() -> None:
    medic = Medic()
    assert medic.hire(_world(medics_alive=2), Budget(10_000, reserve=0), 2) == ()
    assert medic.hire(_world(medics_alive=0), Budget(10_000, reserve=0), 0) == ()


def test_no_offer_means_no_hire_and_no_saving() -> None:
    """Before the tier opens there is nothing to save toward."""
    medic = Medic()
    budget = Budget(2_000, reserve=0)
    assert medic.hire(_world(medics_alive=0, offer=False), budget, 2) == ()
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True


def test_the_channel_hires_whatever_type_it_was_opened_for() -> None:
    """The bunker arm is the medic machinery with one word changed.

    Measured need: ``produce:mechBunker asked 1178 got 0`` at Impossible --
    ordinary production never accumulates 4,500, so the mobile turret is
    funded through the same saving hire as the combat engineer.
    """
    world = sample(
        entity(213, "commandCenter"),
        entity(700, "mechFactory"),
        credits=5000,
        options=(option(700, "mechBunker", key="u_mechBunker", price=4500),),
    )
    bunker = Medic("mechBunker")
    budget = Budget(10_000, reserve=0)
    orders = bunker.hire(world, budget, 1)
    assert [o["type_name"] for o in orders] == ["mechBunker"]
    assert budget.spent() == 4500
    # A refused hire withholds the bunker's own price, not the medic's.
    short = Budget(3_000, reserve=0)
    assert Medic("mechBunker").hire(world, short, 1) == ()
    assert short.claim("produce:c_tank", 350, protected=True)["granted"] is False


def test_hostile_roster_and_foreign_options_are_ignored() -> None:
    """The count is ours; the offer is the medic's, not the first option."""
    world = sample(
        entity(213, "commandCenter"),
        entity(500, "landFactory"),
        entity(9, "combatEngineer", mine=False, hostile=True, x=800.0),
        entity(10, "c_tank", complete=False, x=90.0),
        credits=4000,
        options=(
            option(500, "c_tank", key="u_c_tank", price=350),
            option(500, "combatEngineer", key="u_combatEngineer", index=1, price=3500),
        ),
    )
    orders = Medic().hire(world, Budget(10_000, reserve=0), 1)
    assert [o["type_name"] for o in orders] == ["combatEngineer"]
