"""The per-sample record, which exists so a run can be read back.

Aggregates could not say whether thirty-seven units died at once or bled away
over six minutes, and those call for opposite fixes. These cover the two
questions the trace answers -- when, and where.
"""

from __future__ import annotations

from rw_bot.policy.trace import Loss, Tick, format_trace, losses_between, owned_by_id
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import entity


def _entity(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
) -> Entity:
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=not mine,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=True,
        queued=0,
    )


def _sample(*entities: Entity) -> Sample:
    return Sample(
        frame=7,
        clock_ms=25,
        credits=4000,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=entities,
        pools=(),
        players=(),
        options=(),
        refusals=(),
    )


def test_only_our_own_units_are_indexed() -> None:
    """An opponent leaving our view is not a loss of ours."""
    world = _sample(_entity(1, "c_tank"), _entity(9, "c_tank", mine=False))
    assert sorted(owned_by_id(world)) == [1]


def test_a_unit_that_left_the_roster_is_reported_with_where_it_stood() -> None:
    """Position comes from the previous sample; the current one cannot say."""
    before = owned_by_id(_sample(_entity(1, "c_tank", 900.0, 250.0), _entity(2, "builder")))
    after = owned_by_id(_sample(_entity(2, "builder")))
    assert losses_between(before, after, 12) == (
        Loss(frame=12, unit_id=1, type_name="c_tank", x=900.0, y=250.0, killer=""),
    )


def test_a_roster_that_only_grew_reports_no_loss() -> None:
    before = owned_by_id(_sample(_entity(1, "c_tank")))
    after = owned_by_id(_sample(_entity(1, "c_tank"), _entity(2, "c_tank")))
    assert losses_between(before, after, 12) == ()


def test_the_first_sample_has_nothing_to_compare_against() -> None:
    assert losses_between({}, owned_by_id(_sample(_entity(1, "c_tank"))), 1) == ()


def test_a_whole_army_lost_at_once_is_reported_unit_by_unit() -> None:
    """Which is the distinction the aggregate could not make."""
    before = owned_by_id(
        _sample(_entity(1, "c_tank", 100.0, 0.0), _entity(2, "c_tank", 110.0, 0.0))
    )
    assert [loss["unit_id"] for loss in losses_between(before, {}, 30)] == [1, 2]


def test_both_tables_are_rendered_with_a_blank_line_between() -> None:
    lines = format_trace(
        (
            Tick(
                frame=7,
                army=3,
                credits=4000,
                enemies=12,
                extractors=5,
                lost=1,
                producers=2,
                idle=1,
                orders=1,
                refused=0,
                worth=3500,
                rival=9000,
                income=54,
                rival_income=180,
                navy_seen=2,
                air_seen=1,
                navy_blood=3,
                events="RT",
                world=123456789,
                plan="building",
                workers=4,
                eco_covered=1,
                own_covered=3,
                foe_covered=2,
            ),
        ),
        (Loss(frame=7, unit_id=1, type_name="c_tank", x=900.0, y=250.0, killer="c_artillery"),),
    )
    assert lines[0].split() == [
        "frame",
        "army",
        "credits",
        "enemies",
        "extractors",
        "lost",
        "producers",
        "idle",
        "orders",
        "refused",
        "worth",
        "rival",
        "income",
        "rival_income",
        "world",
        "plan",
        "workers",
        "navy_seen",
        "air_seen",
        "navy_blood",
        "events",
        "eco_covered",
        "own_covered",
        "foe_covered",
    ]
    assert lines[1].split() == [
        "7",
        "3",
        "4000",
        "12",
        "5",
        "1",
        "2",
        "1",
        "1",
        "0",
        "3500",
        "9000",
        "54",
        "180",
        "123456789",
        "building",
        "4",
        "2",
        "1",
        "3",
        "RT",
        "1",
        "3",
        "2",
    ]
    assert lines[2] == ""
    assert lines[3].split() == ["frame", "unit", "type", "x", "y", "killer"]
    assert lines[4].split() == ["7", "1", "c_tank", "900", "250", "c_artillery"]


def test_a_run_that_lost_nothing_still_renders_both_headers() -> None:
    """An empty loss table is a finding; a missing one is a gap in the record."""
    lines = format_trace(
        (
            Tick(
                frame=1,
                army=0,
                credits=0,
                enemies=0,
                extractors=0,
                lost=0,
                producers=0,
                idle=0,
                orders=0,
                refused=0,
                worth=0,
                rival=0,
                income=0,
                rival_income=0,
                navy_seen=0,
                eco_covered=0,
                own_covered=0,
                foe_covered=0,
                air_seen=0,
                navy_blood=0,
                events="-",
                world=0,
                plan="done",
                workers=0,
            ),
        ),
        (),
    )
    assert lines[-1].split() == ["frame", "unit", "type", "x", "y", "killer"]


def test_the_income_pair_sits_before_the_world_digest() -> None:
    """Existing readers index extractors 4, lost 5, worth 10 and rival 11 by
    position, so the new pair lands at 12-13 and only the digest moves."""
    header = format_trace((), ())[0].split()
    assert header.index("income") == 12
    assert header.index("rival_income") == 13
    assert header.index("world") == 14
    assert header.index("plan") == 15
    assert header.index("workers") == 16
    # The enemy-shape trio appends past every positional reader's reach.
    assert header.index("navy_seen") == 17
    assert header.index("air_seen") == 18
    assert header.index("navy_blood") == 19
    assert header.index("events") == 20
