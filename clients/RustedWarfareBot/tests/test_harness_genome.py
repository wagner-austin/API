"""The genome compiler: weights in, ordinary doctrine out, deterministic.

Every apportionment is pinned to exact slot lists -- two runs of one
genome must agree entry for entry, because the compiled doctrine is the
frozen record a candidate's matches replay from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.harness.genome import (
    ARMY_VOCABULARY,
    GenomeError,
    army_split,
    compile_genome,
    ratio_slots,
)
from rw_bot.policy.doctrine_file import parse_doctrine_lines

_CHAMPION = parse_doctrine_lines(
    Path("doctrines/close0-flame4.doctrine").read_text(encoding="utf-8").splitlines()
)


def test_the_vocabulary_is_sorted_because_ties_break_by_it() -> None:
    assert tuple(sorted(ARMY_VOCABULARY)) == ARMY_VOCABULARY


def test_army_split_keeps_the_scaffold_in_build_order() -> None:
    scaffold, slots = army_split(_CHAMPION["goals"])
    assert scaffold == ("extractorT1", "extractorT1", "extractorT1")
    assert slots == 5


def test_largest_remainder_apportions_exactly_and_deterministically() -> None:
    # 5 slots at 60/20/20: quotas 3.0/1.0/1.0 -- no remainder to argue over.
    tail = ratio_slots({"c_tank": 0.6, "hoverTank": 0.2, "c_artillery": 0.2}, 5)
    assert tail == ("c_tank", "c_tank", "c_tank", "c_artillery", "hoverTank")
    # 5 slots at equal thirds: quotas 1.667 each, floors 1+1+1, two
    # leftovers -- identical remainders break by vocabulary order.
    tail = ratio_slots({"c_tank": 1.0, "hoverTank": 1.0, "heavyTank": 1.0}, 5)
    assert tail == ("c_tank", "c_tank", "heavyTank", "heavyTank", "hoverTank")


def test_zero_weight_units_get_no_slots() -> None:
    tail = ratio_slots({"c_tank": 1.0, "heavyTank": 0.0}, 3)
    assert tail == ("c_tank", "c_tank", "c_tank")


def test_weights_outside_the_vocabulary_or_range_are_refused() -> None:
    with pytest.raises(GenomeError) as caught:
        ratio_slots({"battleShip": 1.0}, 5)
    assert caught.value.code == "RW-GENOME-001"
    with pytest.raises(GenomeError) as caught:
        ratio_slots({"c_tank": -0.1}, 5)
    assert caught.value.code == "RW-GENOME-001"
    with pytest.raises(GenomeError) as caught:
        ratio_slots({"c_tank": float("nan")}, 5)
    assert caught.value.code == "RW-GENOME-001"
    with pytest.raises(GenomeError) as caught:
        ratio_slots({"c_tank": 0.0}, 5)
    assert caught.value.code == "RW-GENOME-001"
    assert "names no army" in caught.value.message


def test_a_compiled_genome_is_an_ordinary_validated_doctrine() -> None:
    variant = compile_genome(
        _CHAMPION,
        {"c_artillery": 0.4, "heavyTank": 0.4, "c_tank": 0.2},
        {"raid": 5},
        "gen0-m3",
    )
    assert variant["name"] == "gen0-m3"
    assert variant["raid"] == 5
    # The scaffold survives in front; the tail is the apportionment.
    assert variant["goals"] == (
        "extractorT1",
        "extractorT1",
        "extractorT1",
        "c_artillery",
        "c_artillery",
        "heavyTank",
        "heavyTank",
        "c_tank",
    )
    # Untouched fields carry over from the base.
    assert variant["flame"] == _CHAMPION["flame"]
    assert variant["heavies"] == _CHAMPION["heavies"]


def test_a_knob_outside_the_doctrine_is_refused() -> None:
    with pytest.raises(GenomeError) as caught:
        compile_genome(_CHAMPION, {"c_tank": 1.0}, {"riad": 5}, "typo")
    assert caught.value.code == "RW-GENOME-003"


def test_a_base_with_no_army_slots_is_refused() -> None:
    scaffold_only = _CHAMPION.copy()
    scaffold_only["goals"] = ("extractorT1", "extractorT1")
    with pytest.raises(GenomeError) as caught:
        compile_genome(scaffold_only, {"c_tank": 1.0}, {}, "hollow")
    assert caught.value.code == "RW-GENOME-002"
