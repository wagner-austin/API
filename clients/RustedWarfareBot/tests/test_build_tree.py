"""Decoding the engine's build tree out of the archived type dump.

The headline cases run against the real dump under ``wiki/sources/``, so what
is asserted is the engine's own answer rather than a fixture written to match
the decoder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.mechanics.build_tree import (
    BuildTreeError,
    decode_build_tree,
    producers_of,
)
from rw_bot.validation import DecodeError
from rw_bot.wire.ndjson import NdjsonError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DUMP = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"

_EDGE = '{"kind":"buildedge","index":0,"producer":"builder","produces":"landFactory"}'
_PLACEMENT = '{"kind":"unittype","index":0,"name":"extractorT1","needs_pool":true}'


def _dump_lines() -> list[str]:
    return _DUMP.read_text(encoding="utf-8").splitlines()


def test_the_real_dump_gives_the_builder_its_thirteen_structures() -> None:
    """Cross-checks the registry against the live per-entity option stream.

    The world stream reaches the same question by a completely different route
    -- it asks each owned unit -- and reports exactly these thirteen for the
    Builder. Two unrelated paths to one answer is what makes the binding
    trustworthy ([[policy-loop]]).
    """
    tree = decode_build_tree(_dump_lines())
    assert tree["builder"] == frozenset(
        {
            "extractorT1",
            "c_turret_t1",
            "c_antiAirTurret",
            "landFactory",
            "airFactory",
            "seaFactory",
            "mechFactory",
            "laserDefence",
            "repairbay",
            "fabricatorT1",
            "experimentalLandFactory",
            "nukeLauncherC",
            "antiNukeLauncherC",
        }
    )


def test_the_land_factory_makes_the_units_a_plan_would_ask_for() -> None:
    tree = decode_build_tree(_dump_lines())
    assert tree["landFactory"] == frozenset(
        {"builder", "scout", "c_tank", "hoverTank", "c_artillery"}
    )


def test_nothing_makes_a_laboratory() -> None:
    """The whole of the historical stall, stated as a fact about the tree.

    A plan naming a laboratory ran three hundred samples reporting progress
    while the engine refused it silently. No type in the registry produces one
    at the base tech level, so the plan was never executable.
    """
    tree = decode_build_tree(_dump_lines())
    assert producers_of(tree, "laboratory") == frozenset()


def test_the_tree_and_the_placement_rules_share_one_dump() -> None:
    """Both kinds are in the same file, and each decoder reads only its own."""
    lines = _dump_lines()
    assert any('"kind":"unittype"' in line for line in lines)
    assert any('"kind":"buildedge"' in line for line in lines)
    assert decode_build_tree(lines)


def test_placement_records_are_skipped_rather_than_rejected() -> None:
    assert decode_build_tree([_PLACEMENT, _EDGE]) == {"builder": frozenset({"landFactory"})}


def test_blank_lines_are_skipped() -> None:
    assert decode_build_tree(["", _EDGE, "   "]) == {"builder": frozenset({"landFactory"})}


def test_an_empty_dump_yields_no_edges() -> None:
    assert decode_build_tree([]) == {}


def test_two_products_from_one_producer_merge() -> None:
    second = '{"kind":"buildedge","index":1,"producer":"builder","produces":"airFactory"}'
    assert decode_build_tree([_EDGE, second]) == {
        "builder": frozenset({"landFactory", "airFactory"})
    }


def test_producers_of_finds_every_maker() -> None:
    other = '{"kind":"buildedge","index":1,"producer":"fabricatorT1","produces":"landFactory"}'
    tree = decode_build_tree([_EDGE, other])
    assert producers_of(tree, "landFactory") == frozenset({"builder", "fabricatorT1"})


def test_an_unknown_kind_is_rejected() -> None:
    """A record neither decoder claims must not pass silently through both."""
    with pytest.raises(BuildTreeError) as caught:
        decode_build_tree(['{"kind":"weather","index":0}'])
    assert caught.value.code == "RW-BUILDTREE-001"


def test_a_missing_field_propagates_as_a_decode_error() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_build_tree(['{"kind":"buildedge","index":0,"producer":"builder"}'])
    assert caught.value.code == "RW-DECODE-001"


def test_a_malformed_line_propagates_as_an_ndjson_error() -> None:
    with pytest.raises(NdjsonError):
        decode_build_tree(["{oops}"])
