"""One declaration of what the type-registry dump carries.

Each decoder used to hold its own list of its neighbours' record kinds, so a
kind added for one reader broke the others — and it did, on a live run, when a
third kind appeared ([[mechanics-build-tree]]). Declaring the set once is what
makes a fourth kind a one-line change.
"""

from __future__ import annotations

import pytest

from rw_bot.mechanics.registry_dump import (
    KIND_BUILD_EDGE,
    KIND_UNIT_COMBAT,
    KIND_UNIT_TYPE,
    KINDS,
    RegistryDumpError,
    records_of_kind,
)
from rw_bot.validation import DecodeError

_UNIT_TYPE = '{"kind":"unittype","index":0,"name":"extractorT1","needs_pool":true}'
_BUILD_EDGE = '{"kind":"buildedge","index":0,"producer":"builder","produces":"landFactory"}'
_UNIT_COMBAT = (
    '{"kind":"unitcombat","index":0,"name":"turret","attack_range":165.0,'
    '"hits_land":true,"hits_air":false,"hits_underwater":false,'
    '"hits_land_out_of_water":true}'
)
_ALL = [_UNIT_TYPE, _BUILD_EDGE, _UNIT_COMBAT]


def test_every_kind_the_dump_carries_is_declared_once() -> None:
    assert frozenset({KIND_UNIT_TYPE, KIND_BUILD_EDGE, KIND_UNIT_COMBAT}) == KINDS


def test_each_kind_projects_only_its_own_records() -> None:
    assert [r["kind"] for r in records_of_kind(_ALL, KIND_UNIT_TYPE)] == ["unittype"]
    assert [r["kind"] for r in records_of_kind(_ALL, KIND_BUILD_EDGE)] == ["buildedge"]
    assert [r["kind"] for r in records_of_kind(_ALL, KIND_UNIT_COMBAT)] == ["unitcombat"]


def test_records_come_back_in_dump_order() -> None:
    doubled = [_UNIT_TYPE, _BUILD_EDGE, _UNIT_TYPE.replace('"index":0', '"index":1')]
    assert [r["index"] for r in records_of_kind(doubled, KIND_UNIT_TYPE)] == [0, 1]


def test_a_kind_nobody_claims_is_rejected() -> None:
    """The property worth having: it cannot pass silently through all three readers."""
    with pytest.raises(RegistryDumpError) as caught:
        records_of_kind(['{"kind":"weather","index":0}'], KIND_UNIT_TYPE)
    assert caught.value.code == "RW-REGISTRY-001"
    assert "weather" in caught.value.message


def test_asking_for_a_kind_the_dump_does_not_define_is_a_programming_error() -> None:
    with pytest.raises(RegistryDumpError) as caught:
        records_of_kind(_ALL, "weather")
    assert caught.value.code == "RW-REGISTRY-002"


def test_blank_lines_are_skipped() -> None:
    """The dump is appended to, so a trailing newline is the normal case."""
    assert len(records_of_kind(["", _UNIT_TYPE, "   ", ""], KIND_UNIT_TYPE)) == 1


def test_a_record_with_no_kind_is_a_decode_error() -> None:
    with pytest.raises(DecodeError):
        records_of_kind(['{"index":0}'], KIND_UNIT_TYPE)


def test_an_empty_dump_yields_nothing() -> None:
    assert records_of_kind([], KIND_UNIT_TYPE) == ()
