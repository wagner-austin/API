"""Decoding the placement rules the agent reads out of the live type registry.

The headline case runs against the real dump archived under
``wiki/sources/m11-pools/``, so the contract is tested against bytes the agent
actually wrote rather than a fixture written to match the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.mechanics.placement import PlacementError, decode_placements, encode_placement
from rw_bot.mechanics.registry_dump import RegistryDumpError
from rw_bot.validation import DecodeError
from rw_bot.wire.ndjson import NdjsonError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DUMP = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"

_RECORD = '{"kind":"unittype","index":0,"name":"landFactory","needs_pool":false}'


def _dump_lines() -> list[str]:
    return _DUMP.read_text(encoding="utf-8").splitlines()


def test_the_real_dump_covers_every_registered_type() -> None:
    placements = decode_placements(_dump_lines())
    assert len(placements) == 173
    assert len({p["type_name"] for p in placements}) == 173


def test_the_real_dump_names_exactly_the_pool_bound_types() -> None:
    """Eight, and the extractor family is what they are.

    Pinned by name rather than by count alone. The count is what a mod would
    change; the names are what the placement rule means, and a build that
    stopped reporting the extractors as pool-bound would be silently broken
    rather than merely different.
    """
    placements = decode_placements(_dump_lines())
    assert sorted(p["type_name"] for p in placements if p["needs_pool"]) == [
        "bugExtractor",
        "bugExtractorT2",
        "extractor",
        "extractorT1",
        "extractorT2",
        "extractorT3",
        "extractorT3_overclocked",
        "extractorT3_reinforced",
    ]


def test_the_structures_the_bot_builds_are_present_and_correctly_ruled() -> None:
    by_name = {p["type_name"]: p for p in decode_placements(_dump_lines())}
    assert by_name["extractorT1"]["needs_pool"] is True
    assert by_name["landFactory"]["needs_pool"] is False
    assert by_name["commandCenter"]["needs_pool"] is False
    assert by_name["builder"]["needs_pool"] is False


def test_an_empty_dump_yields_nothing() -> None:
    assert decode_placements([]) == ()


def test_blank_lines_are_skipped() -> None:
    assert len(decode_placements(["", _RECORD, "   "])) == 1


def test_a_record_decodes_every_field() -> None:
    assert decode_placements([_RECORD])[0] == {
        "index": 0,
        "type_name": "landFactory",
        "needs_pool": False,
    }


def test_an_unknown_record_kind_is_rejected() -> None:
    with pytest.raises(RegistryDumpError) as caught:
        decode_placements(['{"kind":"entity","index":0,"name":"x","needs_pool":false}'])
    assert caught.value.code == "RW-REGISTRY-001"


def test_a_repeated_type_name_is_rejected() -> None:
    """The agent resolves shadowed names before writing, so a repeat is a bug."""
    with pytest.raises(PlacementError) as caught:
        decode_placements([_RECORD, _RECORD])
    assert caught.value.code == "RW-PLACEMENT-002"
    assert "concatenated" in caught.value.message


def test_a_missing_field_propagates_as_a_decode_error() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_placements(['{"kind":"unittype","index":0,"name":"x"}'])
    assert caught.value.code == "RW-DECODE-001"


def test_a_non_boolean_flag_is_rejected_rather_than_coerced() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_placements(['{"kind":"unittype","index":0,"name":"x","needs_pool":1}'])
    assert caught.value.code == "RW-DECODE-002"


def test_a_blank_type_name_is_rejected() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_placements(['{"kind":"unittype","index":0,"name":"","needs_pool":false}'])
    assert caught.value.code == "RW-DECODE-003"


def test_a_malformed_line_propagates_as_an_ndjson_error() -> None:
    with pytest.raises(NdjsonError) as caught:
        decode_placements(["{oops}"])
    assert caught.value.code == "RW-NDJSON-003"


def test_a_placement_rule_round_trips_through_its_record() -> None:
    """The encoder exists so a decoded dump can be re-emitted as a fixture."""
    decoded = decode_placements([_RECORD])
    assert decode_placements([encode_placement(decoded[0])]) == decoded
