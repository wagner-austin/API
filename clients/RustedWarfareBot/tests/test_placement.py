"""Decoding the placement rules the agent reads out of the live type registry.

The headline case runs against the real dump archived under
``wiki/sources/m11-pools/``, so the contract is tested against bytes the agent
actually wrote rather than a fixture written to match the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.mechanics.placement import PlacementError, decode_placements, decode_reaches
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
    with pytest.raises(PlacementError) as caught:
        decode_placements(['{"kind":"entity","index":0,"name":"x","needs_pool":false}'])
    assert caught.value.code == "RW-PLACEMENT-001"


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


def test_reaches_are_decoded_for_armed_and_unarmed_alike() -> None:
    """Zero is an answer, not an absence.

    The whole reason this dump exists is that the stat catalogue omits 83 of the
    173 registered types, so a missing name and an unarmed unit became
    indistinguishable ([[policy-threat]]). Carrying the unarmed at zero is what
    lets the reader index instead of defaulting.
    """
    reaches = decode_reaches(
        [
            '{"kind":"unitcombat","index":0,"name":"turret","attack_range":165.0}',
            '{"kind":"unitcombat","index":1,"name":"builder","attack_range":0.0}',
        ]
    )
    assert reaches == {"turret": 165.0, "builder": 0.0}


def test_the_reach_decoder_skips_the_kinds_it_does_not_own() -> None:
    """One file, three kinds, three decoders that each project their own."""
    lines = [
        '{"kind":"unittype","index":0,"name":"extractorT1","needs_pool":true}',
        '{"kind":"buildedge","index":0,"producer":"builder","produces":"landFactory"}',
        '{"kind":"unitcombat","index":0,"name":"turret","attack_range":165.0}',
    ]
    assert decode_reaches(lines) == {"turret": 165.0}


def test_an_unknown_kind_is_rejected_by_the_reach_decoder_too() -> None:
    with pytest.raises(PlacementError) as caught:
        decode_reaches(['{"kind":"nonsense","index":0,"name":"x","attack_range":1.0}'])
    assert caught.value.code == "RW-PLACEMENT-001"


def test_a_repeated_type_name_is_rejected_by_the_reach_decoder_too() -> None:
    """It is the join key to live entities, so it must identify one type."""
    line = '{"kind":"unitcombat","index":0,"name":"turret","attack_range":165.0}'
    with pytest.raises(PlacementError) as caught:
        decode_reaches([line, line])
    assert caught.value.code == "RW-PLACEMENT-002"


def test_the_reach_decoder_skips_blank_lines() -> None:
    """The dump is appended to, so a trailing newline is the normal case."""
    line = '{"kind":"unitcombat","index":0,"name":"turret","attack_range":165.0}'
    assert decode_reaches(["", line, "   ", ""]) == {"turret": 165.0}
