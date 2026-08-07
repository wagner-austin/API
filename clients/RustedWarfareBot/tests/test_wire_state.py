"""Folding the agent's record stream into typed samples.

The headline case runs against the real capture archived under
``wiki/sources/m6-wire/``, so the contract is tested against bytes the agent
actually wrote rather than against a fixture written to match the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.validation import DecodeError
from rw_bot.wire.codec import WireError, decode_samples, encode_sample
from rw_bot.wire.ndjson import NdjsonError
from rw_bot.wire.state import BuildOption, ResourcePool, Sample
from tests.wire_fixtures import entity

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CAPTURE = _PROJECT_ROOT / "wiki" / "sources" / "m6-wire" / "world-sample.ndjson"

_FRAME = (
    '{"kind":"frame","frame":7,"clock_ms":25,"visible":1,"pools":0,"options":0,"players":0,'
    '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
)
_ENTITY = (
    '{"kind":"entity","frame":7,"index":0,"id":214,"type":"builder",'
    '"class":"units.e.b","x":1.5,"y":-2.5,"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,"flying":false,"submerged":false,"touching_water":false,'
    '"hp":170.0,"max_hp":170.0,"complete":true,"queued":0,"damaged_by":""}'
)
_POOL = (
    '{"kind":"pool","frame":7,"index":0,"tile_x":115,"tile_y":6,'
    '"x":2310.0,"y":130.0,"group_land":1}'
)


def _capture_lines() -> list[str]:
    return _CAPTURE.read_text(encoding="utf-8").splitlines()


def test_decodes_the_real_capture_into_three_samples() -> None:
    samples = decode_samples(_capture_lines())
    assert len(samples) == 3
    frames = [s["frame"] for s in samples]
    clocks = [s["clock_ms"] for s in samples]
    # Exact counter values are deliberately not pinned. The capture is
    # regenerated whenever the contract changes, and two sessions regenerating
    # it have already raced a hardcoded expectation. Monotonicity and
    # distinctness are what must hold of any real capture.
    assert frames == sorted(frames)
    assert len(set(frames)) == 3
    assert clocks == sorted(clocks)
    assert len(set(clocks)) == 3


def test_the_real_capture_carries_the_documented_roster() -> None:
    """Command Center, Builder, and the placeholder parked off-map -- ours only."""
    first = decode_samples(_capture_lines())[0]
    mine = [e for e in first["entities"] if e["mine"]]
    assert [e["type_name"] for e in mine] == [
        "commandCenter",
        "builder",
        "editorOrBuilder",
    ]
    assert [e["class_name"] for e in mine] == [
        "com.corrodinggames.rts.game.units.d.e",
        "com.corrodinggames.rts.game.units.e.b",
        "com.corrodinggames.rts.game.units.h",
    ]
    # The third entity is the map editor's placeholder, parked off-map. Its
    # type name is what identified it; the class alone never did.
    assert mine[2]["x"] == -1000.0
    assert mine[2]["y"] == -1000.0


def test_engine_ids_are_distinct_and_stable_across_samples() -> None:
    """id is the dispatch handle; index renumbers, id does not.

    Stability is asserted for the units alive throughout, not for the
    roster: the capture is a live world, and a factory finishing a unit
    between samples grows the list without renaming anyone in it.
    """
    samples = decode_samples(_capture_lines())
    ids = [e["unit_id"] for e in samples[0]["entities"]]
    assert len(set(ids)) == len(ids)
    for later in samples[1:]:
        later_ids = [e["unit_id"] for e in later["entities"]]
        assert len(set(later_ids)) == len(later_ids)
        surviving = [unit_id for unit_id in later_ids if unit_id in set(ids)]
        assert surviving == ids


def test_the_real_capture_advances_at_the_measured_frame_rate() -> None:
    """~300 frames per second, cross-checking the wire against the clock."""
    samples = decode_samples(_capture_lines())
    frames = samples[2]["frame"] - samples[1]["frame"]
    millis = samples[2]["clock_ms"] - samples[1]["clock_ms"]
    assert 290.0 < frames / (millis / 1000.0) < 310.0


def test_engine_ids_are_the_dispatch_handles() -> None:
    """Index is enumeration order; id is what an order addresses."""
    first = decode_samples(_capture_lines())[0]
    mine = [e for e in first["entities"] if e["mine"]]
    assert [e["index"] for e in mine] == [6, 7, 10]
    assert [e["unit_id"] for e in mine] == [213, 214, 217]


def test_the_capture_carries_other_players_too() -> None:
    """Three entities are ours; the rest belong to several opponents.

    The total is deliberately not pinned. It moves between captures as the
    built-in AI builds and loses things, and pinning it made a recapture into a
    test edit -- the same brittleness already removed from the frame counters.
    What must hold of any capture is that ownership discriminates.
    """
    first = decode_samples(_capture_lines())[0]
    mine = [e for e in first["entities"] if e["mine"]]
    theirs = [e for e in first["entities"] if not e["mine"]]
    assert len(mine) == 3
    assert len({e["team"] for e in theirs}) > 1
    assert {e["team"] for e in mine} == {0}


def test_health_is_carried_for_every_visible_entity() -> None:
    first = decode_samples(_capture_lines())[0]
    centres = [e for e in first["entities"] if e["type_name"] == "commandCenter"]
    assert [e["max_hp"] for e in centres] == [4000.0] * len(centres)


def test_an_empty_stream_yields_no_samples() -> None:
    assert decode_samples([]) == ()


def test_blank_lines_are_skipped() -> None:
    assert len(decode_samples(["", _FRAME, "   ", _ENTITY, ""])) == 1


def test_a_sample_with_no_entities_is_valid() -> None:
    empty = (
        '{"kind":"frame","frame":1,"clock_ms":0,"visible":0,"pools":0,"options":0,"players":0,'
        '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
    )
    samples = decode_samples([empty])
    assert samples[0]["entities"] == ()
    assert samples[0]["pools"] == ()


def test_the_real_capture_carries_the_maps_resource_pools() -> None:
    """Forty-six, which the map file independently agrees with.

    The agent reaches them by walking the live tile grid reflectively; the
    count and the coordinates were separately read out of the ``.tmx`` by
    decompressing its Items layer. Two unrelated routes to the same answer is
    what makes the tile binding trustworthy ([[mechanics-resource-pools]]).
    """
    first = decode_samples(_capture_lines())[0]
    assert len(first["pools"]) == 46
    assert first["pools"][0]["tile_x"] == 115
    assert first["pools"][0]["tile_y"] == 6


def test_pool_world_points_are_the_centre_of_their_tile() -> None:
    """A build order is addressed in world space, so the conversion carries."""
    first = decode_samples(_capture_lines())[0]
    for pool in first["pools"]:
        assert pool["x"] == pool["tile_x"] * 20.0 + 10.0
        assert pool["y"] == pool["tile_y"] * 20.0 + 10.0


def test_the_pools_are_the_same_ones_in_every_sample() -> None:
    """Terrain does not move, so a differing set would mean a bad read."""
    samples = decode_samples(_capture_lines())
    tiles = [{(p["tile_x"], p["tile_y"]) for p in s["pools"]} for s in samples]
    assert tiles[0] == tiles[1] == tiles[2]


def test_a_pool_before_any_frame_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        decode_samples([_POOL])
    assert caught.value.code == "RW-WIRE-002"


def test_a_sample_short_of_its_declared_pools_is_rejected() -> None:
    short = (
        '{"kind":"frame","frame":7,"clock_ms":25,"visible":0,"pools":2,"options":0,"players":0,'
        '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
    )
    with pytest.raises(WireError) as caught:
        decode_samples([short, _POOL])
    assert caught.value.code == "RW-WIRE-005"
    assert "truncated" in caught.value.message


def test_a_sample_short_of_its_declared_options_is_rejected() -> None:
    """The same completeness rule the entity and pool counts get.

    A half-read option list is the worst of the three to act on: it does not
    look wrong, it looks like a unit that cannot make the thing the plan wants,
    and the planner would answer that by declaring the plan dead.
    """
    short = (
        '{"kind":"frame","frame":7,"clock_ms":25,"visible":0,"pools":0,"options":2,"players":0,'
        '"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
    )
    option = (
        '{"kind":"option","frame":7,"index":0,"unit_id":214,'
        '"produces":"landFactory","key":"u_landFactory","placed":true,"available":true,'
        '"makes_something":true,"price":100}'
    )
    with pytest.raises(WireError) as caught:
        decode_samples([short, option])
    assert caught.value.code == "RW-WIRE-006"
    assert "truncated" in caught.value.message


def test_an_interleaved_pool_frame_is_rejected() -> None:
    mismatched = '{"kind":"pool","frame":99,"index":0,"tile_x":1,"tile_y":2,"x":30.0,"y":50.0}'
    with pytest.raises(WireError) as caught:
        decode_samples([_FRAME, _ENTITY, mismatched])
    assert caught.value.code == "RW-WIRE-004"


def test_an_entity_before_any_frame_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        decode_samples([_ENTITY])
    assert caught.value.code == "RW-WIRE-002"


def test_a_short_sample_is_rejected_rather_than_silently_truncated() -> None:
    """A planner acting on a roster it cannot fully see is the failure guarded."""
    with pytest.raises(WireError) as caught:
        short = (
            '{"kind":"frame","frame":7,"clock_ms":25,"visible":3,'
            '"pools":0,"options":0,"players":0,"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
        )
        decode_samples([short, _ENTITY])
    assert caught.value.code == "RW-WIRE-003"
    assert "truncated" in caught.value.message


def test_a_long_sample_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        long_frame = (
            '{"kind":"frame","frame":7,"clock_ms":25,"visible":1,'
            '"pools":0,"options":0,"players":0,"credits":4000,"defeated":false,"wiped":false,"players_left":6}'
        )
        decode_samples([long_frame, _ENTITY, _ENTITY])
    assert caught.value.code == "RW-WIRE-003"


def test_an_interleaved_entity_frame_is_rejected() -> None:
    mismatched = '{"kind":"entity","frame":99,"index":0,"class":"u","x":0,"y":0}'
    with pytest.raises(WireError) as caught:
        decode_samples([_FRAME, mismatched])
    assert caught.value.code == "RW-WIRE-004"


def test_an_unknown_record_kind_is_rejected() -> None:
    """Inside a sample, because outside one it is the earlier fault.

    A record before any frame is rejected as a stream that does not begin at a
    sample boundary, which is a different and more specific complaint. Opening a
    frame first is what makes this test about the unknown kind.
    """
    with pytest.raises(WireError) as caught:
        decode_samples(
            [
                '{"kind":"frame","frame":1,"clock_ms":0,"visible":0,"pools":0,'
                '"options":0,"players":0,"credits":0,"defeated":false,'
                '"wiped":false,"players_left":6}',
                '{"kind":"weather","frame":1}',
            ]
        )
    assert caught.value.code == "RW-WIRE-001"


def test_a_record_before_any_frame_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        decode_samples(['{"kind":"weather","frame":1}'])
    assert caught.value.code == "RW-WIRE-002"


def test_a_missing_field_propagates_as_a_decode_error() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_samples(['{"kind":"frame","frame":7,"clock_ms":25,"visible":0}'])
    assert caught.value.code == "RW-DECODE-001"


def test_a_malformed_line_propagates_as_an_ndjson_error() -> None:
    with pytest.raises(NdjsonError) as caught:
        decode_samples(["{oops}"])
    assert caught.value.code == "RW-NDJSON-003"


def test_encode_decode_round_trips_the_real_capture() -> None:
    original = decode_samples(_capture_lines())
    lines: list[str] = []
    for sample in original:
        lines.extend(encode_sample(sample))
    assert decode_samples(lines) == original


def test_encode_escapes_characters_that_would_break_the_line() -> None:
    """Class names never need this, so the round-trip claim is what tests it."""
    hostile = Sample(
        frame=1,
        clock_ms=0,
        credits=4000,
        defeated=False,
        wiped=False,
        players_left=6,
        players=(),
        entities=(
            entity(
                214,
                'a"b\\c\nd\te\rf\x01g',
                class_name='a"b\\c\nd\te\rf\x01g',
                team=3,
                mine=False,
                hostile=True,
                hp=1.5,
                max_hp=2.5,
            ),
        ),
        pools=(ResourcePool(index=0, tile_x=115, tile_y=6, x=2310.0, y=130.0, group_land=1),),
        options=(
            BuildOption(
                index=0,
                unit_id=214,
                produces='a"b\\c\nd\te\rf\x01g',
                key='a"b\\c\nd\te\rf\x01g',
                placed=True,
                available=False,
                makes_something=True,
                price=350,
            ),
        ),
    )
    lines = encode_sample(hostile)
    assert len(lines) == 4
    assert decode_samples(list(lines)) == (hostile,)


def test_the_real_capture_shows_credits_accruing() -> None:
    """The Command Center generates income, so credits rise between samples.

    The property is asserted, not the figures. Exact credit values depend on
    when the capture was taken, and pinning them makes every recapture a test
    edit -- the same brittleness already removed from the frame counters.
    """
    credits = [s["credits"] for s in decode_samples(_capture_lines())]
    assert len(credits) >= 2
    assert credits == sorted(credits)
    assert credits[-1] > credits[0]
