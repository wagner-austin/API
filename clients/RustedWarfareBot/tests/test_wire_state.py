"""Folding the agent's record stream into typed samples.

The headline case runs against the real capture archived under
``wiki/sources/m6-wire/``, so the contract is tested against bytes the agent
actually wrote rather than against a fixture written to match the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.validation import DecodeError
from rw_bot.wire.ndjson import NdjsonError
from rw_bot.wire.state import (
    Entity,
    Sample,
    WireError,
    decode_samples,
    encode_sample,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CAPTURE = _PROJECT_ROOT / "wiki" / "sources" / "m6-wire" / "world-sample.ndjson"

_FRAME = '{"kind":"frame","frame":7,"clock_ms":25,"owned":1}'
_ENTITY = '{"kind":"entity","frame":7,"index":0,"class":"units.e.b","x":1.5,"y":-2.5}'


def _capture_lines() -> list[str]:
    return _CAPTURE.read_text(encoding="utf-8").splitlines()


def test_decodes_the_real_capture_into_three_samples() -> None:
    samples = decode_samples(_capture_lines())
    assert len(samples) == 3
    assert [s["frame"] for s in samples] == [1, 1597, 3397]
    assert [s["clock_ms"] for s in samples] == [50, 5388, 11388]


def test_the_real_capture_carries_the_documented_roster() -> None:
    """Command Center, Builder, and the entity parked off-map."""
    first = decode_samples(_capture_lines())[0]
    assert [e["class_name"] for e in first["entities"]] == [
        "com.corrodinggames.rts.game.units.d.e",
        "com.corrodinggames.rts.game.units.e.b",
        "com.corrodinggames.rts.game.units.h",
    ]
    assert first["entities"][2]["x"] == -1000.0
    assert first["entities"][2]["y"] == -1000.0


def test_the_real_capture_advances_at_the_measured_frame_rate() -> None:
    """~300 frames per second, cross-checking the wire against the clock."""
    samples = decode_samples(_capture_lines())
    frames = samples[2]["frame"] - samples[1]["frame"]
    millis = samples[2]["clock_ms"] - samples[1]["clock_ms"]
    assert 290.0 < frames / (millis / 1000.0) < 310.0


def test_indices_are_the_dispatch_handles() -> None:
    first = decode_samples(_capture_lines())[0]
    assert [e["index"] for e in first["entities"]] == [0, 1, 2]


def test_an_empty_stream_yields_no_samples() -> None:
    assert decode_samples([]) == ()


def test_blank_lines_are_skipped() -> None:
    assert len(decode_samples(["", _FRAME, "   ", _ENTITY, ""])) == 1


def test_a_sample_with_no_entities_is_valid() -> None:
    empty = '{"kind":"frame","frame":1,"clock_ms":0,"owned":0}'
    samples = decode_samples([empty])
    assert samples[0]["entities"] == ()


def test_an_entity_before_any_frame_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        decode_samples([_ENTITY])
    assert caught.value.code == "RW-WIRE-002"


def test_a_short_sample_is_rejected_rather_than_silently_truncated() -> None:
    """A planner acting on a roster it cannot fully see is the failure guarded."""
    with pytest.raises(WireError) as caught:
        decode_samples(['{"kind":"frame","frame":7,"clock_ms":25,"owned":3}', _ENTITY])
    assert caught.value.code == "RW-WIRE-003"
    assert "truncated" in caught.value.message


def test_a_long_sample_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        decode_samples(['{"kind":"frame","frame":7,"clock_ms":25,"owned":1}', _ENTITY, _ENTITY])
    assert caught.value.code == "RW-WIRE-003"


def test_an_interleaved_entity_frame_is_rejected() -> None:
    mismatched = '{"kind":"entity","frame":99,"index":0,"class":"u","x":0,"y":0}'
    with pytest.raises(WireError) as caught:
        decode_samples([_FRAME, mismatched])
    assert caught.value.code == "RW-WIRE-004"


def test_an_unknown_record_kind_is_rejected() -> None:
    with pytest.raises(WireError) as caught:
        decode_samples(['{"kind":"weather","frame":1}'])
    assert caught.value.code == "RW-WIRE-001"


def test_a_missing_field_propagates_as_a_decode_error() -> None:
    with pytest.raises(DecodeError) as caught:
        decode_samples(['{"kind":"frame","frame":7,"clock_ms":25}'])
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
        entities=(Entity(index=0, class_name='a"b\\c\nd\te\rf\x01g', x=0.0, y=0.0),),
    )
    lines = encode_sample(hostile)
    assert len(lines) == 2
    assert decode_samples(list(lines)) == (hostile,)
