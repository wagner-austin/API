"""Round-trip tests for the self-alignment TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.diagnostics.self_alignment_types import (
    SelfAlignmentSampleDict,
    SelfFieldCandidateDict,
    SelfMapReportDict,
    decode_self_alignment_sample,
    decode_self_field_candidate,
    decode_self_map_report,
    encode_self_alignment_sample,
    encode_self_field_candidate,
    encode_self_map_report,
)


def _round_trip(encoded: JSONObject) -> JSONObject:
    """Round-trip a dict through ``dump_json_str`` / ``load_json_str``."""
    return narrow_json_to_dict(load_json_str(dump_json_str(encoded)))


def _make_sample() -> SelfAlignmentSampleDict:
    """Return a sample carrying every primitive value shape."""
    return SelfAlignmentSampleDict(
        timestamp="2026-06-09T12:00:00",
        belief_tank_id=99,
        belief_x=131,
        belief_y=110,
        belief_fuel=1100,
        self_fields={
            "a": 131,
            "b": 110,
            "cy": 1100,
            "ratio": 0.5,
            "flag": True,
            "name": "tank",
            "n": None,
        },
    )


def test_self_alignment_sample_round_trip() -> None:
    """``SelfAlignmentSampleDict`` round-trips through JSON encoding."""
    sample = _make_sample()

    decoded = decode_self_alignment_sample(_round_trip(encode_self_alignment_sample(sample)))

    assert decoded == sample


def test_self_alignment_sample_rejects_non_int_belief() -> None:
    """A non-int ``belief_x`` raises ``JSONTypeError`` at decode."""
    raw = encode_self_alignment_sample(_make_sample())
    raw["belief_x"] = "131"

    with pytest.raises(JSONTypeError, match="belief_x"):
        decode_self_alignment_sample(raw)


def test_self_alignment_sample_rejects_non_primitive_self_field() -> None:
    """A nested object inside ``self_fields`` raises at decode."""
    raw = encode_self_alignment_sample(_make_sample())
    raw["self_fields"] = {"a": [1, 2]}

    with pytest.raises(JSONTypeError, match=r"self_fields.*'a'.*JSON primitive"):
        decode_self_alignment_sample(raw)


def test_self_alignment_sample_rejects_missing_self_fields() -> None:
    """A missing ``self_fields`` object raises at decode."""
    raw = encode_self_alignment_sample(_make_sample())
    del raw["self_fields"]

    with pytest.raises(JSONTypeError, match="self_fields"):
        decode_self_alignment_sample(raw)


def _make_candidate() -> SelfFieldCandidateDict:
    """Return a fully populated candidate row."""
    return SelfFieldCandidateDict(
        dimension="x",
        matching_keys=["a", "fx"],
        distinct_belief_values=3,
        sample_count=5,
    )


def test_self_field_candidate_round_trip() -> None:
    """``SelfFieldCandidateDict`` round-trips through JSON encoding."""
    candidate = _make_candidate()

    decoded = decode_self_field_candidate(_round_trip(encode_self_field_candidate(candidate)))

    assert decoded == candidate


def test_self_field_candidate_rejects_non_str_matching_key() -> None:
    """A non-str entry in ``matching_keys`` raises at decode."""
    raw = encode_self_field_candidate(_make_candidate())
    raw["matching_keys"] = ["a", 7]

    with pytest.raises(JSONTypeError, match=r"matching_keys\[1\] must be str"):
        decode_self_field_candidate(raw)


def _make_report() -> SelfMapReportDict:
    """Return a report carrying one candidate per dimension."""
    return SelfMapReportDict(
        source_path="runs/bot/latest.events.jsonl",
        mode="bot",
        sample_count=5,
        candidates=[
            SelfFieldCandidateDict(
                dimension="tank_id",
                matching_keys=["A"],
                distinct_belief_values=1,
                sample_count=5,
            ),
            _make_candidate(),
        ],
    )


def test_self_map_report_round_trip() -> None:
    """``SelfMapReportDict`` round-trips through JSON encoding."""
    report = _make_report()

    decoded = decode_self_map_report(_round_trip(encode_self_map_report(report)))

    assert decoded == report


def test_self_map_report_rejects_non_object_candidate() -> None:
    """A non-object element in ``candidates`` raises at decode."""
    raw = encode_self_map_report(_make_report())
    raw["candidates"] = ["bad"]

    with pytest.raises(JSONTypeError, match=r"candidates\[0\] must be object"):
        decode_self_map_report(raw)
