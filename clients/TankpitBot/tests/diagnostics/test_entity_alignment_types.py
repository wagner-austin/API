"""Round-trip tests for the entity-alignment TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.diagnostics.entity_alignment_types import (
    EntityAlignmentSampleDict,
    EntityCollectionCandidateDict,
    EntityMapReportDict,
    decode_entity_alignment_sample,
    decode_entity_collection_candidate,
    decode_entity_map_report,
    encode_entity_alignment_sample,
    encode_entity_collection_candidate,
    encode_entity_map_report,
)
from tankpit_bot.state.types import make_container_state


def _round_trip(encoded: JSONObject) -> JSONObject:
    """Round-trip a dict through ``dump_json_str`` / ``load_json_str``."""
    return narrow_json_to_dict(load_json_str(dump_json_str(encoded)))


def _make_sample() -> EntityAlignmentSampleDict:
    """Return a sample carrying containers and mixed-type collections."""
    return EntityAlignmentSampleDict(
        timestamp="2026-06-10T12:00:00",
        belief_containers=[
            make_container_state(146, 44, True, 500, timestamp_ms=1000),
            make_container_state(150, 48, False, 0, timestamp_ms=1200),
        ],
        world_collections={
            "ba": [
                {"u": 146, "v": 44, "w": True},
                {"u": 150, "v": 48, "w": False},
            ],
            "cc": [{"n": "Artax", "z": None, "s": 346}],
        },
    )


def test_entity_alignment_sample_round_trip() -> None:
    """``EntityAlignmentSampleDict`` round-trips through JSON encoding."""
    sample = _make_sample()

    decoded = decode_entity_alignment_sample(_round_trip(encode_entity_alignment_sample(sample)))

    assert decoded == sample


def test_entity_alignment_sample_rejects_non_object_container() -> None:
    """A non-object element in ``belief_containers`` raises at decode."""
    raw = encode_entity_alignment_sample(_make_sample())
    raw["belief_containers"] = ["bad"]

    with pytest.raises(JSONTypeError, match=r"belief_containers\[0\] must be object"):
        decode_entity_alignment_sample(raw)


def test_entity_alignment_sample_rejects_missing_collections() -> None:
    """A missing ``world_collections`` object raises at decode."""
    raw = encode_entity_alignment_sample(_make_sample())
    del raw["world_collections"]

    with pytest.raises(JSONTypeError, match="world_collections"):
        decode_entity_alignment_sample(raw)


def _make_candidate() -> EntityCollectionCandidateDict:
    """Return a fully populated candidate row."""
    return EntityCollectionCandidateDict(
        collection_key="ba",
        x_key="u",
        y_key="v",
        matched_items=3,
        total_items=5,
        belief_matched=3,
        belief_total=3,
    )


def test_entity_collection_candidate_round_trip() -> None:
    """``EntityCollectionCandidateDict`` round-trips through JSON encoding."""
    candidate = _make_candidate()

    decoded = decode_entity_collection_candidate(
        _round_trip(encode_entity_collection_candidate(candidate))
    )

    assert decoded == candidate


def test_entity_collection_candidate_rejects_non_int_count() -> None:
    """A non-int ``matched_items`` raises ``JSONTypeError`` at decode."""
    raw = encode_entity_collection_candidate(_make_candidate())
    raw["matched_items"] = "3"

    with pytest.raises(JSONTypeError, match="matched_items"):
        decode_entity_collection_candidate(raw)


def _make_report() -> EntityMapReportDict:
    """Return a report carrying one candidate."""
    return EntityMapReportDict(
        source_path="runs/bot/latest.events.jsonl",
        mode="bot",
        sample_count=2,
        candidates=[_make_candidate()],
    )


def test_entity_map_report_round_trip() -> None:
    """``EntityMapReportDict`` round-trips through JSON encoding."""
    report = _make_report()

    decoded = decode_entity_map_report(_round_trip(encode_entity_map_report(report)))

    assert decoded == report


def test_entity_map_report_rejects_non_object_candidate() -> None:
    """A non-object element in ``candidates`` raises at decode."""
    raw = encode_entity_map_report(_make_report())
    raw["candidates"] = ["bad"]

    with pytest.raises(JSONTypeError, match=r"candidates\[0\] must be object"):
        decode_entity_map_report(raw)
