"""Strict TypedDict payloads for belief-vs-truth self-state alignment.

A *sample* pairs the bot's wire-derived belief about its own tank
(:class:`tankpit_bot.state.types.SelfStateDict` fields) with the live
JS client's minified self-tank field map
(:attr:`tankpit_bot.browser.page_client_snapshot.PageClientSnapshotDict.self_fields`)
captured at the same tick. The mapping report aggregates samples to
discover which minified keys carry tank_id / x / y / fuel.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.browser.page_client_snapshot_codecs import (
    decode_client_field_map,
    encode_client_field_map,
)


class SelfAlignmentSampleDict(TypedDict):
    """One belief-vs-truth capture taken at a single tick.

    Attributes:
        timestamp: Wall-clock timestamp of the emitting event record.
        belief_tank_id: Bot's wire-derived tank ID at capture time.
        belief_x: Bot's wire-derived X coordinate at capture time.
        belief_y: Bot's wire-derived Y coordinate at capture time.
        belief_fuel: Bot's wire-derived fuel at capture time.
        self_fields: Live JS client self-tank field map keyed by
            minified property name, captured in the same tick.
    """

    timestamp: str
    belief_tank_id: int
    belief_x: int
    belief_y: int
    belief_fuel: int
    self_fields: dict[str, int | float | bool | str | None]


class SelfFieldCandidateDict(TypedDict):
    """Mapping candidates for one belief dimension.

    Attributes:
        dimension: Belief dimension name (``tank_id``, ``x``, ``y``,
            ``fuel``).
        matching_keys: Minified self-field keys whose numeric value
            equals the belief value in EVERY sample. Sorted for stable
            output.
        distinct_belief_values: Number of distinct belief values
            observed across samples. ``1`` means the match could be a
            constant coincidence; higher counts mean higher confidence.
        sample_count: Number of samples the intersection ran over.
    """

    dimension: str
    matching_keys: list[str]
    distinct_belief_values: int
    sample_count: int


class SelfMapReportDict(TypedDict):
    """Aggregated mapping-discovery report for one artifact.

    Attributes:
        source_path: Artifact path the report was built from.
        mode: Runtime mode string recorded in the artifact (``bot``,
            ``probe:<name>``, ...).
        sample_count: Total ``self_alignment_sample`` records found.
        candidates: One candidate row per belief dimension, in the
            fixed order tank_id, x, y, fuel.
    """

    source_path: str
    mode: str
    sample_count: int
    candidates: list[SelfFieldCandidateDict]


def encode_self_alignment_sample(sample: SelfAlignmentSampleDict) -> JSONObject:
    """Encode a self-alignment sample to JSON.

    Args:
        sample: Sample to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "timestamp": sample["timestamp"],
        "belief_tank_id": sample["belief_tank_id"],
        "belief_x": sample["belief_x"],
        "belief_y": sample["belief_y"],
        "belief_fuel": sample["belief_fuel"],
        "self_fields": encode_client_field_map(sample["self_fields"]),
    }


def decode_self_alignment_sample(data: JSONObject) -> SelfAlignmentSampleDict:
    """Decode a self-alignment sample from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated sample.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    return SelfAlignmentSampleDict(
        timestamp=require_str(data, "timestamp"),
        belief_tank_id=require_int(data, "belief_tank_id"),
        belief_x=require_int(data, "belief_x"),
        belief_y=require_int(data, "belief_y"),
        belief_fuel=require_int(data, "belief_fuel"),
        self_fields=decode_client_field_map(require_dict(data, "self_fields"), field="self_fields"),
    )


def encode_self_field_candidate(candidate: SelfFieldCandidateDict) -> JSONObject:
    """Encode a per-dimension mapping candidate to JSON.

    Args:
        candidate: Candidate row to encode.

    Returns:
        JSON-compatible representation.
    """
    matching_keys: list[JSONValue] = list(candidate["matching_keys"])
    return {
        "dimension": candidate["dimension"],
        "matching_keys": matching_keys,
        "distinct_belief_values": candidate["distinct_belief_values"],
        "sample_count": candidate["sample_count"],
    }


def decode_self_field_candidate(data: JSONObject) -> SelfFieldCandidateDict:
    """Decode a per-dimension mapping candidate from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated candidate row.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    raw_keys = require_list(data, "matching_keys")
    matching_keys: list[str] = []
    for index, raw in enumerate(raw_keys):
        if not isinstance(raw, str):
            raise JSONTypeError(f"matching_keys[{index}] must be str, got {type(raw).__name__}")
        matching_keys.append(raw)
    return SelfFieldCandidateDict(
        dimension=require_str(data, "dimension"),
        matching_keys=matching_keys,
        distinct_belief_values=require_int(data, "distinct_belief_values"),
        sample_count=require_int(data, "sample_count"),
    )


def encode_self_map_report(report: SelfMapReportDict) -> JSONObject:
    """Encode a mapping-discovery report to JSON.

    Args:
        report: Report to encode.

    Returns:
        JSON-compatible representation.
    """
    candidates: list[JSONValue] = [encode_self_field_candidate(c) for c in report["candidates"]]
    return {
        "source_path": report["source_path"],
        "mode": report["mode"],
        "sample_count": report["sample_count"],
        "candidates": candidates,
    }


def decode_self_map_report(data: JSONObject) -> SelfMapReportDict:
    """Decode a mapping-discovery report from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated report.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    raw_candidates = require_list(data, "candidates")
    candidates: list[SelfFieldCandidateDict] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise JSONTypeError(f"candidates[{index}] must be object, got {type(raw).__name__}")
        candidates.append(decode_self_field_candidate(raw))
    return SelfMapReportDict(
        source_path=require_str(data, "source_path"),
        mode=require_str(data, "mode"),
        sample_count=require_int(data, "sample_count"),
        candidates=candidates,
    )


__all__ = [
    "SelfAlignmentSampleDict",
    "SelfFieldCandidateDict",
    "SelfMapReportDict",
    "decode_self_alignment_sample",
    "decode_self_field_candidate",
    "decode_self_map_report",
    "encode_self_alignment_sample",
    "encode_self_field_candidate",
    "encode_self_map_report",
]
