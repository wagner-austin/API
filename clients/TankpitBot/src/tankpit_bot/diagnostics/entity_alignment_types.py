"""Strict TypedDict payloads for belief-vs-truth entity alignment.

An entity alignment *sample* pairs the bot's wire-derived container
beliefs (``world_state["containers"]``) with the live JS client's
array-of-objects properties of ``activeGame.h``
(:attr:`tankpit_bot.browser.page_client_snapshot.PageClientSnapshotDict.world_collections`)
captured at the same tick. The mapping report aggregates samples to
discover which minified collection carries the client's container list
and which item fields carry x/y -- the prerequisite for detecting
containers the client renders but the bot never learned about.
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
    decode_client_collections,
    encode_client_collections,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
    decode_container_state,
    encode_container_state,
)


class EntityAlignmentSampleDict(TypedDict):
    """One container belief-vs-truth capture taken at a single tick.

    Attributes:
        timestamp: Wall-clock timestamp of the emitting event record.
        belief_containers: Every container the bot's world state tracked
            at capture time.
        world_collections: Live JS client array-of-objects properties of
            ``activeGame.h`` captured in the same tick, keyed by
            minified property name.
    """

    timestamp: str
    belief_containers: list[ContainerStateDict]
    world_collections: dict[str, list[dict[str, int | float | bool | str | None]]]


class EntityCollectionCandidateDict(TypedDict):
    """Discovery result for one minified collection key.

    Attributes:
        collection_key: Minified property name of the collection on
            ``activeGame.h``.
        x_key: Item field name whose value best matches belief container
            X coordinates. Empty when no field pair matches anything.
        y_key: Item field name paired with ``x_key`` for Y. Empty when
            no field pair matches anything.
        matched_items: Total items (across samples) whose
            ``(x_key, y_key)`` pair equals a belief container position.
        total_items: Total items observed in this collection across
            samples.
        belief_matched: Total belief containers (across samples) that
            had at least one matching item under the best pair.
        belief_total: Total belief containers observed across samples.
    """

    collection_key: str
    x_key: str
    y_key: str
    matched_items: int
    total_items: int
    belief_matched: int
    belief_total: int


class EntityMapReportDict(TypedDict):
    """Aggregated entity-mapping discovery report for one artifact.

    Attributes:
        source_path: Artifact path the report was built from.
        mode: Runtime mode string recorded in the artifact.
        sample_count: Total ``entity_alignment_sample`` records found.
        candidates: One candidate row per minified collection key,
            sorted by descending ``matched_items`` then key name.
    """

    source_path: str
    mode: str
    sample_count: int
    candidates: list[EntityCollectionCandidateDict]


def encode_entity_alignment_sample(sample: EntityAlignmentSampleDict) -> JSONObject:
    """Encode an entity alignment sample to JSON.

    Args:
        sample: Sample to encode.

    Returns:
        JSON-compatible representation.
    """
    containers: list[JSONValue] = [encode_container_state(c) for c in sample["belief_containers"]]
    return {
        "timestamp": sample["timestamp"],
        "belief_containers": containers,
        "world_collections": encode_client_collections(sample["world_collections"]),
    }


def decode_entity_alignment_sample(data: JSONObject) -> EntityAlignmentSampleDict:
    """Decode an entity alignment sample from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated sample.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    raw_containers = require_list(data, "belief_containers")
    containers: list[ContainerStateDict] = []
    for index, raw in enumerate(raw_containers):
        if not isinstance(raw, dict):
            raise JSONTypeError(
                f"belief_containers[{index}] must be object, got {type(raw).__name__}"
            )
        containers.append(decode_container_state(raw))
    return EntityAlignmentSampleDict(
        timestamp=require_str(data, "timestamp"),
        belief_containers=containers,
        world_collections=decode_client_collections(
            require_dict(data, "world_collections"), field="world_collections"
        ),
    )


def encode_entity_collection_candidate(
    candidate: EntityCollectionCandidateDict,
) -> JSONObject:
    """Encode a per-collection discovery candidate to JSON.

    Args:
        candidate: Candidate row to encode.

    Returns:
        JSON-compatible representation.
    """
    return {
        "collection_key": candidate["collection_key"],
        "x_key": candidate["x_key"],
        "y_key": candidate["y_key"],
        "matched_items": candidate["matched_items"],
        "total_items": candidate["total_items"],
        "belief_matched": candidate["belief_matched"],
        "belief_total": candidate["belief_total"],
    }


def decode_entity_collection_candidate(data: JSONObject) -> EntityCollectionCandidateDict:
    """Decode a per-collection discovery candidate from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated candidate row.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    return EntityCollectionCandidateDict(
        collection_key=require_str(data, "collection_key"),
        x_key=require_str(data, "x_key"),
        y_key=require_str(data, "y_key"),
        matched_items=require_int(data, "matched_items"),
        total_items=require_int(data, "total_items"),
        belief_matched=require_int(data, "belief_matched"),
        belief_total=require_int(data, "belief_total"),
    )


def encode_entity_map_report(report: EntityMapReportDict) -> JSONObject:
    """Encode an entity-mapping discovery report to JSON.

    Args:
        report: Report to encode.

    Returns:
        JSON-compatible representation.
    """
    candidates: list[JSONValue] = [
        encode_entity_collection_candidate(c) for c in report["candidates"]
    ]
    return {
        "source_path": report["source_path"],
        "mode": report["mode"],
        "sample_count": report["sample_count"],
        "candidates": candidates,
    }


def decode_entity_map_report(data: JSONObject) -> EntityMapReportDict:
    """Decode an entity-mapping discovery report from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated report.

    Raises:
        JSONTypeError: When required fields are missing or invalid.
    """
    raw_candidates = require_list(data, "candidates")
    candidates: list[EntityCollectionCandidateDict] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise JSONTypeError(f"candidates[{index}] must be object, got {type(raw).__name__}")
        candidates.append(decode_entity_collection_candidate(raw))
    return EntityMapReportDict(
        source_path=require_str(data, "source_path"),
        mode=require_str(data, "mode"),
        sample_count=require_int(data, "sample_count"),
        candidates=candidates,
    )


__all__ = [
    "EntityAlignmentSampleDict",
    "EntityCollectionCandidateDict",
    "EntityMapReportDict",
    "decode_entity_alignment_sample",
    "decode_entity_collection_candidate",
    "decode_entity_map_report",
    "encode_entity_alignment_sample",
    "encode_entity_collection_candidate",
    "encode_entity_map_report",
]
