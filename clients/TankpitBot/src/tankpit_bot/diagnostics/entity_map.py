"""Discover the client's container collection from entity alignment samples.

Reads a JSONL events artifact, collects every ``entity_alignment_sample``
DIAGNOSTIC (emitted by
:func:`tankpit_bot.diagnostics.entity_alignment.maybe_emit_entity_alignment_sample`),
and for each minified ``activeGame.h`` collection searches all ordered
pairs of numeric item fields for the ``(x_key, y_key)`` pair that best
matches the bot's wire-derived container positions. A collection whose
items consistently land on belief container coordinates IS the client's
container list; once identified, items in it that the bot does NOT
track are containers the bot is blind to (for example containers
discovered by other players before the bot joined the room).
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot.diagnostics.entity_alignment_types import (
    EntityAlignmentSampleDict,
    EntityCollectionCandidateDict,
    EntityMapReportDict,
    decode_entity_alignment_sample,
)
from tankpit_bot.diagnostics.event_stream import (
    load_event_records,
    run_analyzer_cli,
    scan_diagnostic_records,
)
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    require_str_field,
)

log = get_logger(__name__)


def _classify_entity_alignment_sample(
    record: RuntimeEventRecordDict,
) -> EntityAlignmentSampleDict:
    """Build a typed entity alignment sample from a DIAGNOSTIC event.

    Reassembles the two JSON-string fields into the canonical sample
    payload shape and delegates validation to
    :func:`tankpit_bot.diagnostics.entity_alignment_types.decode_entity_alignment_sample`
    so emit and decode can never drift apart.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``entity_alignment_sample``.

    Returns:
        Strict-typed entity alignment sample.

    Raises:
        KeyError: When a JSON-string field is absent from the record.
        JSONTypeError: When either payload fails strict decoding.
    """
    fields = record["fields"]
    belief_raw = narrow_json_to_dict(
        load_json_str(require_str_field(fields, "belief_containers_json"))
    )
    collections_raw = narrow_json_to_dict(
        load_json_str(require_str_field(fields, "world_collections_json"))
    )
    payload: JSONObject = {
        "timestamp": record["timestamp"],
        "belief_containers": belief_raw.get("containers"),
        "world_collections": collections_raw,
    }
    return decode_entity_alignment_sample(payload)


def _numeric_item_fields(
    item: dict[str, int | float | bool | str | None],
) -> dict[str, int | float]:
    """Return the numeric (non-bool) fields of one collection item."""
    numeric: dict[str, int | float] = {}
    for key, value in item.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric[key] = value
    return numeric


def _score_field_pairs(
    samples: list[EntityAlignmentSampleDict],
    collection_key: str,
) -> dict[tuple[str, str], int]:
    """Count belief-position hits for every ordered numeric field pair.

    Args:
        samples: Every entity alignment sample found in the artifact.
        collection_key: Minified collection name to score.

    Returns:
        Mapping of ``(x_key, y_key)`` to the number of items (across
        all samples) whose pair of values equals a belief container
        position.
    """
    scores: dict[tuple[str, str], int] = {}
    for sample in samples:
        positions = {(c["x"], c["y"]) for c in sample["belief_containers"]}
        if not positions:
            continue
        for item in sample["world_collections"].get(collection_key, []):
            numeric = _numeric_item_fields(item)
            for x_key, x_value in numeric.items():
                for y_key, y_value in numeric.items():
                    if x_key == y_key:
                        continue
                    if (x_value, y_value) in positions:
                        scores[(x_key, y_key)] = scores.get((x_key, y_key), 0) + 1
    return scores


def _candidate_counts(
    samples: list[EntityAlignmentSampleDict],
    collection_key: str,
    x_key: str,
    y_key: str,
) -> tuple[int, int, int, int]:
    """Compute the match counters for one collection under a chosen pair.

    Args:
        samples: Every entity alignment sample found in the artifact.
        collection_key: Minified collection name to count.
        x_key: Chosen item field carrying X.
        y_key: Chosen item field carrying Y.

    Returns:
        ``(matched_items, total_items, belief_matched, belief_total)``
        aggregated across all samples.
    """
    matched_items = 0
    total_items = 0
    belief_matched = 0
    belief_total = 0
    for sample in samples:
        items = sample["world_collections"].get(collection_key, [])
        total_items += len(items)
        item_positions: set[tuple[int | float, int | float]] = set()
        for item in items:
            numeric = _numeric_item_fields(item)
            if x_key in numeric and y_key in numeric:
                position = (numeric[x_key], numeric[y_key])
                item_positions.add(position)
        positions = {(c["x"], c["y"]) for c in sample["belief_containers"]}
        matched_items += sum(1 for position in item_positions if position in positions)
        belief_total += len(sample["belief_containers"])
        belief_matched += sum(1 for position in positions if position in item_positions)
    return (matched_items, total_items, belief_matched, belief_total)


def _candidate_for_collection(
    samples: list[EntityAlignmentSampleDict],
    collection_key: str,
) -> EntityCollectionCandidateDict:
    """Build the discovery candidate row for one collection key.

    Args:
        samples: Every entity alignment sample found in the artifact.
        collection_key: Minified collection name to analyze.

    Returns:
        Candidate row with the best-scoring field pair and counters.
    """
    scores = _score_field_pairs(samples, collection_key)
    if not scores:
        total_items = sum(
            len(sample["world_collections"].get(collection_key, [])) for sample in samples
        )
        belief_total = sum(len(sample["belief_containers"]) for sample in samples)
        return EntityCollectionCandidateDict(
            collection_key=collection_key,
            x_key="",
            y_key="",
            matched_items=0,
            total_items=total_items,
            belief_matched=0,
            belief_total=belief_total,
        )

    def _pair_rank(pair: tuple[str, str]) -> tuple[int, tuple[str, str]]:
        """Order pairs best-score-first with a stable lexicographic tiebreak."""
        return (-scores[pair], pair)

    best_pair = min(scores, key=_pair_rank)
    matched_items, total_items, belief_matched, belief_total = _candidate_counts(
        samples, collection_key, best_pair[0], best_pair[1]
    )
    return EntityCollectionCandidateDict(
        collection_key=collection_key,
        x_key=best_pair[0],
        y_key=best_pair[1],
        matched_items=matched_items,
        total_items=total_items,
        belief_matched=belief_matched,
        belief_total=belief_total,
    )


def build_entity_map_report(source_path: Path) -> EntityMapReportDict:
    """Build an :class:`EntityMapReportDict` from a JSONL events artifact.

    Args:
        source_path: Path to a runtime events JSONL artifact.

    Returns:
        Aggregated entity-mapping discovery report.

    Raises:
        FileNotFoundError: When ``source_path`` does not exist on disk.
        JSONTypeError: When any event line or sample payload fails
            strict decoding; malformed artifacts are surfaced instead
            of silently dropped.
    """
    records = load_event_records(source_path)
    mode, matches = scan_diagnostic_records(records, "entity_alignment_sample")
    samples = [_classify_entity_alignment_sample(record) for record in matches]
    collection_keys: set[str] = set()
    for sample in samples:
        collection_keys.update(sample["world_collections"])
    candidates = [_candidate_for_collection(samples, key) for key in sorted(collection_keys)]
    candidates.sort(key=lambda c: (-c["matched_items"], c["collection_key"]))
    return EntityMapReportDict(
        source_path=str(source_path),
        mode=mode,
        sample_count=len(samples),
        candidates=candidates,
    )


def _render_candidate_lines(candidate: EntityCollectionCandidateDict) -> list[str]:
    """Return the rendered lines for one collection's candidate row."""
    if not candidate["x_key"]:
        return [
            f"  {candidate['collection_key']:8s} -> no coordinate pair matches "
            f"belief containers (items={candidate['total_items']})",
        ]
    lines = [
        f"  {candidate['collection_key']:8s} -> x={candidate['x_key']} y={candidate['y_key']}",
        f"           items matching a belief container: "
        f"{candidate['matched_items']}/{candidate['total_items']}",
        f"           belief containers found in client list: "
        f"{candidate['belief_matched']}/{candidate['belief_total']}",
    ]
    unmatched = candidate["total_items"] - candidate["matched_items"]
    if unmatched > 0:
        lines.append(
            f"           DIVERGENCE -- {unmatched} client item(s) the bot does not "
            "track (or non-container entries in a mixed list)"
        )
    return lines


def render_entity_map_report(report: EntityMapReportDict) -> str:
    """Render an :class:`EntityMapReportDict` to a human-readable string.

    Args:
        report: Report to render.

    Returns:
        Multi-line string suitable for printing to a terminal.
    """
    lines: list[str] = ["=" * 72, "TANKPIT ENTITY COLLECTION DISCOVERY", "=" * 72]
    lines.append(f"Source:  {report['source_path']}")
    lines.append(f"Mode:    {report['mode']}")
    lines.append(f"Samples: {report['sample_count']}")
    lines.append("")
    lines.append("=== CLIENT COLLECTIONS vs BOT CONTAINER BELIEFS ===")
    if report["sample_count"] == 0:
        lines.append("  (no entity_alignment_sample events in artifact -- run the bot first)")
    else:
        for candidate in report["candidates"]:
            lines.extend(_render_candidate_lines(candidate))
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    """Run the ``tankpit-entity-map`` CLI entrypoint.

    Reads a JSONL events artifact (path resolved from the user-supplied
    args -- ``sys.argv`` with the script name stripped -- defaulting to
    ``runs/bot/latest.events.jsonl``), builds an
    :class:`EntityMapReportDict`, and prints it to the rich console
    logger.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    return run_analyzer_cli(build_entity_map_report, render_entity_map_report, log)


__all__ = [
    "build_entity_map_report",
    "main",
    "render_entity_map_report",
]
