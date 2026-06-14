"""Discover the minified self-field key mapping from alignment samples.

Reads a JSONL events artifact, collects every ``self_alignment_sample``
DIAGNOSTIC (emitted by
:func:`tankpit_bot.diagnostics.self_alignment.maybe_emit_self_alignment_sample`),
and for each belief dimension (tank_id, x, y, fuel) intersects the
minified ``activeGame.i`` keys whose numeric value equals the belief
value in EVERY sample. Keys that survive across many distinct belief
values are the semantic mapping; a key that only matches one distinct
value may be a constant coincidence and is reported with its low
confidence visible.

This is the offline half of the belief-vs-truth divergence detector:
once the mapping is confirmed stable across runs, the live detector can
compare belief to truth per tick and flag drift.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot.diagnostics.event_stream import (
    load_event_records,
    run_analyzer_cli,
    scan_diagnostic_records,
)
from tankpit_bot.diagnostics.self_alignment_types import (
    SelfAlignmentSampleDict,
    SelfFieldCandidateDict,
    SelfMapReportDict,
    decode_self_alignment_sample,
)
from tankpit_bot.runtime_logging import RuntimeEventRecordDict, require_str_field

log = get_logger(__name__)


_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("tank_id", "belief_tank_id"),
    ("x", "belief_x"),
    ("y", "belief_y"),
    ("fuel", "belief_fuel"),
)


def _classify_self_alignment_sample(
    record: RuntimeEventRecordDict,
) -> SelfAlignmentSampleDict:
    """Build a typed alignment sample from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``self_alignment_sample``.

    Returns:
        Strict-typed alignment sample.

    Raises:
        KeyError: When ``self_fields_json`` is absent from the record.
        JSONTypeError: When any belief field is absent/mistyped or the
            ``self_fields_json`` payload fails strict decoding.
    """
    fields = record["fields"]
    raw_map = narrow_json_to_dict(load_json_str(require_str_field(fields, "self_fields_json")))
    payload: JSONObject = {
        "timestamp": record["timestamp"],
        "belief_tank_id": fields.get("belief_tank_id"),
        "belief_x": fields.get("belief_x"),
        "belief_y": fields.get("belief_y"),
        "belief_fuel": fields.get("belief_fuel"),
        "self_fields": raw_map,
    }
    return decode_self_alignment_sample(payload)


def _belief_value(sample: SelfAlignmentSampleDict, belief_field: str) -> int:
    """Return one belief dimension value from a sample.

    Args:
        sample: Alignment sample to read.
        belief_field: One of ``belief_tank_id`` / ``belief_x`` /
            ``belief_y`` / ``belief_fuel``.

    Returns:
        The belief value for that dimension.

    Raises:
        ValueError: When ``belief_field`` is not a known dimension;
            the dimension table is the single source of valid names.
    """
    if belief_field == "belief_tank_id":
        return sample["belief_tank_id"]
    if belief_field == "belief_x":
        return sample["belief_x"]
    if belief_field == "belief_y":
        return sample["belief_y"]
    if belief_field == "belief_fuel":
        return sample["belief_fuel"]
    raise ValueError(f"unknown belief dimension field: {belief_field!r}")


def _keys_matching_belief(sample: SelfAlignmentSampleDict, belief: int) -> set[str]:
    """Return the minified keys whose numeric value equals ``belief``.

    Booleans are excluded explicitly: Python treats ``True == 1`` as
    numeric equality, but a boolean client flag can never be the
    semantic carrier of a coordinate / fuel / ID value.

    Args:
        sample: Alignment sample to scan.
        belief: Belief value to match against.

    Returns:
        Set of matching minified key names.
    """
    matching: set[str] = set()
    for key, value in sample["self_fields"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value == belief:
            matching.add(key)
    return matching


def _candidate_for_dimension(
    samples: list[SelfAlignmentSampleDict],
    dimension: str,
    belief_field: str,
) -> SelfFieldCandidateDict:
    """Intersect matching keys for one belief dimension across all samples.

    Args:
        samples: Every alignment sample found in the artifact.
        dimension: Human-facing dimension name.
        belief_field: Sample field carrying the belief value.

    Returns:
        Candidate row with the surviving keys and confidence counters.
    """
    surviving: set[str] | None = None
    belief_values: set[int] = set()
    for sample in samples:
        belief = _belief_value(sample, belief_field)
        belief_values.add(belief)
        matching = _keys_matching_belief(sample, belief)
        surviving = matching if surviving is None else surviving & matching
    return SelfFieldCandidateDict(
        dimension=dimension,
        matching_keys=sorted(surviving) if surviving is not None else [],
        distinct_belief_values=len(belief_values),
        sample_count=len(samples),
    )


def build_self_map_report(source_path: Path) -> SelfMapReportDict:
    """Build a :class:`SelfMapReportDict` from a JSONL events artifact.

    Args:
        source_path: Path to a runtime events JSONL artifact.

    Returns:
        Aggregated mapping-discovery report.

    Raises:
        FileNotFoundError: When ``source_path`` does not exist on disk.
        JSONTypeError: When any event line or sample payload fails
            strict decoding; malformed artifacts are surfaced instead
            of silently dropped.
    """
    records = load_event_records(source_path)
    mode, matches = scan_diagnostic_records(records, "self_alignment_sample")
    samples = [_classify_self_alignment_sample(record) for record in matches]
    candidates = [
        _candidate_for_dimension(samples, dimension, belief_field)
        for dimension, belief_field in _DIMENSIONS
    ]
    return SelfMapReportDict(
        source_path=str(source_path),
        mode=mode,
        sample_count=len(samples),
        candidates=candidates,
    )


def _render_candidate_lines(candidate: SelfFieldCandidateDict) -> list[str]:
    """Return the rendered lines for one dimension's candidate row."""
    keys = ", ".join(candidate["matching_keys"]) if candidate["matching_keys"] else "(none)"
    lines = [
        f"  {candidate['dimension']:8s} -> {keys}",
        f"           distinct_belief_values={candidate['distinct_belief_values']} "
        f"sample_count={candidate['sample_count']}",
    ]
    if not candidate["matching_keys"]:
        lines.append("           NO KEY tracks this dimension -- truth source missing or renamed")
    elif candidate["distinct_belief_values"] <= 1:
        lines.append(
            "           LOW CONFIDENCE -- only one distinct value observed; "
            "rerun with more movement/fuel change"
        )
    elif len(candidate["matching_keys"]) > 1:
        lines.append("           AMBIGUOUS -- multiple keys survived; collect more varied samples")
    return lines


def render_self_map_report(report: SelfMapReportDict) -> str:
    """Render a :class:`SelfMapReportDict` to a human-readable string.

    Args:
        report: Report to render.

    Returns:
        Multi-line string suitable for printing to a terminal.
    """
    lines: list[str] = ["=" * 72, "TANKPIT SELF-FIELD MAPPING DISCOVERY", "=" * 72]
    lines.append(f"Source:  {report['source_path']}")
    lines.append(f"Mode:    {report['mode']}")
    lines.append(f"Samples: {report['sample_count']}")
    lines.append("")
    lines.append("=== CANDIDATE MINIFIED KEYS PER BELIEF DIMENSION ===")
    if report["sample_count"] == 0:
        lines.append("  (no self_alignment_sample events in artifact -- run the bot first)")
    else:
        for candidate in report["candidates"]:
            lines.extend(_render_candidate_lines(candidate))
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    """Run the ``tankpit-self-map`` CLI entrypoint.

    Reads a JSONL events artifact (path resolved from the user-supplied
    args -- ``sys.argv`` with the script name stripped -- defaulting to
    ``runs/bot/latest.events.jsonl``), builds a
    :class:`SelfMapReportDict`, and prints it to the rich console
    logger.

    Returns:
        Process exit code (``0`` on success). Errors propagate as
        exceptions.
    """
    return run_analyzer_cli(build_self_map_report, render_self_map_report, log)


__all__ = [
    "build_self_map_report",
    "main",
    "render_self_map_report",
]
