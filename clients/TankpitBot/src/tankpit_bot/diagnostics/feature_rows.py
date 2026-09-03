"""Tick-level feature table: one row per tick, for tabular modelling.

The run corpus is already tabular at RUN level -- ``runs/bot/_index.tsv``
is a design matrix with one row per session, and ``<stamp>.digest.json``
reduces a run to 28 fields. Neither answers a per-decision question:
"given this state, this action was chosen, and this is what happened".

That join has always been derivable and never derived. Every diagnostic
in an events artifact except ``session_room_joined`` carries ``tick_n``
and ``bot_state``, so the stream reshapes into one row per tick with no
new instrumentation -- this module is a DERIVATION over data already on
disk, not a new emitter.

What a row deliberately is NOT: an inferred label. The action and its
outcome are recorded facts from ``action_outcome``; the per-kind counts
are counts of what the tick actually emitted. Nothing here scores a
decision as good or bad, because the artifact does not say so.

Provenance caveat for anyone modelling on this: the corpus is
SELF-OBSERVED. Every field is the bot's own belief at decision time,
which is the right frame for learning its policy and the wrong frame
for treating as ground truth about the world. The capture sessions are
the independent record.

CLI: ``tankpit-feature-rows [events.jsonl]`` (defaults to the latest bot
artifact) prints the table and writes ``<stem>.features.jsonl`` beside
the source, one JSON row per line.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.event_stream import load_event_records, run_analyzer_cli
from tankpit_bot.diagnostics.feature_provenance import (
    content_digest,
    feature_fingerprint,
    feature_run_record,
    write_run_record,
)
from tankpit_bot.diagnostics.feature_row_types import (
    COUNTED_KINDS,
    NO_ACTION,
    FeatureRowDict,
)
from tankpit_bot.runtime_records import RuntimeEventRecordDict

log = get_logger(__name__)


def encode_feature_row(row: FeatureRowDict) -> JSONObject:
    """Encode one feature row to a JSON-serializable dict.

    Args:
        row: The row to encode.

    Returns:
        JSON object carrying every field of the row.
    """
    return {
        "tick_n": row["tick_n"],
        "bot_state": row["bot_state"],
        "action_kind": row["action_kind"],
        "outcome": row["outcome"],
        "duration_ms": row["duration_ms"],
        "attempt_id": row["attempt_id"],
        "hop_declined": row["hop_declined"],
        "radar_dispatch": row["radar_dispatch"],
        "container_pickup_dispatched": row["container_pickup_dispatched"],
        "plan_released": row["plan_released"],
        "command_error": row["command_error"],
        "fleet_knowledge_merged": row["fleet_knowledge_merged"],
    }


def decode_feature_row(data: JSONObject) -> FeatureRowDict:
    """Decode one feature row from a JSON object.

    Args:
        data: JSON object to decode.

    Returns:
        The decoded row.

    Raises:
        JSONTypeError: If a field is missing or of the wrong type.
    """
    return FeatureRowDict(
        tick_n=require_int(data, "tick_n"),
        bot_state=require_str(data, "bot_state"),
        action_kind=require_str(data, "action_kind"),
        outcome=require_str(data, "outcome"),
        duration_ms=require_int(data, "duration_ms"),
        attempt_id=require_int(data, "attempt_id"),
        hop_declined=require_int(data, "hop_declined"),
        radar_dispatch=require_int(data, "radar_dispatch"),
        container_pickup_dispatched=require_int(data, "container_pickup_dispatched"),
        plan_released=require_int(data, "plan_released"),
        command_error=require_int(data, "command_error"),
        fleet_knowledge_merged=require_int(data, "fleet_knowledge_merged"),
    )


def _blank_row(tick_n: int) -> FeatureRowDict:
    """Build a row for a tick before its events are folded in.

    Args:
        tick_n: The tick the row describes.

    Returns:
        A row with no action and every count at zero.
    """
    return FeatureRowDict(
        tick_n=tick_n,
        bot_state="",
        action_kind=NO_ACTION,
        outcome=NO_ACTION,
        duration_ms=-1,
        attempt_id=-1,
        hop_declined=0,
        radar_dispatch=0,
        container_pickup_dispatched=0,
        plan_released=0,
        command_error=0,
        fleet_knowledge_merged=0,
    )


def _tick_of(record: RuntimeEventRecordDict) -> int:
    """Return a record's tick, or -1 when it carries none.

    Args:
        record: The event record.

    Returns:
        The tick number, or ``-1`` for session-level records that
        belong to no tick (``session_room_joined`` is the only kind
        in the corpus without one).
    """
    value = record["fields"].get("tick_n")
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return value


def _int_field(record: RuntimeEventRecordDict, key: str, absent: int) -> int:
    """Read one structured field as an int, or report it absent.

    Args:
        record: Event record.
        key: Field name.
        absent: Value to return when the field is missing or not an
            int, so an absent field stays distinguishable from a real
            zero rather than being imputed.

    Returns:
        The field's value, or ``absent``.
    """
    value = record["fields"].get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return absent
    return value


def _apply_record(row: FeatureRowDict, record: RuntimeEventRecordDict, kind: str) -> None:
    """Fold one event record into its tick's row.

    Args:
        row: The row for the record's tick, mutated in place.
        record: The event record.
        kind: The record's ``diagnostic_kind``.
    """
    state = record["fields"].get("bot_state")
    if isinstance(state, str) and not row["bot_state"]:
        row["bot_state"] = state
    if kind == "action_outcome":
        # The last outcome wins: a tick that resolves more than one
        # attempt is reported by how it ENDED, and attempt_id carries
        # the retry depth that got it there.
        action_kind = record["fields"].get("action_kind")
        outcome = record["fields"].get("outcome")
        row["action_kind"] = action_kind if isinstance(action_kind, str) else NO_ACTION
        row["outcome"] = outcome if isinstance(outcome, str) else NO_ACTION
        row["duration_ms"] = _int_field(record, "duration_ms", row["duration_ms"])
        row["attempt_id"] = _int_field(record, "attempt_id", row["attempt_id"])
        return
    # Literal keys, not a dynamic index. A TypedDict is keyed by
    # literal, so indexing it with a variable needs a suppression
    # comment this codebase does not permit -- the branch is the
    # honest way to say the same thing.
    if kind == "hop_declined":
        row["hop_declined"] += 1
    elif kind == "radar_dispatch":
        row["radar_dispatch"] += 1
    elif kind == "container_pickup_dispatched":
        row["container_pickup_dispatched"] += 1
    elif kind == "plan_released":
        row["plan_released"] += 1
    elif kind == "command_error":
        row["command_error"] += 1
    elif kind == "fleet_knowledge_merged":
        row["fleet_knowledge_merged"] += 1


def build_feature_rows(source_path: Path) -> list[FeatureRowDict]:
    """Reshape one events artifact into a tick-indexed feature table.

    Args:
        source_path: JSONL events artifact to read.

    Returns:
        One row per tick that emitted at least one diagnostic, in
        ascending tick order. Ticks that emitted nothing are absent
        rather than zero-filled -- the artifact does not record them,
        and inventing them would put rows in the table that no tick
        produced.

    Raises:
        JSONTypeError: When the artifact fails strict event decoding.
    """
    rows: dict[int, FeatureRowDict] = {}
    for record in load_event_records(source_path):
        kind = record["fields"].get("diagnostic_kind")
        if not isinstance(kind, str):
            continue
        tick_n = _tick_of(record)
        if tick_n < 0:
            continue
        if tick_n not in rows:
            rows[tick_n] = _blank_row(tick_n)
        _apply_record(rows[tick_n], record, kind)
    return [rows[tick_n] for tick_n in sorted(rows)]


def render_feature_rows(rows: list[FeatureRowDict]) -> str:
    """Render the feature table as a fixed-width report.

    Args:
        rows: Rows to render.

    Returns:
        The report text.
    """
    header = (
        f"{'tick':>6} {'state':22} {'action':10} {'outcome':16} "
        f"{'ms':>6} {'att':>4} {'hopX':>5} {'radar':>5} {'pick':>5} {'plan':>5} {'err':>4}"
    )
    lines = [f"FEATURE ROWS ({len(rows)} ticks)", header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['tick_n']:6} {row['bot_state'][:22]:22} {row['action_kind'][:10]:10} "
            f"{row['outcome'][:16]:16} {row['duration_ms']:6} {row['attempt_id']:4} "
            f"{row['hop_declined']:5} {row['radar_dispatch']:5} "
            f"{row['container_pickup_dispatched']:5} {row['plan_released']:5} "
            f"{row['command_error']:4}"
        )
    return "\n".join(lines)


def feature_rows_body(rows: list[FeatureRowDict]) -> str:
    """Render the table as the JSONL text that gets written.

    Named rather than inlined into :func:`write_feature_rows` because the
    run record's payload digest must be taken over the EXACT content
    written, and a second place rebuilding the same join would be a
    second spelling free to drift from this one.

    Args:
        rows: Rows to render.

    Returns:
        One JSON object per line, newline-terminated.
    """
    body = "\n".join(dump_json_str(encode_feature_row(row)) for row in rows)
    return f"{body}\n"


def write_feature_rows(source_path: Path, rows: list[FeatureRowDict]) -> Path:
    """Write the table beside its source as JSONL, one row per line.

    JSONL rather than one JSON array so a consumer can stream a large
    corpus without holding a whole run in memory, and so rows from many
    runs concatenate by ``cat``.

    Args:
        source_path: The events artifact the rows came from.
        rows: Rows to write.

    Returns:
        The path written.
    """
    destination = source_path.with_suffix("").with_suffix(".features.jsonl")
    _test_hooks.write_text(destination, feature_rows_body(rows))
    return destination


def _build_and_write(source_path: Path) -> list[FeatureRowDict]:
    """Build the table, persist it, and record what produced it.

    The run record is written HERE rather than behind a second command,
    because a provenance step a caller can skip is one that stops being
    run: the table would go back to being untraceable the first time
    somebody exported in a hurry.

    Args:
        source_path: JSONL events artifact to read.

    Returns:
        The rows built.
    """
    rows = build_feature_rows(source_path)
    destination = write_feature_rows(source_path, rows)
    record = feature_run_record(
        source_path,
        rows,
        feature_fingerprint(
            _test_hooks.get_env,
            _test_hooks.get_host_probe(),
            source_path,
            _test_hooks.read_distribution_version,
        ),
        content_digest(feature_rows_body(rows)),
    )
    sidecar = write_run_record(destination, record)
    log.info("Feature rows written: %s (%d ticks)", destination, len(rows))
    log.info("Run record written: %s", sidecar)
    return rows


def main() -> int:
    """Entry point for the ``tankpit-feature-rows`` command.

    Returns:
        Process exit code.
    """
    return run_analyzer_cli(_build_and_write, render_feature_rows, log)


__all__ = [
    "COUNTED_KINDS",
    "NO_ACTION",
    "FeatureRowDict",
    "build_feature_rows",
    "decode_feature_row",
    "encode_feature_row",
    "main",
    "render_feature_rows",
    "write_feature_rows",
]
