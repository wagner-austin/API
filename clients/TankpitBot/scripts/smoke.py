"""Smoke health gate for a 30-second live bot run.

Reads ``runs/bot/latest.events.jsonl`` produced by the most recent
``poetry run tankpit-bot`` session and asserts five concrete signals
that prove the bot reached real gameplay. Exits 0 on success, 1 on
the first failed assertion -- writing the failed assertion plus the
surrounding 10 JSONL events to stderr for triage.

The five assertions (per HANDOFF.md Tier 1 spec):

  1. Login completed -- STATE ladder
     ``INITIALIZING -> WAITING_FOR_POSITION -> IDLE`` in order.
  2. At least one ``map_open`` action cleared via the authoritative
     ``map_data_processed`` signal.
  3. HUNT acquired a non-zero target at least once.
  4. At least one bot action attempted on the wire.
  5. Zero ``stall_timeout`` events in the first 10 seconds of the run.

Architecture notes (per project strictness rules):

  - File reads go through :mod:`scripts._test_hooks` (path_exists,
    read_text). Tests inject fakes via save-and-restore on those
    attributes -- no monkeypatch, no module-constant overrides.
  - JSON parsing goes through :func:`platform_core.json_utils.load_json_str`
    with explicit ``narrow_json_to_dict`` and ``require_*`` validation.
    No ``json.loads``, no ``dict[str, object]``.
  - Each assertion returns ``SmokeFailureDict | None`` instead of
    raising; the run() loop short-circuits on the first non-None.
    No ``try/except`` in core logic.
  - CLI output is written directly to ``sys.stdout`` / ``sys.stderr``
    (this is a CLI script, not a service module).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    optional_int,
    optional_str,
    require_str,
)

from scripts import _test_hooks

#: Default JSONL path scanned by the CLI. Override by passing ``path``
#: to :func:`run`. Tests pass their own temp paths directly.
LATEST_EVENTS_PATH = Path("runs/bot/latest.events.jsonl")

#: STATE-channel ladder that every successful login produces.
LOGIN_LADDER = (
    "INITIALIZING -> WAITING_FOR_POSITION",
    "WAITING_FOR_POSITION -> IDLE",
)

#: All bot-action kinds the WIRE channel may emit. Assertion 4 passes
#: as soon as any of these surface.
ACTION_KINDS = frozenset(
    {
        "shoot",
        "teleport",
        "move",
        "pickup_fuel",
        "pickup_equipment",
        "radar",
        "map_open",
    }
)

# Reserved record-level keys stripped from a record's ``fields`` view
# so structured filtering does not collide with them. Matches the
# reserved-key set enforced at write time in ``runtime_logging``.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "timestamp",
        "level",
        "logger",
        "mode",
        "channel",
        "message",
    }
)


class SmokeFailureDict(TypedDict):
    """Result of a failed smoke assertion.

    Each ``assert_*`` function returns one of these instead of raising
    so the run() loop can report context without ``try/except``.

    Attributes:
        message: One-line description of the failed assertion. The CLI
            prints this verbatim under ``SMOKE FAILED:``.
        pivot: Index into the ``records`` list nearest the failure;
            :func:`context_window` shows a window around this index.
    """

    message: str
    pivot: int


class SmokeRecord:
    """One parsed line from ``latest.events.jsonl``.

    Attributes:
        line_no: 1-based source-file line number.
        raw: Original raw JSONL line, used for context dumps on failure.
        channel: Event channel (``STATE``, ``WIRE``, ``WIRE_COMPLETE``,
            ``AI``, etc.).
        message: Human-readable message body.
        timestamp: ISO timestamp string written by the runtime logger.
        fields: Structured-field view with reserved record-level keys
            stripped out.
    """

    __slots__ = ("channel", "fields", "line_no", "message", "raw", "timestamp")

    def __init__(
        self,
        *,
        line_no: int,
        raw: str,
        channel: str,
        message: str,
        timestamp: str,
        fields: JSONObject,
    ) -> None:
        """Initialise from already-validated fields.

        Args:
            line_no: 1-based source-file line number.
            raw: Original raw JSONL line.
            channel: Event channel string.
            message: Event message string.
            timestamp: Event timestamp string.
            fields: Structured fields with reserved record-level keys
                already stripped.
        """
        self.line_no = line_no
        self.raw = raw.rstrip()
        self.channel = channel
        self.message = message
        self.timestamp = timestamp
        self.fields = fields


def _strip_reserved_keys(parsed: JSONObject) -> JSONObject:
    """Return a copy of ``parsed`` with reserved record-level keys removed.

    Args:
        parsed: Parsed record object.

    Returns:
        New dict containing only structured-field entries.
    """
    return {k: v for k, v in parsed.items() if k not in _RESERVED_RECORD_KEYS}


def _decode_smoke_record(line_no: int, raw: str, parsed: JSONObject) -> SmokeRecord:
    """Validate and decode a JSONObject into a :class:`SmokeRecord`.

    Args:
        line_no: 1-based source-file line number for diagnostics.
        raw: Original raw JSONL line.
        parsed: Parsed record object.

    Returns:
        Validated SmokeRecord.

    Raises:
        JSONTypeError: If a required field is missing or has the
            wrong type. The error is propagated unchanged so the
            CLI surfaces it with full context.
    """
    return SmokeRecord(
        line_no=line_no,
        raw=raw,
        channel=require_str(parsed, "channel"),
        message=require_str(parsed, "message"),
        timestamp=require_str(parsed, "timestamp"),
        fields=_strip_reserved_keys(parsed),
    )


class _MissingEventsFileError(SystemExit):
    """Raised by :func:`load_records` when the JSONL file is missing.

    A dedicated subclass keeps the CLI's exit-code path testable
    without catching the bare ``SystemExit`` thrown by ``sys.exit``.
    """

    def __init__(self, path: Path) -> None:
        """Initialise with a diagnostic CLI message.

        Args:
            path: Path the loader expected to find.
        """
        super().__init__(1)
        self.path = path
        self.message = (
            f"smoke: {path} does not exist -- did you run `poetry run tankpit-bot` first?"
        )


def load_records(path: Path) -> list[SmokeRecord]:
    """Read every JSONL line into a list of :class:`SmokeRecord`.

    Args:
        path: Path to the events JSONL file.

    Returns:
        Parsed records in file order.

    Raises:
        _MissingEventsFileError: If the file does not exist. The error
            carries the requested path and exits with status 1.
        JSONTypeError: If a record's top-level value is not an object
            or a required field is missing.
    """
    if not _test_hooks.path_exists(path):
        raise _MissingEventsFileError(path)
    text = _test_hooks.read_text(path)
    records: list[SmokeRecord] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        parsed_value: JSONValue = load_json_str(raw)
        parsed = narrow_json_to_dict(parsed_value)
        records.append(_decode_smoke_record(line_no=line_no, raw=raw, parsed=parsed))
    return records


def parse_iso_timestamp_seconds(ts: str) -> float:
    """Return total elapsed seconds-of-day from an ISO-ish timestamp.

    The bot writes timestamps like ``2026-06-20T15:01:55``; we only
    need ordering and relative offsets within a run, so we compute
    ``H*3600 + M*60 + S`` from the time-of-day portion.

    Args:
        ts: ISO timestamp string (must contain ``T`` and a seconds
            component).

    Returns:
        Seconds-of-day as a float.

    Raises:
        ValueError: If ``ts`` is missing the ``T`` separator or the
            seconds component.
    """
    if "T" not in ts:
        raise ValueError(f"smoke: timestamp '{ts}' missing 'T' separator")
    time_part = ts.split("T", 1)[1]
    pieces = time_part.split(":")
    if len(pieces) < 3:
        raise ValueError(f"smoke: timestamp '{ts}' missing seconds component")
    hours = int(pieces[0])
    minutes = int(pieces[1])
    seconds = float(pieces[2])
    return hours * 3600 + minutes * 60 + seconds


def context_window(records: list[SmokeRecord], pivot: int, radius: int = 5) -> str:
    """Return the +/- ``radius`` records around the pivot index as text.

    Args:
        records: All records.
        pivot: Index of the record to centre the window on.
        radius: Number of records to show on each side. Defaults to 5,
            giving an 11-line window when records permit.

    Returns:
        The selected records' raw lines joined by newlines.
    """
    start = max(0, pivot - radius)
    end = min(len(records), pivot + radius + 1)
    return "\n".join(records[i].raw for i in range(start, end))


def assert_login_completed(records: list[SmokeRecord]) -> SmokeFailureDict | None:
    """Verify the STATE ladder records both login transitions in order.

    Args:
        records: All loaded events in file order.

    Returns:
        ``None`` when both transitions appear in order, otherwise a
        :class:`SmokeFailureDict` pointing at the last STATE event seen.
    """
    seen_index = 0
    last_state_index = -1
    for index, rec in enumerate(records):
        if rec.channel != "STATE":
            continue
        last_state_index = index
        target = LOGIN_LADDER[seen_index]
        if rec.message == target:
            seen_index += 1
            if seen_index == len(LOGIN_LADDER):
                return None
    return SmokeFailureDict(
        message="1) login ladder INITIALIZING -> WAITING_FOR_POSITION -> IDLE not observed",
        pivot=last_state_index if last_state_index >= 0 else 0,
    )


def assert_map_open_cleared_via_map_data(
    records: list[SmokeRecord],
) -> SmokeFailureDict | None:
    """Verify at least one map_open cleared via ``map_data_processed``.

    Args:
        records: All loaded events in file order.

    Returns:
        ``None`` on success, otherwise a :class:`SmokeFailureDict`.
    """
    for rec in records:
        if rec.channel != "WIRE_COMPLETE":
            continue
        if optional_str(rec.fields, "action_kind") != "map_open":
            continue
        if optional_str(rec.fields, "signal") == "map_data_processed":
            return None
    return SmokeFailureDict(
        message="2) no map_open action cleared via signal=map_data_processed",
        pivot=0,
    )


def assert_hunt_scored_target(records: list[SmokeRecord]) -> SmokeFailureDict | None:
    """Verify at least one HUNT score event has a non-zero target.

    Args:
        records: All loaded events in file order.

    Returns:
        ``None`` on success, otherwise a :class:`SmokeFailureDict`
        pointing at the last HUNT score event seen.
    """
    last_hunt_index = -1
    for index, rec in enumerate(records):
        if rec.channel != "AI":
            continue
        if not rec.message.startswith("HUNT score="):
            continue
        last_hunt_index = index
        target_x = optional_int(rec.fields, "combat_target_x")
        target_y = optional_int(rec.fields, "combat_target_y")
        if (target_x is not None and target_x != 0) or (target_y is not None and target_y != 0):
            return None
    return SmokeFailureDict(
        message=("3) HUNT never scored a non-zero target (combat_target_x/y both 0 every tick)"),
        pivot=last_hunt_index if last_hunt_index >= 0 else 0,
    )


def assert_action_attempted(records: list[SmokeRecord]) -> SmokeFailureDict | None:
    """Verify at least one WIRE event carries a known ``action_kind``.

    Args:
        records: All loaded events in file order.

    Returns:
        ``None`` on success, otherwise a :class:`SmokeFailureDict`
        pointing at the last WIRE event seen.
    """
    last_wire_index = -1
    for index, rec in enumerate(records):
        if rec.channel != "WIRE":
            continue
        last_wire_index = index
        kind = optional_str(rec.fields, "action_kind")
        if kind is not None and kind in ACTION_KINDS:
            return None
    return SmokeFailureDict(
        message="4) no bot action attempted (no WIRE event with a known action_kind)",
        pivot=last_wire_index if last_wire_index >= 0 else 0,
    )


def assert_no_early_stall(records: list[SmokeRecord]) -> SmokeFailureDict | None:
    """Verify no ``stall_timeout`` signal fires in the first 10 seconds.

    Args:
        records: All loaded events in file order.

    Returns:
        ``None`` on success, otherwise a :class:`SmokeFailureDict`.

    Raises:
        ValueError: If a record carries an unparseable timestamp string.
    """
    if not records:
        return SmokeFailureDict(
            message="5) no events at all -- session produced an empty JSONL",
            pivot=0,
        )
    start_seconds = parse_iso_timestamp_seconds(records[0].timestamp)
    for index, rec in enumerate(records):
        if rec.channel != "WIRE_COMPLETE":
            continue
        if optional_str(rec.fields, "signal") != "stall_timeout":
            continue
        elapsed = parse_iso_timestamp_seconds(rec.timestamp) - start_seconds
        if elapsed <= 10.0:
            action_kind = optional_str(rec.fields, "action_kind")
            return SmokeFailureDict(
                message=(
                    f"5) stall_timeout fired at t+{elapsed:.1f}s "
                    f"(action_kind={action_kind!r}) within the 10s window"
                ),
                pivot=index,
            )
    return None


# Tuple of every assertion in the order :func:`run` evaluates them.
# Exposed at module scope so tests can iterate it.
SmokeAssertion = Callable[[list[SmokeRecord]], SmokeFailureDict | None]
SMOKE_ASSERTIONS: tuple[SmokeAssertion, ...] = (
    assert_login_completed,
    assert_map_open_cleared_via_map_data,
    assert_hunt_scored_target,
    assert_action_attempted,
    assert_no_early_stall,
)


def _write_failure(records: list[SmokeRecord], failure: SmokeFailureDict) -> None:
    """Emit the failure summary and surrounding context to stderr.

    Args:
        records: All loaded events (for context window).
        failure: The first failed assertion's result.
    """
    sys.stderr.write(f"SMOKE FAILED: {failure['message']}\n")
    sys.stderr.write("Surrounding context:\n")
    sys.stderr.write(context_window(records, failure["pivot"]))
    sys.stderr.write("\n")


def _write_success(records: list[SmokeRecord]) -> None:
    """Emit the success summary to stdout.

    Args:
        records: All loaded events (for the event count).
    """
    sys.stdout.write(
        f"SMOKE PASSED: {len(records)} events, "
        "5/5 assertions green (login, map_data_processed, HUNT target, "
        "action attempted, no early stall).\n"
    )


def evaluate(records: list[SmokeRecord]) -> SmokeFailureDict | None:
    """Run every assertion and return the first failure, or ``None``.

    Args:
        records: All loaded events in file order.

    Returns:
        First non-``None`` assertion result, or ``None`` when every
        assertion passes.
    """
    for check in SMOKE_ASSERTIONS:
        failure = check(records)
        if failure is not None:
            return failure
    return None


def run(path: Path = LATEST_EVENTS_PATH) -> int:
    """Load ``path``, evaluate all assertions, and return an exit code.

    Args:
        path: JSONL events file to read. Defaults to
            :data:`LATEST_EVENTS_PATH`.

    Returns:
        ``0`` when every assertion passes, ``1`` when an assertion fails
        or the events file is missing.
    """
    records = load_records(path)
    failure = evaluate(records)
    if failure is not None:
        _write_failure(records, failure)
        return 1
    _write_success(records)
    return 0


def main() -> None:
    """Entry point for the ``tankpit-smoke`` console script."""
    sys.exit(run())


if __name__ == "__main__":
    main()
