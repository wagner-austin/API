"""Centralized runtime logging and structured event emission.

This module owns the canonical run log/event streams for bot and sniffer
executions. CLI entry points configure one runtime mode per process, and
high-signal subsystems emit structured AI/SYNC/STATE/WIRE/WORLD events
through the helpers defined here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str, require_str
from platform_core.logging import get_logger, setup_rich_logging, stdlib_logging
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_artifacts import (
    BotRunArtifactsDict,
    ProbeRunArtifactsDict,
    SniffRunArtifactsDict,
    build_bot_run_artifacts,
    build_probe_run_artifacts,
    build_sniff_run_artifacts,
    make_run_stamp,
)

_EMITTER_LOGGER = get_logger("tankpit_bot.runtime.events")
_ARTIFACT_HANDLER_NAME_PREFIX = "tankpit_bot.runtime.artifacts."

_BOT_ARTIFACTS: BotRunArtifactsDict | None = None
_SNIFF_ARTIFACTS: SniffRunArtifactsDict | None = None
_PROBE_ARTIFACTS: ProbeRunArtifactsDict | None = None


_RESERVED_EVENT_KEYS: frozenset[str] = frozenset(
    {"timestamp", "level", "logger", "mode", "channel", "message"}
)

#: Context-field key names auto-attached to every emit_* event when
#: present in ``_RUNTIME_CONTEXT``. Documented separately so consumers
#: can introspect what to expect from JSONL queries.
RUNTIME_CONTEXT_KEYS: frozenset[str] = frozenset({"tick_n", "bot_state", "in_flight_action_kind"})


class RuntimeContextDict(TypedDict, total=False):
    """Per-tick context auto-attached to every emit_* event.

    Each field is optional; absent fields are omitted from the JSONL
    record. The tick loop sets these once per tick so every event
    emitted that tick carries the same context. Explicit fields passed
    to an emit_* call override the context (last-write-wins).

    Attributes:
        tick_n: 1-based index of the currently-executing tick. Use 0
            when no tick is active (boot, login, shutdown). Always
            attached when set, even if the value is 0.
        bot_state: ``"<mode>/<mode_state>"`` snapshot of the durable
            AI mode and its inner state. Empty string when none.
        in_flight_action_kind: ``ActionKind`` literal of the bot's
            current in-flight action, or ``"none"`` when idle.
    """

    tick_n: int
    bot_state: str
    in_flight_action_kind: str


# Internal storage for the active context, split into one typed slot per
# field so each value is mypy-narrowed at its source. The public
# :class:`RuntimeContextDict` view is assembled by :func:`get_runtime_context`.
_RUNTIME_CONTEXT_TICK_N: int | None = None
_RUNTIME_CONTEXT_BOT_STATE: str | None = None
_RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND: str | None = None


def set_runtime_context(
    *,
    tick_n: int | None = None,
    bot_state: str | None = None,
    in_flight_action_kind: str | None = None,
) -> None:
    """Set or update the active per-tick runtime context.

    Each subsequent ``emit_*`` call attaches the present context fields
    to its structured payload (under the field names ``tick_n``,
    ``bot_state``, ``in_flight_action_kind``). Pass ``None`` to leave a
    previous value unchanged; use :func:`clear_runtime_context` to
    remove every value.

    Args:
        tick_n: 1-based current tick index, or ``None`` to keep the
            previous value.
        bot_state: ``"<mode>/<mode_state>"`` snapshot, or ``None`` to
            keep the previous value.
        in_flight_action_kind: ``ActionKind`` string, or ``None`` to
            keep the previous value.
    """
    global _RUNTIME_CONTEXT_TICK_N
    global _RUNTIME_CONTEXT_BOT_STATE
    global _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    if tick_n is not None:
        _RUNTIME_CONTEXT_TICK_N = tick_n
    if bot_state is not None:
        _RUNTIME_CONTEXT_BOT_STATE = bot_state
    if in_flight_action_kind is not None:
        _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND = in_flight_action_kind


def clear_runtime_context() -> None:
    """Remove every field from the active runtime context.

    Subsequent ``emit_*`` calls emit without context until
    :func:`set_runtime_context` is called again. The tick loop's
    teardown path calls this so test/probe sessions start clean.
    """
    global _RUNTIME_CONTEXT_TICK_N
    global _RUNTIME_CONTEXT_BOT_STATE
    global _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    _RUNTIME_CONTEXT_TICK_N = None
    _RUNTIME_CONTEXT_BOT_STATE = None
    _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND = None


def get_runtime_context() -> RuntimeContextDict:
    """Return a typed defensive copy of the current runtime context.

    Returns:
        A typed snapshot of the active context. Callers may mutate the
        returned dict without affecting the module-level state.
    """
    snapshot: RuntimeContextDict = {}
    if _RUNTIME_CONTEXT_TICK_N is not None:
        snapshot["tick_n"] = _RUNTIME_CONTEXT_TICK_N
    if _RUNTIME_CONTEXT_BOT_STATE is not None:
        snapshot["bot_state"] = _RUNTIME_CONTEXT_BOT_STATE
    if _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND is not None:
        snapshot["in_flight_action_kind"] = _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    return snapshot


class RuntimeEventRecordDict(TypedDict):
    """Structured runtime event record.

    Attributes:
        timestamp: Local wall-clock timestamp for the emitted log record.
        level: Logging level name.
        logger: Logger name that emitted the event.
        mode: Runtime mode, either ``bot`` or ``sniff``.
        channel: High-level event channel such as ``AI`` or ``WORLD``.
        message: Human-readable event text without the channel prefix.
        fields: Structured key/value payload spread into the JSONL record at
            the top level (no nesting). Reserved keys (timestamp, level,
            logger, mode, channel, message) may not appear here -- collision
            is rejected at encode time so queries against the JSONL never
            see ambiguous data.
    """

    timestamp: str
    level: str
    logger: str
    mode: str
    channel: str
    message: str
    fields: dict[str, str | int | float | bool]


class RuntimeLogExtraDict(TypedDict):
    """Structured extra fields carried on high-signal runtime log records."""

    runtime_channel: str
    runtime_message: str
    runtime_mode: str
    runtime_fields: dict[str, str | int | float | bool]


class _RuntimeRecordMapping(Protocol):
    """Minimal typed access to runtime-specific LogRecord extras."""

    def __contains__(self, key: str) -> bool:
        """Return True when a runtime extra exists."""
        ...

    def __getitem__(
        self, key: str
    ) -> str | int | float | bool | dict[str, str | int | float | bool] | None:
        """Return a runtime extra value."""
        ...

    def get(
        self,
        key: str,
        default: None = None,
    ) -> str | int | float | bool | dict[str, str | int | float | bool] | None:
        """Return a runtime extra value or the default when absent."""
        ...


def encode_runtime_event_record(record: RuntimeEventRecordDict) -> JSONObject:
    """Encode a runtime event record to JSON-compatible data.

    Structured ``fields`` are spread into the top-level JSON object so
    consumers can query them directly (``jq '.duration_ms'``). A field
    whose name collides with one of the reserved record keys raises --
    silent overwriting of ``timestamp``/``channel``/``message`` would
    produce ambiguous traces.

    Args:
        record: Runtime event record.

    Returns:
        JSON-compatible representation.

    Raises:
        ValueError: When a structured field name collides with a
            reserved top-level event key.
    """
    encoded: JSONObject = {
        "timestamp": record["timestamp"],
        "level": record["level"],
        "logger": record["logger"],
        "mode": record["mode"],
        "channel": record["channel"],
        "message": record["message"],
    }
    for key, value in record["fields"].items():
        if key in _RESERVED_EVENT_KEYS:
            raise ValueError(f"runtime event field name {key!r} collides with reserved record key")
        encoded[key] = value
    return encoded


def decode_runtime_event_record(data: JSONObject) -> RuntimeEventRecordDict:
    """Decode a runtime event record from JSON-compatible data.

    Reverse-spreads structured fields: any top-level key that is not in
    :data:`_RESERVED_EVENT_KEYS` is collected into ``fields``.

    Args:
        data: JSON object to decode.

    Returns:
        Validated runtime event record.
    """
    fields: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        if key in _RESERVED_EVENT_KEYS:
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise JSONTypeError(
                f"runtime event field {key!r} has non-primitive type {type(value).__name__}"
            )
        fields[key] = value
    return RuntimeEventRecordDict(
        timestamp=require_str(data, "timestamp"),
        level=require_str(data, "level"),
        logger=require_str(data, "logger"),
        mode=require_str(data, "mode"),
        channel=require_str(data, "channel"),
        message=require_str(data, "message"),
        fields=fields,
    )


class _HookTextArtifactHandler(stdlib_logging.Handler):
    """Logging handler that mirrors formatted log lines to artifact files."""

    def __init__(self, paths: tuple[Path, Path]) -> None:
        """Initialize the handler.

        Args:
            paths: ``(latest_path, archive_path)`` text log destinations.
        """
        super().__init__()
        self._latest_path = paths[0]
        self._archive_path = paths[1]

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        """Append the formatted log line to both text log files.

        Args:
            record: Log record to persist.
        """
        line = self.format(record) + "\n"
        _test_hooks.append_text(self._latest_path, line)
        _test_hooks.append_text(self._archive_path, line)


class _HookEventArtifactHandler(stdlib_logging.Handler):
    """Logging handler that writes structured high-signal runtime events."""

    def __init__(self, mode: str, paths: tuple[Path, Path]) -> None:
        """Initialize the handler.

        Args:
            mode: Runtime mode, either ``bot`` or ``sniff``.
            paths: ``(latest_path, archive_path)`` JSONL event destinations.
        """
        super().__init__()
        self._mode = mode
        self._latest_path = paths[0]
        self._archive_path = paths[1]

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        """Append a structured event when the record carries runtime metadata.

        Args:
            record: Log record to inspect and possibly persist.
        """
        record_dict: _RuntimeRecordMapping = record.__dict__
        if "runtime_channel" not in record_dict or "runtime_message" not in record_dict:
            return
        channel_raw = record_dict["runtime_channel"]
        message_raw = record_dict["runtime_message"]
        if not isinstance(channel_raw, str) or not isinstance(message_raw, str):
            return
        fields_raw = record_dict.get("runtime_fields")
        if not isinstance(fields_raw, dict):
            return
        fields: dict[str, str | int | float | bool] = {}
        bad = False
        for raw_key, raw_value in fields_raw.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, (str, int, float, bool)):
                bad = True
                break
            fields[raw_key] = raw_value
        if bad:
            return
        event = RuntimeEventRecordDict(
            timestamp=datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S"),
            level=record.levelname,
            logger=record.name,
            mode=self._mode,
            channel=channel_raw,
            message=message_raw,
            fields=fields,
        )
        line = dump_json_str(encode_runtime_event_record(event), compact=True) + "\n"
        _test_hooks.append_text(self._latest_path, line)
        _test_hooks.append_text(self._archive_path, line)


def configure_bot_runtime_logging(stamp: str | None = None) -> BotRunArtifactsDict:
    """Configure console logging plus canonical bot artifact outputs.

    Args:
        stamp: Optional archive timestamp stamp for deterministic tests.

    Returns:
        Configured bot runtime artifacts.
    """
    global _BOT_ARTIFACTS, _SNIFF_ARTIFACTS, _PROBE_ARTIFACTS
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = build_bot_run_artifacts(resolved_stamp)
    setup_rich_logging(level="INFO")
    _reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _install_artifact_handlers(
        "bot",
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _BOT_ARTIFACTS = artifacts
    _SNIFF_ARTIFACTS = None
    _PROBE_ARTIFACTS = None
    return artifacts


def configure_sniff_runtime_logging(stamp: str | None = None) -> SniffRunArtifactsDict:
    """Configure console logging plus canonical sniffer artifact outputs.

    Args:
        stamp: Optional archive timestamp stamp for deterministic tests.

    Returns:
        Configured sniffer runtime artifacts.
    """
    global _BOT_ARTIFACTS, _SNIFF_ARTIFACTS, _PROBE_ARTIFACTS
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = build_sniff_run_artifacts(resolved_stamp)
    setup_rich_logging(level="INFO")
    _reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
        Path(artifacts["latest_capture_path"]),
        Path(artifacts["latest_raw_capture_path"]),
        Path(artifacts["latest_summary_path"]),
    )
    _install_artifact_handlers(
        "sniff",
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _BOT_ARTIFACTS = None
    _SNIFF_ARTIFACTS = artifacts
    _PROBE_ARTIFACTS = None
    return artifacts


def configure_probe_runtime_logging(
    probe_name: str,
    stamp: str | None = None,
) -> ProbeRunArtifactsDict:
    """Configure console logging plus canonical probe artifact outputs.

    Args:
        probe_name: Probe identifier (``fuel``, ``equipment``, ``movement``,
            ``teleport``, ``enemy_teleport``, ``fuel_drill``). Embedded in
            archive filenames so multiple probe kinds share
            ``runs/probe/``.
        stamp: Optional archive timestamp stamp for deterministic tests.

    Returns:
        Configured probe runtime artifacts.

    Raises:
        ValueError: When ``probe_name`` is empty (validated by
            :func:`tankpit_bot.runtime_artifacts.build_probe_run_artifacts`).
    """
    global _BOT_ARTIFACTS, _SNIFF_ARTIFACTS, _PROBE_ARTIFACTS
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = build_probe_run_artifacts(probe_name, resolved_stamp)
    setup_rich_logging(level="INFO")
    _reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _install_artifact_handlers(
        f"probe:{probe_name}",
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _BOT_ARTIFACTS = None
    _SNIFF_ARTIFACTS = None
    _PROBE_ARTIFACTS = artifacts
    return artifacts


def get_bot_runtime_artifacts() -> BotRunArtifactsDict | None:
    """Return the active bot runtime artifacts for this process, if configured."""
    return _BOT_ARTIFACTS


def get_sniff_runtime_artifacts() -> SniffRunArtifactsDict | None:
    """Return the active sniffer runtime artifacts for this process, if configured."""
    return _SNIFF_ARTIFACTS


def get_probe_runtime_artifacts() -> ProbeRunArtifactsDict | None:
    """Return the active probe runtime artifacts for this process, if configured."""
    return _PROBE_ARTIFACTS


def emit_ai(
    message: str,
    *args: str | int | float | bool,
    **fields: str | int | float | bool,
) -> None:
    """Emit a structured AI event.

    Args:
        message: ``printf``-style message string.
        *args: Format arguments for the message string.
        **fields: Optional structured key/value payload spread into the
            JSONL event at the top level. Use for fields that tooling
            should query (e.g. ``combat_target_x``) so the smoke gate
            and ``bot-query`` reach them without parsing the message.
    """
    _emit_runtime_event("AI", message, *args, **fields)


def emit_sync(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured synchronization event."""
    _emit_runtime_event("SYNC", message, *args)


def emit_state(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured state-machine event."""
    _emit_runtime_event("STATE", message, *args)


def emit_wire(
    message: str,
    *args: str | int | float | bool,
    **fields: str | int | float | bool,
) -> None:
    """Emit a structured wire/protocol command event.

    Args:
        message: ``printf``-style message string.
        *args: Format arguments for the message string.
        **fields: Optional structured payload (e.g. ``action_kind``,
            ``target_x``) spread into the JSONL event at the top
            level so smoke + ``bot-query`` can reach them without
            parsing the message text.
    """
    _emit_runtime_event("WIRE", message, *args, **fields)


def emit_world(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured world-state event."""
    _emit_runtime_event("WORLD", message, *args)


def require_int_field(
    fields: dict[str, str | int | float | bool],
    key: str,
) -> int:
    """Extract a required int-valued structured field.

    Mirrors ``platform_core.json_utils.require_int`` for the
    :data:`RuntimeEventRecordDict.fields` payload, whose value type is
    the narrow primitive union ``str | int | float | bool``. Booleans
    are rejected so callers reading ``duration_ms`` / ``timeout_ms`` /
    coordinate fields are guaranteed a numeric int.

    Args:
        fields: Decoded structured payload from a runtime event record.
        key: Field name to extract.

    Returns:
        Validated int value.

    Raises:
        KeyError: When ``key`` is absent from ``fields``.
        TypeError: When the field is not a numeric int (bools are
            rejected even though Python treats ``bool`` as ``int``).
    """
    if key not in fields:
        raise KeyError(f"runtime field {key!r} is required")
    value = fields[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"runtime field {key!r} must be int, got {type(value).__name__}")
    return value


def require_str_field(
    fields: dict[str, str | int | float | bool],
    key: str,
) -> str:
    """Extract a required str-valued structured field.

    Args:
        fields: Decoded structured payload from a runtime event record.
        key: Field name to extract.

    Returns:
        Validated str value.

    Raises:
        KeyError: When ``key`` is absent from ``fields``.
        TypeError: When the field is not a str.
    """
    if key not in fields:
        raise KeyError(f"runtime field {key!r} is required")
    value = fields[key]
    if not isinstance(value, str):
        raise TypeError(f"runtime field {key!r} must be str, got {type(value).__name__}")
    return value


def require_bool_field(
    fields: dict[str, str | int | float | bool],
    key: str,
) -> bool:
    """Extract a required bool-valued structured field.

    Args:
        fields: Decoded structured payload from a runtime event record.
        key: Field name to extract.

    Returns:
        Validated bool value.

    Raises:
        KeyError: When ``key`` is absent from ``fields``.
        TypeError: When the field is not a bool.
    """
    if key not in fields:
        raise KeyError(f"runtime field {key!r} is required")
    value = fields[key]
    if not isinstance(value, bool):
        raise TypeError(f"runtime field {key!r} must be bool, got {type(value).__name__}")
    return value


def emit_diagnostic(
    *,
    diagnostic_kind: str,
    **fields: str | int | float | bool,
) -> None:
    """Emit a structured diagnostic event.

    The ``DIAGNOSTIC`` channel carries observability-only emissions:
    target-selection breakdowns, attempt-window message timelines,
    invariant-violation reports, and the like. Every emit lands on
    ``runs/<mode>/latest.events.jsonl`` with ``diagnostic_kind`` plus
    any caller-supplied primitive fields spread at the top level, so
    queries against the JSONL can filter by kind and compare timing
    distributions across runs.

    Args:
        diagnostic_kind: Stable identifier for the diagnostic shape
            (``teleport_attempt``, ``fuel_target_selection``,
            ``action_phase_overlap``, ``map_positions_parsed``,
            ``movement_probe_map_already_showing``,
            ``command_dispatch_failure``, etc.). The kind names the
            payload schema; callers passing structured fields must
            match the kind's documented schema.
        **fields: Caller-supplied structured fields. Each field name
            must not collide with the reserved top-level event keys
            (timestamp, level, logger, mode, channel, message) --
            collision raises at encode time.
    """
    message = f"diagnostic_kind={diagnostic_kind}"
    _emit_runtime_event(
        "DIAGNOSTIC",
        message,
        diagnostic_kind=diagnostic_kind,
        **fields,
    )


def emit_wire_complete(
    *,
    action_kind: str,
    duration_ms: int,
    signal: str,
    **extra: str | int | float | bool,
) -> None:
    """Emit a structured completion event for a dispatched bot action.

    Symmetric to :func:`emit_wire`: where ``WIRE`` records the moment the
    bot dispatched a command, ``WIRE_COMPLETE`` records the moment the
    HFSM observed the authoritative completion signal for that command.
    The resulting JSONL line carries ``action_kind`` / ``duration_ms`` /
    ``signal`` at the top level so consumers can run queries like
    ``jq 'select(.channel=="WIRE_COMPLETE" and .action_kind=="map_open")
    | .duration_ms'`` directly against ``runs/bot/latest.events.jsonl``.

    Args:
        action_kind: Kind of action that completed (e.g. ``map_open``,
            ``move``, ``teleport``, ``collect``, ``scan``).
        duration_ms: Wall-clock milliseconds between dispatch and the
            observed completion. Negative values mean the gate fired
            with no recorded ``started_ms`` and are passed through
            verbatim so reviewers can spot the case.
        signal: Name of the authoritative completion signal -- e.g.
            ``map_data_processed``, ``teleport_landed``,
            ``radar_scan_complete``, ``position_reached``,
            ``container_consumed_or_reached``, ``stall_timeout``.
        **extra: Additional structured fields to attach (e.g. target
            coordinates). Field names must not collide with the reserved
            top-level event keys -- collision raises at encode time.
    """
    message = f"{action_kind} completed in {duration_ms}ms via {signal}"
    _emit_runtime_event(
        "WIRE_COMPLETE",
        message,
        action_kind=action_kind,
        duration_ms=duration_ms,
        signal=signal,
        **extra,
    )


def _reset_artifact_files(*paths: Path) -> None:
    """Clear artifact files at process startup.

    Args:
        paths: Artifact paths to reset to empty content.
    """
    for path in paths:
        _test_hooks.write_text(path, "")


def _install_artifact_handlers(
    mode: str,
    latest_log_path: Path,
    archive_log_path: Path,
    latest_events_path: Path,
    archive_events_path: Path,
) -> None:
    """Attach artifact mirroring handlers to the root logger.

    Args:
        mode: Runtime mode, either ``bot`` or ``sniff``.
        latest_log_path: Stable latest text log path.
        archive_log_path: Timestamped archived text log path.
        latest_events_path: Stable latest JSONL event path.
        archive_events_path: Timestamped archived JSONL event path.
    """
    root = stdlib_logging.getLogger()
    _remove_artifact_handlers(root)
    text_handler = _HookTextArtifactHandler((latest_log_path, archive_log_path))
    text_handler.set_name(_ARTIFACT_HANDLER_NAME_PREFIX + "text")
    text_handler.setLevel(root.level)
    text_handler.setFormatter(
        stdlib_logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%m/%d/%y %H:%M:%S")
    )
    root.addHandler(text_handler)

    event_handler = _HookEventArtifactHandler(mode, (latest_events_path, archive_events_path))
    event_handler.set_name(_ARTIFACT_HANDLER_NAME_PREFIX + "events")
    event_handler.setLevel(root.level)
    root.addHandler(event_handler)


def _remove_artifact_handlers(root: stdlib_logging.Logger) -> None:
    """Remove previously installed runtime artifact handlers from the root logger.

    Args:
        root: Root logger to clean before installing fresh handlers.
    """
    handlers_to_keep: list[stdlib_logging.Handler] = []
    for handler in root.handlers:
        name = handler.get_name() or ""
        if not name.startswith(_ARTIFACT_HANDLER_NAME_PREFIX):
            handlers_to_keep.append(handler)
    root.handlers = handlers_to_keep


def _merge_context_into_fields(
    fields: dict[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Return a new field dict with the active runtime context attached.

    Context fields are written first; explicit ``fields`` win on
    collision so call-site arguments override the per-tick context.

    Args:
        fields: Explicit fields passed to the emit_* call.

    Returns:
        New dict containing the context fields (when set) plus the
        original ``fields`` overrides.
    """
    merged: dict[str, str | int | float | bool] = {}
    if _RUNTIME_CONTEXT_TICK_N is not None:
        merged["tick_n"] = _RUNTIME_CONTEXT_TICK_N
    if _RUNTIME_CONTEXT_BOT_STATE is not None:
        merged["bot_state"] = _RUNTIME_CONTEXT_BOT_STATE
    if _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND is not None:
        merged["in_flight_action_kind"] = _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    merged.update(fields)
    return merged


def _emit_runtime_event(
    channel: str,
    message: str,
    *args: str | int | float | bool,
    **fields: str | int | float | bool,
) -> None:
    """Emit a runtime event to both console logs and JSONL artifacts.

    The active runtime context (``tick_n`` / ``bot_state`` /
    ``in_flight_action_kind`` set via :func:`set_runtime_context`) is
    merged into the structured payload before write. Explicit
    ``fields`` override the context fields on collision.

    Args:
        channel: Event channel such as ``AI`` or ``WORLD``.
        message: ``printf``-style message string without the channel prefix.
        *args: Format arguments for the message string.
        **fields: Structured key/value payload spread into the JSONL event
            at the top level. Must not collide with the reserved event keys
            (timestamp, level, logger, mode, channel, message).
    """
    formatted = message % args if args else message
    extra = RuntimeLogExtraDict(
        runtime_channel=channel,
        runtime_message=formatted,
        runtime_mode=_runtime_mode_name(),
        runtime_fields=_merge_context_into_fields(dict(fields)),
    )
    _EMITTER_LOGGER.info("%s: %s", channel, formatted, extra=extra)


def _runtime_mode_name() -> str:
    """Return the active runtime mode name.

    Returns:
        ``bot``, ``sniff``, or ``probe:<name>`` when configured,
        otherwise ``unconfigured``.
    """
    if _BOT_ARTIFACTS is not None:
        return "bot"
    if _SNIFF_ARTIFACTS is not None:
        return "sniff"
    if _PROBE_ARTIFACTS is not None:
        return f"probe:{_PROBE_ARTIFACTS['probe_name']}"
    return "unconfigured"


__all__ = [
    "RUNTIME_CONTEXT_KEYS",
    "RuntimeContextDict",
    "RuntimeEventRecordDict",
    "clear_runtime_context",
    "configure_bot_runtime_logging",
    "configure_probe_runtime_logging",
    "configure_sniff_runtime_logging",
    "decode_runtime_event_record",
    "emit_ai",
    "emit_diagnostic",
    "emit_state",
    "emit_sync",
    "emit_wire",
    "emit_wire_complete",
    "emit_world",
    "encode_runtime_event_record",
    "get_bot_runtime_artifacts",
    "get_probe_runtime_artifacts",
    "get_runtime_context",
    "get_sniff_runtime_artifacts",
    "require_bool_field",
    "require_int_field",
    "require_str_field",
    "set_runtime_context",
]
