"""The runtime event record and its codec.

One wire shape for every runtime event the bot emits, its encode/decode
pair, and the field-narrowing helpers consumers use to read one back.
A leaf of the logging stack: it knows nothing about handlers or
artifacts.
"""

from __future__ import annotations

from typing import Protocol

from platform_core.json_utils import JSONObject, JSONTypeError, require_str
from typing_extensions import TypedDict

_RESERVED_EVENT_KEYS: frozenset[str] = frozenset(
    {"timestamp", "level", "logger", "mode", "channel", "message"}
)


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
    """Structured extra fields carried on high-signal runtime log records.

    There is deliberately no ``runtime_mode`` here. The mode is captured
    by :class:`_HookEventArtifactHandler` when the run is configured and
    written from there, so a per-record copy had no reader
    ([[session-state-deglobalisation]] step 10).
    """

    runtime_channel: str
    runtime_message: str
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


__all__ = [
    "RuntimeEventRecordDict",
    "RuntimeLogExtraDict",
    "decode_runtime_event_record",
    "encode_runtime_event_record",
    "require_bool_field",
    "require_int_field",
    "require_str_field",
]
