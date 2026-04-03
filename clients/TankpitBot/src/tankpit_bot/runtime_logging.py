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

from platform_core.json_utils import JSONObject, dump_json_str, require_str
from platform_core.logging import get_logger, setup_rich_logging, stdlib_logging
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_artifacts import (
    BotRunArtifactsDict,
    SniffRunArtifactsDict,
    build_bot_run_artifacts,
    build_sniff_run_artifacts,
    make_run_stamp,
)

_EMITTER_LOGGER = get_logger("tankpit_bot.runtime.events")
_ARTIFACT_HANDLER_NAME_PREFIX = "tankpit_bot.runtime.artifacts."

_BOT_ARTIFACTS: BotRunArtifactsDict | None = None
_SNIFF_ARTIFACTS: SniffRunArtifactsDict | None = None


class RuntimeEventRecordDict(TypedDict):
    """Structured runtime event record.

    Attributes:
        timestamp: Local wall-clock timestamp for the emitted log record.
        level: Logging level name.
        logger: Logger name that emitted the event.
        mode: Runtime mode, either ``bot`` or ``sniff``.
        channel: High-level event channel such as ``AI`` or ``WORLD``.
        message: Human-readable event text without the channel prefix.
    """

    timestamp: str
    level: str
    logger: str
    mode: str
    channel: str
    message: str


class RuntimeLogExtraDict(TypedDict):
    """Structured extra fields carried on high-signal runtime log records."""

    runtime_channel: str
    runtime_message: str
    runtime_mode: str


class _RuntimeRecordMapping(Protocol):
    """Minimal typed access to runtime-specific LogRecord extras."""

    def __contains__(self, key: str) -> bool:
        """Return True when a runtime extra exists."""
        ...

    def __getitem__(self, key: str) -> str | int | float | bool | None:
        """Return a runtime extra value."""
        ...


def encode_runtime_event_record(record: RuntimeEventRecordDict) -> JSONObject:
    """Encode a runtime event record to JSON-compatible data.

    Args:
        record: Runtime event record.

    Returns:
        JSON-compatible representation.
    """
    return {
        "timestamp": record["timestamp"],
        "level": record["level"],
        "logger": record["logger"],
        "mode": record["mode"],
        "channel": record["channel"],
        "message": record["message"],
    }


def decode_runtime_event_record(data: JSONObject) -> RuntimeEventRecordDict:
    """Decode a runtime event record from JSON-compatible data.

    Args:
        data: JSON object to decode.

    Returns:
        Validated runtime event record.
    """
    return RuntimeEventRecordDict(
        timestamp=require_str(data, "timestamp"),
        level=require_str(data, "level"),
        logger=require_str(data, "logger"),
        mode=require_str(data, "mode"),
        channel=require_str(data, "channel"),
        message=require_str(data, "message"),
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
        event = RuntimeEventRecordDict(
            timestamp=datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S"),
            level=record.levelname,
            logger=record.name,
            mode=self._mode,
            channel=channel_raw,
            message=message_raw,
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
    global _BOT_ARTIFACTS, _SNIFF_ARTIFACTS
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
    return artifacts


def configure_sniff_runtime_logging(stamp: str | None = None) -> SniffRunArtifactsDict:
    """Configure console logging plus canonical sniffer artifact outputs.

    Args:
        stamp: Optional archive timestamp stamp for deterministic tests.

    Returns:
        Configured sniffer runtime artifacts.
    """
    global _BOT_ARTIFACTS, _SNIFF_ARTIFACTS
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = build_sniff_run_artifacts(resolved_stamp)
    setup_rich_logging(level="INFO")
    _reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
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
    return artifacts


def get_bot_runtime_artifacts() -> BotRunArtifactsDict | None:
    """Return the active bot runtime artifacts for this process, if configured."""
    return _BOT_ARTIFACTS


def get_sniff_runtime_artifacts() -> SniffRunArtifactsDict | None:
    """Return the active sniffer runtime artifacts for this process, if configured."""
    return _SNIFF_ARTIFACTS


def emit_ai(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured AI event."""
    _emit_runtime_event("AI", message, *args)


def emit_sync(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured synchronization event."""
    _emit_runtime_event("SYNC", message, *args)


def emit_state(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured state-machine event."""
    _emit_runtime_event("STATE", message, *args)


def emit_wire(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured wire/protocol command event."""
    _emit_runtime_event("WIRE", message, *args)


def emit_world(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured world-state event."""
    _emit_runtime_event("WORLD", message, *args)


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


def _emit_runtime_event(
    channel: str,
    message: str,
    *args: str | int | float | bool,
) -> None:
    """Emit a runtime event to both console logs and JSONL artifacts.

    Args:
        channel: Event channel such as ``AI`` or ``WORLD``.
        message: ``printf``-style message string without the channel prefix.
        *args: Format arguments for the message string.
    """
    formatted = message % args if args else message
    extra = RuntimeLogExtraDict(
        runtime_channel=channel,
        runtime_message=formatted,
        runtime_mode=_runtime_mode_name(),
    )
    _EMITTER_LOGGER.info("%s: %s", channel, formatted, extra=extra)


def _runtime_mode_name() -> str:
    """Return the active runtime mode name.

    Returns:
        ``bot`` or ``sniff`` when configured, otherwise ``unconfigured``.
    """
    if _BOT_ARTIFACTS is not None:
        return "bot"
    if _SNIFF_ARTIFACTS is not None:
        return "sniff"
    return "unconfigured"


__all__ = [
    "RuntimeEventRecordDict",
    "configure_bot_runtime_logging",
    "configure_sniff_runtime_logging",
    "decode_runtime_event_record",
    "emit_ai",
    "emit_state",
    "emit_sync",
    "emit_wire",
    "emit_world",
    "encode_runtime_event_record",
    "get_bot_runtime_artifacts",
    "get_sniff_runtime_artifacts",
]
