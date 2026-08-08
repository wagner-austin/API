"""Artifact log handlers and the logger names a run mounts them on.

Split from :mod:`tankpit_bot.runtime_logging` at the 600-line ceiling
(2026-08-08). This module owns the logging plumbing -- the two handler
classes, the names that identify a run's logger, and installing or
removing a run's handlers. It knows nothing about which run is active;
that is the ambient state :mod:`tankpit_bot.runtime_logging` holds, and
keeping the dependency one-way is what stops the pair closing a cycle.

The two handlers are scoped differently, on purpose. The event stream is
a *session* artifact, so its handler mounts on a per-run logger and two
concurrent sessions each write their own ``events.jsonl``. The text log
is a *process* artifact, so its handler mounts on the root logger --
root is the only logger that sees library records, and a
``world_service`` warning belongs in the run log
([[session-state-deglobalisation]] step 10).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.logging import stdlib_logging

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    _RuntimeRecordMapping,
    encode_runtime_event_record,
)

EMITTER_LOGGER_NAME = "tankpit_bot.runtime.events"

ARTIFACT_HANDLER_NAME_PREFIX = "tankpit_bot.runtime.artifacts."

_TEXT_HANDLER_NAME = ARTIFACT_HANDLER_NAME_PREFIX + "text"

_EVENT_HANDLER_NAME = ARTIFACT_HANDLER_NAME_PREFIX + "events"


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


def session_logger_name(run_id: str) -> str:
    """Return the per-run emitter logger name for ``run_id``.

    A child of the base emitter logger, so records still propagate up to
    the root logger's text handler and the console while the run's event
    handler sits on the child alone.

    Args:
        run_id: Run identity from :func:`make_run_id`.

    Returns:
        Dotted logger name for this run's emitter.
    """
    return f"{EMITTER_LOGGER_NAME}.{run_id}"


def make_run_id(mode: str, stamp: str) -> str:
    """Return a logger-safe identity for one configured run.

    Dots would split into extra logger-hierarchy levels, so they are
    replaced; the result still distinguishes every concurrent run,
    because a run is identified by its mode and its archive stamp.

    Args:
        mode: Runtime mode name (``bot``, ``sniff``, ``probe:<name>``).
        stamp: Archive timestamp stamp for this run.

    Returns:
        Run identity safe to embed in a logger name.
    """
    return f"{mode}:{stamp}".replace(".", "_")


def reset_artifact_files(*paths: Path) -> None:
    """Clear artifact files at process startup.

    Args:
        paths: Artifact paths to reset to empty content.
    """
    for path in paths:
        _test_hooks.write_text(path, "")


def install_artifact_handlers(
    run_id: str,
    mode: str,
    latest_log_path: Path,
    archive_log_path: Path,
    latest_events_path: Path,
    archive_events_path: Path,
) -> None:
    """Attach this run's artifact mirroring handlers.

    The two handlers land in different places, and the asymmetry is the
    point. The **text** handler goes on the root logger, because root is
    the only logger that sees library records — a ``world_service``
    warning belongs in the run log. It replaces any previous text
    handler: the process has one text log, and the most recent
    ``configure_*`` owns it. The **event** handler goes on this run's own
    logger, so a second concurrent session writes its own
    ``events.jsonl`` instead of stealing the first's stream.

    Args:
        run_id: Run identity from :func:`make_run_id`.
        mode: Runtime mode (``bot``, ``sniff``, ``probe:<name>``),
            written into every event record this run emits.
        latest_log_path: Stable latest text log path.
        archive_log_path: Timestamped archived text log path.
        latest_events_path: Stable latest JSONL event path.
        archive_events_path: Timestamped archived JSONL event path.
    """
    root = stdlib_logging.getLogger()
    remove_artifact_handlers(root)
    text_handler = _HookTextArtifactHandler((latest_log_path, archive_log_path))
    text_handler.set_name(_TEXT_HANDLER_NAME)
    text_handler.setLevel(root.level)
    text_handler.setFormatter(
        stdlib_logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%m/%d/%y %H:%M:%S")
    )
    root.addHandler(text_handler)

    session_logger = stdlib_logging.getLogger(session_logger_name(run_id))
    # Re-configuring the same run id (deterministic stamps in tests, a
    # service session restarting on the same stamp) must not stack a
    # second handler and double every event.
    remove_artifact_handlers(session_logger)
    event_handler = _HookEventArtifactHandler(mode, (latest_events_path, archive_events_path))
    event_handler.set_name(_EVENT_HANDLER_NAME)
    event_handler.setLevel(root.level)
    session_logger.addHandler(event_handler)


def remove_artifact_handlers(logger: stdlib_logging.Logger) -> None:
    """Remove previously installed runtime artifact handlers from ``logger``.

    Args:
        logger: Logger to clean before installing fresh handlers.
    """
    handlers_to_keep: list[stdlib_logging.Handler] = []
    for handler in logger.handlers:
        name = handler.get_name() or ""
        if not name.startswith(ARTIFACT_HANDLER_NAME_PREFIX):
            handlers_to_keep.append(handler)
    logger.handlers = handlers_to_keep


__all__ = [
    "ARTIFACT_HANDLER_NAME_PREFIX",
    "EMITTER_LOGGER_NAME",
    "install_artifact_handlers",
    "make_run_id",
    "remove_artifact_handlers",
    "reset_artifact_files",
    "session_logger_name",
]
