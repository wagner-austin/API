"""Shared helpers for tests that read or drive the runtime event stream.

``test_runtime_logging.py`` and ``test_runtime_records.py`` both inject
synthetic ``LogRecord``s to exercise ``_HookEventArtifactHandler``'s
malformed-record guards, and both need the same thing: a logger whose
records actually reach that handler.

:func:`capture_runtime_events` serves the other direction -- reading what
the emitters produced -- which is the only way to pin a guard whose
entire effect is a diagnostic that does NOT fire.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from platform_core.logging import stdlib_logging

from tankpit_bot.runtime_logging_handlers import (
    EMITTER_LOGGER_NAME,
    make_run_id,
    session_logger_name,
)
from tankpit_bot.runtime_records import _RuntimeRecordMapping


def run_child_logger(stamp: str, suffix: str) -> stdlib_logging.Logger:
    """Return a logger whose records reach the bot run's event handler.

    As of [[session-state-deglobalisation]] step 10 the event handler
    mounts on the run's own logger rather than root, so a synthetic
    record has to be logged inside that subtree to reach it. A record on
    an unrelated logger never arrives at the handler at all — the
    "artifact stayed empty" assertion would still pass, and the test
    would have quietly stopped exercising the guard it documents.

    Args:
        stamp: Archive stamp the bot run was configured with.
        suffix: Leaf name distinguishing this test's logger.

    Returns:
        A logger under the configured run's emitter subtree.
    """
    parent = session_logger_name(make_run_id("bot", stamp))
    return stdlib_logging.getLogger(f"{parent}.{suffix}")


def event_fields(record: stdlib_logging.LogRecord) -> dict[str, str | int | float | bool]:
    """Return a runtime-event record's structured fields.

    The emitters nest caller-supplied fields under ``runtime_fields``
    rather than flattening them onto the record, so reading
    ``record.diagnostic_kind`` finds nothing and any filter built on it
    silently matches zero records (which is exactly how one assertion
    passed while its mutant lived, 2026-08-08).

    Args:
        record: Record captured from the emitter logger.

    Returns:
        The record's structured fields, or an empty dict for a record
        that carries none.
    """
    mapping: _RuntimeRecordMapping = record.__dict__
    fields = mapping.get("runtime_fields")
    if isinstance(fields, dict):
        return fields
    return {}


@contextmanager
def capture_runtime_events() -> Generator[list[stdlib_logging.LogRecord], None, None]:
    """Collect every runtime-event record emitted inside the block.

    ``emit_ai``, ``emit_diagnostic`` and ``emit_wire`` all land on the
    emitter logger, so the event stream is the only place to observe a
    guard whose whole effect is a diagnostic that does NOT fire -- a
    return value alone cannot distinguish "declined for this reason"
    from "declined for the next reason down". Two sniffer tests had
    hand-rolled this same addHandler / setLevel / finally-remove block
    (consolidated 2026-08-08).

    The handler and the level are both restored in a ``finally``: a
    leaked handler would keep appending to a dead list for every later
    test on the same xdist worker.

    Yields:
        The list records accumulate into, in emission order. Use
        :func:`event_fields` to read a record's structured fields.
    """
    records: list[stdlib_logging.LogRecord] = []

    class _Capture(stdlib_logging.Handler):
        def emit(self, record: stdlib_logging.LogRecord) -> None:
            """Append the emitted record.

            Args:
                record: Record emitted on the emitter logger.
            """
            records.append(record)

    logger = stdlib_logging.getLogger(EMITTER_LOGGER_NAME)
    handler = _Capture()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(stdlib_logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
