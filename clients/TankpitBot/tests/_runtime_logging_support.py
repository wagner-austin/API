"""Shared helper for the runtime-logging handler tests.

``test_runtime_logging.py`` and ``test_runtime_records.py`` both inject
synthetic ``LogRecord``s to exercise ``_HookEventArtifactHandler``'s
malformed-record guards, and both need the same thing: a logger whose
records actually reach that handler.
"""

from __future__ import annotations

from platform_core.logging import stdlib_logging

from tankpit_bot.runtime_logging_handlers import make_run_id, session_logger_name


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
