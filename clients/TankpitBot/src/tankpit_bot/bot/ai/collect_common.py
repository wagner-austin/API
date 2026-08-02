"""Shared COLLECT-cascade primitives: score, blacklist, decline telemetry.

Session-scoped state and constants every collect submodule shares:
the cascade's behavior score, the permanent container blacklist
(cleared on death/respawn), and the structured hop-decline emitter.
"""

from __future__ import annotations

from tankpit_bot.runtime_logging import emit_diagnostic

COLLECT_SCORE = 925
"""Behavior score every COLLECT-cascade decision carries."""

_blacklisted_container_keys: set[str] = set()


def blacklist_container(x: int, y: int) -> None:
    """Permanently blacklist a container for this session.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    key = f"{x},{y}"
    if key not in _blacklisted_container_keys:
        emit_diagnostic(diagnostic_kind="container_blacklisted", x=x, y=y)
    _blacklisted_container_keys.add(key)


def is_container_blacklisted(x: int, y: int) -> bool:
    """Check if a container is permanently blacklisted.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.

    Returns:
        True if the container has been blacklisted this session.
    """
    return f"{x},{y}" in _blacklisted_container_keys


def reset_container_blacklist() -> None:
    """Clear the container blacklist (called on death/respawn)."""
    _blacklisted_container_keys.clear()


def emit_hop_declined(hop_kind: str, **tallies: int) -> None:
    """Record a structured hop decline with per-branch tallies.

    The hop selectors' silent ``continue``/``return None`` branches
    made the 2026-07-18 early-exit undiagnosable post-hoc (the run
    ended ``no_productive_collect`` with 10 tracked containers and no
    record of which filter refused each). Every decline now states
    its arithmetic.

    Args:
        hop_kind: Which selector declined (``equipment`` / ``dot``).
        **tallies: Per-branch counts and the governing numbers.
    """
    emit_diagnostic(diagnostic_kind="hop_declined", hop_kind=hop_kind, **tallies)


__all__ = [
    "COLLECT_SCORE",
    "blacklist_container",
    "emit_hop_declined",
    "is_container_blacklisted",
    "reset_container_blacklist",
]
