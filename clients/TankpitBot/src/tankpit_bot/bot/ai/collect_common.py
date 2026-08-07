"""Shared COLLECT-cascade primitives: score and decline telemetry.

Constants and emitters every collect submodule shares: the cascade's
behavior score and the structured hop-decline emitter.

**The container blacklist was deleted 2026-08-07.** It had a reader
consulted at five decision sites, a ``reset`` its docstring said ran
"on death/respawn", and NO writer — ``blacklist_container`` was never
called from ``src/`` in any commit in this repository's history. The
predicate therefore always answered False, so removing it is
behaviour-identical, and the five guards it fed were decisions made
against a set that could never fill ([[session-state-deglobalisation]]
step 7). If per-session container blacklisting is wanted, it needs a
writer first — a reader without one is a decision nobody makes.
"""

from __future__ import annotations

from tankpit_bot.runtime_logging import emit_diagnostic

COLLECT_SCORE = 925
"""Behavior score every COLLECT-cascade decision carries."""


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
    "emit_hop_declined",
]
