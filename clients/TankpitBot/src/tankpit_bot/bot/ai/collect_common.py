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
from tankpit_bot.state.types import ContainerStateDict

COLLECT_SCORE = 925
"""Behavior score every COLLECT-cascade decision carries."""

HOP_SIGHTING_MAX_AGE_MS = 30_000
"""Oldest container sighting a TELEPORT hop may spend fuel on.

Pricing, not expiry: the two-clocks ruling ([[larder-plan]]) keeps
beliefs alive until hard evidence, and this gate does not touch them —
walk service, in-window pickups, and the desperation ladder still use
memory of any age. What it prices is the teleport: in a co-farmed
World (two-plus fleet bots and ~27 practice bots eating containers),
a sighting older than half a minute is more phantom than prize.
Receipts: run bot-20260901-033100 (arterial) drew ELEVEN code-4
empty-container rejections in 2.5 minutes hopping to aged own and
fleet-merged sightings while its zero-radar sweep starved; the
2026-08-14 240 s fleet run drew 18 the same way; run
bot-20260901-024845 (artax) paid three stale-hop teleports. A declined
lane falls through to sweep/frontier discovery — fresh intel instead
of a phantom's teleport bill."""


def is_hop_sighting_fresh(container: ContainerStateDict, now_ms: int) -> bool:
    """Return True when a container sighting is young enough to hop to.

    Args:
        container: Believed container under hop consideration.
        now_ms: Current tick timestamp.

    Returns:
        True when the sighting is within :data:`HOP_SIGHTING_MAX_AGE_MS`.
    """
    return now_ms - container["timestamp_ms"] <= HOP_SIGHTING_MAX_AGE_MS


def split_fresh_hop_sightings(
    containers: list[ContainerStateDict],
    now_ms: int,
) -> tuple[list[ContainerStateDict], int]:
    """Split hop candidates into fresh sightings and a stale count.

    Args:
        containers: Believed containers entering a hop lane.
        now_ms: Current tick timestamp.

    Returns:
        ``(fresh, stale_count)`` — the hop-priceable sightings and how
        many the horizon excluded (the ``stale`` decline tally).
    """
    fresh = [c for c in containers if is_hop_sighting_fresh(c, now_ms)]
    return fresh, len(containers) - len(fresh)


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
