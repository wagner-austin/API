"""The clause a capacity refusal names a node's live load with.

Split out of :mod:`fleet.core.capacity` when the mutation floor's refusal
(MCPs board task 330b0465) took that module past the 600-line ceiling. Every
load refusal :func:`fleet.core.capacity.assess` gives ends its first sentence
with this clause, so a reader told a node is full sees what holds it.
"""

from __future__ import annotations

from fleet.contracts.node import NodeState


def live_clause(state: NodeState) -> str:
    """Name what a node's live runs and reservations hold, for a capacity refusal.

    Args:
        state: What the node reported.

    Returns:
        An empty string when no fleet run is live on the node and no
        reservation holds a core; otherwise ``, and its N live fleet run(s)
        hold W worker(s) on K core(s) and X GB`` and ``, and its reservations
        hold C core(s)``, each only when it holds something.
    """
    live = state["live"]
    runs = (
        ""
        if live["runs"] == 0
        else (
            f", and its {live['runs']} live fleet run(s) hold {live['workers']} worker(s) on "
            f"{live['cores']} core(s) and {live['ram_gb']:.1f} GB"
        )
    )
    held = (
        ""
        if state["reserved_cores"] == 0
        else f", and its reservations hold {state['reserved_cores']} core(s)"
    )
    return runs + held


__all__ = ["live_clause"]
