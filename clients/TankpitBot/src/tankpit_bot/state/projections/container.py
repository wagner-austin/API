"""Phase 1b: project ``ContainerStateDict`` as a ``Fact[T]``.

``ContainerStateDict`` carries the full fact metadata flat
(``source``/``refresh_kind``/``timestamp_ms`` plus the Phase 1b
``confidence`` and ``provenance`` fields); :func:`container_fact`
projects it into a true :class:`~tankpit_bot.facts.fact.Fact` for the
layers that consume Facts (ledger evidence refs, confidence ops).

Deviation from the handoff spec ("retrofit ContainerStateDict to be a
Fact[ContainerValueDict]"): the flat shape is kept and the Fact view
is a projection. Nesting the value under ``["value"]`` would touch
~200 construction sites and ~300 access sites across 68 files for
zero information gain; the flat dict already carries every Fact field.

This module is intentionally NOT re-exported from
``tankpit_bot.facts`` -- it imports ``state.types``, and keeping it
out of the package ``__init__`` keeps the import graph acyclic
(``state.types.container`` imports ``facts.provenance``).
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.facts.fact import Fact, make_fact
from tankpit_bot.state.types.container import ContainerStateDict
from tankpit_bot.types.constants import ContainerRefreshKind


class ContainerValueDict(TypedDict):
    """The believed value of a container fact (metadata stripped).

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        is_fuel: True if fuel container, False if equipment.
        volume: Fuel amount (0 for equipment).
        refresh_kind: Specific confirmation path that most recently
            refreshed this container.
        failed_pickups: How many pickup attempts failed.
    """

    x: int
    y: int
    is_fuel: bool
    volume: int
    refresh_kind: ContainerRefreshKind
    failed_pickups: int


def container_fact(state: ContainerStateDict) -> Fact[ContainerValueDict]:
    """Project a container state into a Fact.

    Args:
        state: Container state carrying flat fact metadata.

    Returns:
        Fact whose value is the container's believed contents.

    Raises:
        NoUnsourcedFactError: If the state's metadata is incomplete.
        ConfidenceOutOfBoundsError: If confidence is out of range.
        ProvenanceRootednessError: If the provenance is not rooted.
    """
    return make_fact(
        value=ContainerValueDict(
            x=state["x"],
            y=state["y"],
            is_fuel=state["is_fuel"],
            volume=state["volume"],
            refresh_kind=state["refresh_kind"],
            failed_pickups=state["failed_pickups"],
        ),
        source=state["provenance"]["origin"],
        observed_ms=state["timestamp_ms"],
        confidence=state["confidence"],
        provenance=state["provenance"],
    )


__all__ = [
    "ContainerValueDict",
    "container_fact",
]
