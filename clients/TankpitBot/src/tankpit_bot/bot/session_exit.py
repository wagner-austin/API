"""Typed session-exit request raised by decision owners.

The user contract (2026-07-02): when the bot cannot do its job, it must
not degenerate into loops -- it ends the session with an explicit,
analyzable reason. Three conditions are terminal:

* ``no_viable_targets`` -- a fresh map snapshot was inspected and no
  enemy passed the acquisition gates (affordability included). Raised
  by the HUNT owner.
* ``out_of_fuel`` -- the COLLECT owner has no lock, no pickup, no
  forage action, and cannot afford any search hop. Raised by the
  COLLECT owner (previously a bare ``ValueError`` crash).
* ``no_productive_collect`` -- the COLLECT cascade produced no action
  while inventory is below combat-ready (Bug 0.4/0.7, 2026-07-06):
  fuel is healthy but duals, homings, or radars are below cap and no
  tracked equipment container is affordably teleport-reachable. Under
  the pre-fix code the cascade would yield to HUNT under-armed and
  the bot would engage a fight it could not finish; ending the
  session cleanly is preferable to that loop.

``run_tick_loop`` converts the request into the same graceful shutdown
path as a tick-budget exit: scorecard emitted, summary written, index
row appended -- with ``exit_reason`` set to the request's reason.
"""

from __future__ import annotations

from typing import Literal

SessionExitReason = Literal["no_viable_targets", "out_of_fuel", "no_productive_collect"]


class SessionExitError(Exception):
    """Raised by a decision owner to end the session with a recorded reason.

    Attributes:
        reason: Machine-readable exit reason recorded in the session
            summary and the runs index.
        detail: Human-readable one-line explanation for the log.
    """

    def __init__(self, reason: SessionExitReason, detail: str) -> None:
        """Initialize the exit request.

        Args:
            reason: Machine-readable exit reason.
            detail: Human-readable one-line explanation.
        """
        super().__init__(f"{reason}: {detail}")
        self.reason: SessionExitReason = reason
        self.detail = detail


__all__ = [
    "SessionExitError",
    "SessionExitReason",
]
