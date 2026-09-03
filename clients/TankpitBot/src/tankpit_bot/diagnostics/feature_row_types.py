"""Row shape and vocabulary for the tick-level feature table.

Split from :mod:`tankpit_bot.diagnostics.feature_rows` (2026-09-02) when the
run record landed: the derivation and its CLI stay there, the provenance
record lives in :mod:`tankpit_bot.diagnostics.feature_provenance`, and both
import the shapes from here.

The split is a dependency direction rather than a file-size measure. The
provenance module summarises rows, and the derivation module writes the
record it produces, so with the shapes in either one of them the two import
each other. Same reason and same shape as
:mod:`tankpit_bot.diagnostics.run_digest_types`.
"""

from __future__ import annotations

from typing_extensions import TypedDict

#: Emitted when a tick recorded no action outcome. An absent action is
#: a fact about the tick, not a missing value to be imputed later.
NO_ACTION = ""

#: Diagnostic kinds counted per tick as features. Chosen because each
#: names a distinct decision-path event rather than a state sample:
#: what the planner declined, dispatched, or failed at on that tick.
COUNTED_KINDS: tuple[str, ...] = (
    "hop_declined",
    "radar_dispatch",
    "container_pickup_dispatched",
    "plan_released",
    "command_error",
    "fleet_knowledge_merged",
)


class FeatureRowDict(TypedDict):
    """One tick of one run, flattened for tabular modelling.

    Attributes:
        tick_n: The tick this row describes; the join key every
            diagnostic but ``session_room_joined`` carries.
        bot_state: HFSM state and mode at the tick, e.g.
            ``"COLLECT/SENSE"``, as recorded on the tick's events.
        action_kind: The action the tick dispatched, or
            :data:`NO_ACTION` when the tick recorded no outcome.
        outcome: How that action resolved (``"hit"``, ``"miss"``,
            ``"radar_complete"``, ...), or :data:`NO_ACTION`.
        duration_ms: How long the action took, or ``-1`` when the tick
            recorded no outcome. Negative marks absence explicitly
            rather than colliding with a real zero-length action.
        attempt_id: Which attempt at this action the outcome belongs
            to, or ``-1`` when absent; a rising value across ticks is
            a retry.
        hop_declined: Count of declined hop lanes on the tick.
        radar_dispatch: Count of radar dispatches on the tick.
        container_pickup_dispatched: Count of pickups dispatched.
        plan_released: Count of plan releases.
        command_error: Count of command errors.
        fleet_knowledge_merged: Count of fleet knowledge merges.
    """

    tick_n: int
    bot_state: str
    action_kind: str
    outcome: str
    duration_ms: int
    attempt_id: int
    hop_declined: int
    radar_dispatch: int
    container_pickup_dispatched: int
    plan_released: int
    command_error: int
    fleet_knowledge_merged: int


__all__ = [
    "COUNTED_KINDS",
    "NO_ACTION",
    "FeatureRowDict",
]
