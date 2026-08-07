"""Emit belief-vs-truth entity alignment samples.

The bot's container beliefs (``world_state["containers"]``) are built
only from wire messages it decodes (radar responses, viewport updates);
the live JS client may render containers the bot never learned about --
for example containers discovered by other players before the bot
joined the room. This module pairs the bot's container list with the
client's raw ``activeGame.h`` collections at the tick boundary and
emits an ``entity_alignment_sample`` DIAGNOSTIC so the offline analyzer
(:mod:`tankpit_bot.diagnostics.entity_map`) can identify the client's
container collection and quantify exactly what the bot is blind to.

Emission is change-gated on the belief container signature (the set of
``(x, y, is_fuel)`` triples): samples are written when the bot's
container knowledge changes -- radar reveals, pickups, expiries -- which
are precisely the moments worth comparing against the client. During
combat the gate is bypassed and every tick emits: enemy tank fields
(position, damage) change tick-to-tick, and the ~9s change-gated cadence
proved too sparse to correlate the client registry's health candidate
against wire damage-tier transitions (run 20260610-233x).
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, dump_json_str

from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.browser.page_client_snapshot_codecs import encode_client_collections
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.state.types import WorldStateDict, encode_container_state


class EntityAlignmentEmitter:
    """Per-session change-gate for ``entity_alignment_sample`` diagnostics.

    The gate is one session's memory of what it last wrote, so it is
    instance state: two sessions in one process each need their own
    ([[session-state-deglobalisation]]). A fresh instance starts with
    the gate clear, which is why no reset function exists.
    """

    def __init__(self) -> None:
        self._last_signature: frozenset[tuple[int, int, bool]] | None = None

    def maybe_emit(
        self,
        world: WorldStateDict,
        snapshot: PageClientSnapshotDict,
        *,
        in_combat: bool,
    ) -> bool:
        """Emit an ``entity_alignment_sample`` DIAGNOSTIC when beliefs changed.

        Args:
            world: Bot's wire-derived world state for this tick.
            snapshot: Live page-client snapshot captured in the same tick.
            in_combat: True while the bot is engaged; bypasses the
                change-gate so enemy tank dynamics (position, damage) are
                sampled every tick for offline field-mapping correlation.

        Returns:
            True when a sample was emitted; False when the capture was
            skipped because the snapshot carries no world collections
            (client world object not yet populated) or the belief container
            signature is unchanged since the last emitted sample outside
            combat.
        """
        if not snapshot["world_collections"]:
            return False
        containers = list(world["containers"].values())
        signature = frozenset((c["x"], c["y"], c["is_fuel"]) for c in containers)
        if not in_combat and signature == self._last_signature:
            return False
        self._last_signature = signature
        belief_payload: JSONObject = {
            "containers": [encode_container_state(c) for c in containers],
        }
        emit_diagnostic(
            diagnostic_kind="entity_alignment_sample",
            belief_container_count=len(containers),
            belief_containers_json=dump_json_str(belief_payload),
            world_collections_json=dump_json_str(
                encode_client_collections(snapshot["world_collections"])
            ),
        )
        return True


__all__ = [
    "EntityAlignmentEmitter",
]
