"""Emit belief-vs-truth self-state alignment samples.

The bot's belief about its own tank (``world_state["self_state"]``) is
built from decoded wire messages; the live JS client's truth is the
minified ``activeGame.i`` field map captured per tick in
:attr:`tankpit_bot.action_lab.page_client_snapshot.PageClientSnapshotDict.self_fields`.
This module pairs the two at the tick boundary and emits a
``self_alignment_sample`` DIAGNOSTIC so the offline analyzer
(:mod:`tankpit_bot.diagnostics.self_map`) can discover which minified
keys carry tank_id / x / y / fuel -- the prerequisite for live
belief-divergence detection.

Emission is change-gated on the belief tuple: a sample is written only
when (tank_id, x, y, fuel) differs from the previously emitted belief.
Repeated identical ticks add no mapping information and would bloat the
artifact; distinct belief values are exactly what the key-discovery
intersection needs.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.page_client_snapshot_codecs import encode_client_field_map
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.state.types import SelfStateDict

_last_emitted_belief: tuple[int, int, int, int] | None = None


def reset_self_alignment_emitter() -> None:
    """Reset the change-gate so the next sample always emits.

    Called from test isolation fixtures; a fresh bot process starts
    with the gate already clear.
    """
    global _last_emitted_belief
    _last_emitted_belief = None


def maybe_emit_self_alignment_sample(
    self_state: SelfStateDict,
    snapshot: PageClientSnapshotDict,
) -> bool:
    """Emit a ``self_alignment_sample`` DIAGNOSTIC when belief changed.

    Args:
        self_state: Bot's wire-derived self-tank belief for this tick.
        snapshot: Live page-client snapshot captured in the same tick.

    Returns:
        True when a sample was emitted; False when the capture was
        skipped because the snapshot carries no self fields (client
        object not yet populated) or the belief tuple is unchanged
        since the last emitted sample.
    """
    global _last_emitted_belief
    if not snapshot["self_fields"]:
        return False
    belief = (
        self_state["tank_id"],
        self_state["x"],
        self_state["y"],
        self_state["fuel"],
    )
    if belief == _last_emitted_belief:
        return False
    _last_emitted_belief = belief
    emit_diagnostic(
        diagnostic_kind="self_alignment_sample",
        belief_tank_id=self_state["tank_id"],
        belief_x=self_state["x"],
        belief_y=self_state["y"],
        belief_fuel=self_state["fuel"],
        self_fields_json=dump_json_str(encode_client_field_map(snapshot["self_fields"])),
    )
    return True


__all__ = [
    "maybe_emit_self_alignment_sample",
    "reset_self_alignment_emitter",
]
