"""Emit belief-vs-truth self-state alignment samples.

The bot's belief about its own tank (``world_state["self_state"]``) is
built from decoded wire messages; the live JS client's truth is the
minified ``activeGame.i`` field map captured per tick in
:attr:`tankpit_bot.browser.page_client_snapshot.PageClientSnapshotDict.self_fields`.
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

from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.browser.page_client_snapshot_codecs import encode_client_field_map
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.state.types import SelfStateDict


class SelfAlignmentEmitter:
    """Per-session change-gate for ``self_alignment_sample`` diagnostics.

    The gate is one session's memory of what it last wrote, so it is
    instance state: two sessions in one process each need their own
    ([[session-state-deglobalisation]]). A fresh instance starts with
    the gate clear, which is why no reset function exists.
    """

    def __init__(self) -> None:
        self._last_belief: tuple[int, int, int, int] | None = None

    def maybe_emit(
        self,
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
        if not snapshot["self_fields"]:
            return False
        belief = (
            self_state["tank_id"],
            self_state["x"],
            self_state["y"],
            self_state["fuel"],
        )
        if belief == self._last_belief:
            return False
        self._last_belief = belief
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
    "SelfAlignmentEmitter",
]
