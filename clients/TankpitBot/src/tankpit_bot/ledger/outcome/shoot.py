"""Shoot outcome emitters.

Resolutions from the per-shot ammo-consumption ledger (consumption =
hit, user contract 2026-07-02): ``hit`` with the signal that proved it
(tile-occupied echo, confirmed kill, or 0x49 ammo-delta
reconciliation), ``miss`` (response arrived, nothing debited),
``command_rejected`` (0x52 -- no ShootEvent, no ammo moved), and the
executor's ``discarded_target_not_tracked`` race guard (the tank
vanished from the registry between plan and dispatch).

The hit/miss payloads carry the target identity, not coordinates --
the classifier resolves shots by ``target_id`` (aim coords are a
server hint under the viewport clamp), so coordinates would be
fabricated context.
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.ring import ActionOutcomeRecordDict

HitSignal = Literal["tile_occupied", "kill_confirmed", "ammo_delta"]
"""Which wire channel proved the hit."""


def emit_shoot_hit(
    *,
    duration_ms: int,
    target_id: int,
    target_name: str,
    victim_id: int,
    on_intended_target: bool,
    hit_signal: HitSignal,
) -> ActionOutcomeRecordDict:
    """Record a landed shot (ammo was debited or the kill confirmed).

    Args:
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_id: Commanded target's tank id.
        target_name: Commanded target's name.
        victim_id: Tank id the wire attributed the impact to (-1 when
            the impact tile is off-viewport and unresolvable).
        on_intended_target: Whether the victim matches the command
            (homing seekers can land on a closer enemy).
        hit_signal: Which wire channel proved the hit.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="shoot",
        outcome="hit",
        duration_ms=duration_ms,
        target_id=target_id,
        target_name=target_name,
        victim_id=victim_id,
        on_intended_target=on_intended_target,
        hit_signal=hit_signal,
    )


def emit_shoot_miss(
    *, duration_ms: int, target_id: int, target_name: str
) -> ActionOutcomeRecordDict:
    """Record a genuine miss (response arrived, nothing was debited).

    Args:
        duration_ms: Dispatch-to-resolution wall-clock ms.
        target_id: Commanded target's tank id.
        target_name: Commanded target's name.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="shoot",
        outcome="miss",
        duration_ms=duration_ms,
        target_id=target_id,
        target_name=target_name,
    )


def emit_shoot_command_rejected(
    *, duration_ms: int, target_id: int, target_name: str, error_code: int
) -> ActionOutcomeRecordDict:
    """Record a shot the server refused with a 0x52 error.

    Args:
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_id: Commanded target's tank id.
        target_name: Commanded target's name.
        error_code: The 0x52 error code.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="shoot",
        outcome="command_rejected",
        duration_ms=duration_ms,
        target_id=target_id,
        target_name=target_name,
        error_code=error_code,
    )


__all__ = [
    "HitSignal",
    "emit_shoot_command_rejected",
    "emit_shoot_hit",
    "emit_shoot_miss",
]
