"""Tick-side claim arbitration: a collect plan must own its container.

The bridge between the pure intent layer (``bot.ai.intent`` owns what
a plan MEANS) and the fleet's authoritative claim files
(``fleetshare.claims`` owns the exclusive-create mutex). It runs once
per full tick between decide and execute, so filesystem I/O never
enters the decision cascade and no command for a contested container
ever reaches the wire ([[fleet-forage-allocation]]).
"""

from __future__ import annotations

from tankpit_bot.bot.ai.intent import current_collect_plan, release_collect_plan
from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import make_hold_command
from tankpit_bot.fleetshare import acquire_container_claim, release_container_claim
from tankpit_bot.fleetshare.claims import CLAIM_TTL_MS
from tankpit_bot.runtime_artifacts import resolve_bot_instance
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_service import WorldService


def _drop_held_claim(ws: WorldService, instance: str) -> None:
    """Release the authoritative claim this session holds, if any.

    Args:
        ws: The session's world service, holding the held-claim pair.
        instance: This bot's instance name.

    Raises:
        ValueError: If a claim is held with no selected room — a claim
            can only ever be acquired inside a room, so this state is
            a broken invariant, not a condition to absorb.
    """
    if ws.held_claim_x < 0:
        return
    room = ws.selected_room
    if room is None:
        raise ValueError(
            f"held container claim ({ws.held_claim_x},{ws.held_claim_y}) "
            "with no selected room - claims are acquired in-room only"
        )
    release_container_claim(room, ws.held_claim_x, ws.held_claim_y, instance=instance)
    ws.held_claim_x = -1
    ws.held_claim_y = -1


def _arbitrate_collect_claim(
    ws: WorldService,
    decision: TickDecisionDict,
    tank_id: int,
    now_ms: int,
) -> TickDecisionDict:
    """Grant or refuse the tick's collect plan its container claim.

    The reconciliation runs on every full tick, BETWEEN decide and
    execute: whatever plan the decision wants to persist must hold the
    container's authoritative claim file before its command may reach
    the wire. Acquisition is idempotent (own claim refreshes), so held
    plans pay one refresh write per tick; a plan that just latched
    pays one exclusive create — and when a sibling won that create in
    the same tick, the plan dies HERE, for the price of one held beat,
    instead of after the journey the contention measurement priced
    ([[fleet-forage-allocation]]: 273 contested tiles, median gap
    0 s). A denied tile lands in the session's own denial memory —
    NOT the advisory claimed set, which every merge pass replaces
    wholesale — so the very next derivation plans around it even when
    the winner crashed before ever publishing its advisory row.

    Reconciling state each tick rather than tracking transitions also
    self-heals the executor-refusal case: when a dispatch fails, the
    old AI state (and its old plan) persists while this tick's claim
    moved on — the next tick's reconciliation simply releases the
    orphan and re-acquires the plan the state actually holds.

    Args:
        ws: The session's world service — the held-claim pair, the
            selected room, and the advisory claimed set all live
            there.
        decision: The tick's decision, carrying the AI state to
            persist.
        tank_id: This session's tank id, for the claim metadata.
        now_ms: Current wall-clock ms.

    Returns:
        The decision unchanged when its plan holds the claim (or holds
        no plan, or the session has no selected room); a hold-command
        decision with the plan released (reason ``claim_lost``) when a
        sibling owns the container.
    """
    instance = resolve_bot_instance()
    ws.claim_denied_tiles = {
        tile: denied_ms
        for tile, denied_ms in ws.claim_denied_tiles.items()
        if now_ms - denied_ms <= CLAIM_TTL_MS
    }
    next_state = decision["updated_ai_state"]
    plan = current_collect_plan(next_state)
    if plan is None:
        _drop_held_claim(ws, instance)
        return decision
    room = ws.selected_room
    if room is None:
        # The same scope law as ``build_fleet_report``'s pre-join
        # return: the fleet exchange exists only in-room, so a session
        # with no selected room (the sim seam's direct-entry harness,
        # pre-join ticks) has no siblings on any channel and nobody to
        # contend with — arbitration passes through rather than
        # claiming into a namespace no coordinates anchor.
        return decision
    target_x = plan["target_x"]
    target_y = plan["target_y"]
    newly_claimed = (ws.held_claim_x, ws.held_claim_y) != (target_x, target_y)
    if newly_claimed:
        _drop_held_claim(ws, instance)
    if acquire_container_claim(
        room, target_x, target_y, instance=instance, tank_id=tank_id, now_ms=now_ms
    ):
        ws.held_claim_x = target_x
        ws.held_claim_y = target_y
        if newly_claimed:
            emit_diagnostic(
                diagnostic_kind="container_claim_acquired",
                x=target_x,
                y=target_y,
            )
        return decision
    emit_diagnostic(
        diagnostic_kind="container_claim_denied",
        x=target_x,
        y=target_y,
    )
    ws.claim_denied_tiles[f"{target_x},{target_y}"] = now_ms
    released = release_collect_plan(next_state, reason="claim_lost")
    return make_tick_decision(
        command=make_hold_command(),
        behavior=make_behavior_score("COLLECT", 0, target_x, target_y, "claim_denied"),
        updated_ai_state=released,
        desired_equipment=decision["desired_equipment"],
    )


__all__ = [
    "_arbitrate_collect_claim",
    "_drop_held_claim",
]
