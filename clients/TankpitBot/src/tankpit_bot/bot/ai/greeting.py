"""One-shot HELLO greeting when a human combat target is acquired.

User request (2026-07-29): when the bot finishes collecting and locks
onto the human player, it says HELLO. The hook rides the decision
pipeline: every HUNT decision that carries a combat lock on a
human-classified tank the bot has not yet greeted gets the chat
attached as its ``secondary_command`` — the executor dispatches it in
the same tick window as the acquisition move, so the greeting lands as
the bot comes for them.

Flood-mute discipline ([[chat-messages]], sniff-20260729-214411: after
8 rapid sends the server silently swallowed chat for the rest of the
session): exactly one greeting per greeted tank id, latched in
``ai_state["greeted_target_id"]``, never retried on silence.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.humans import is_human_name
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import make_chat_command
from tankpit_bot.protocol.chat import CHAT_HELLO
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic


def attach_human_greeting(ctx: DecideCtx, decision: TickDecisionDict) -> TickDecisionDict:
    """Attach a one-shot HELLO to the decision that locks a new human.

    Fires when the decision's updated AI state carries a combat lock on
    a registry tank whose name classifies as human and whose id differs
    from the last greeted id. The greeting never displaces a planned
    secondary command — those ticks skip, and the unchanged latch
    retries on the next locked tick.

    Args:
        ctx: Decision context (registry lookup + self position).
        decision: HUNT-owner decision leaving the arbitrator.

    Returns:
        The decision unchanged, or with the HELLO chat attached as
        ``secondary_command`` and ``greeted_target_id`` latched.
    """
    state = decision["updated_ai_state"]
    target_id = state["combat_target_id"]
    if target_id == -1 or target_id == state["greeted_target_id"]:
        return decision
    if decision["secondary_command"] is not None:
        return decision
    tank = ctx.world["tanks"].get(str(target_id))
    if tank is None or not is_human_name(tank["name"]):
        return decision
    emit_ai("greeting human %s (id=%d) with HELLO", tank["name"], target_id)
    emit_diagnostic(
        diagnostic_kind="chat_greeting",
        target_id=target_id,
        target_name=tank["name"],
        message_id=CHAT_HELLO,
    )
    chat = make_chat_command(CHAT_HELLO, ctx.self_state["x"], ctx.self_state["y"])
    return make_tick_decision(
        command=decision["command"],
        behavior=decision["behavior"],
        updated_ai_state=AIStateDict(**{**state, "greeted_target_id": target_id}),
        desired_equipment=decision["desired_equipment"],
        secondary_command=chat,
    )


__all__ = [
    "attach_human_greeting",
]
