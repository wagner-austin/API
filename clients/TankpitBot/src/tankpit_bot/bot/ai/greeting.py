"""One-shot HELLO greeting when a human is encountered in the viewport.

User contract (2026-07-30, superseding the 2026-07-29 lock-trigger):
the bot never engages a human who has not consented -- "the human must
respond hello or engage the bot first" -- so the HELLO cannot wait for
a combat lock that the consent gate now forbids. Instead the greeting
fires on ENCOUNTER: any viewport-present enemy human the bot has not
yet greeted gets the chat attached as the tick's
``secondary_command``, whatever the primary decision is. The
greeting-approach step (``hunt_mode._greeting_approach``) teleports
the bot a few tiles off a map-known human precisely so this hook
fires with both tanks in sight of each other.

Flood-mute discipline ([[chat-messages]], sniff-20260729-214411: after
8 rapid sends the server silently swallowed chat for the rest of the
session): exactly one greeting per greeted tank id, latched in
``ai_state["greeted_target_id"]``, never retried on silence.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.humans import is_human_name
from tankpit_bot.bot.ai.threats import VIEWPORT_PRESENCE_TTL_MS
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import make_chat_command
from tankpit_bot.protocol.chat import CHAT_HELLO
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic


def attach_human_greeting(ctx: DecideCtx, decision: TickDecisionDict) -> TickDecisionDict:
    """Attach a one-shot HELLO when an ungreeted human is in view.

    Scans the registry for enemy humans with live viewport presence
    (``last_viewport_observation_ms`` inside the presence TTL -- the
    same proof the threat list demands) whose id differs from the last
    greeted id, and greets the nearest one. The greeting never
    displaces a planned secondary command -- those ticks skip, and the
    unchanged latch retries on the next tick with the human still in
    view.

    Args:
        ctx: Decision context (registry lookup + self position).
        decision: The decision leaving the arbitrator this tick.

    Returns:
        The decision unchanged, or with the HELLO chat attached as
        ``secondary_command`` and ``greeted_target_id`` latched.
    """
    state = decision["updated_ai_state"]
    if decision["secondary_command"] is not None:
        return decision
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    target_id = -1
    target_name = ""
    best_dist = 0
    for tank in ctx.world["tanks"].values():
        if tank["is_self"] or tank["team"] == ctx.self_state["team"]:
            continue
        if tank["liveness"] != "alive":
            continue
        if not is_human_name(tank["name"]):
            continue
        if tank["tank_id"] == state["greeted_target_id"]:
            continue
        if ctx.timestamp_ms - tank["last_viewport_observation_ms"] > VIEWPORT_PRESENCE_TTL_MS:
            continue
        dist = abs(tank["x"] - sx) + abs(tank["y"] - sy)
        if target_id == -1 or dist < best_dist:
            target_id = tank["tank_id"]
            target_name = tank["name"]
            best_dist = dist
    if target_id == -1:
        return decision
    emit_ai("greeting human %s (id=%d) with HELLO", target_name, target_id)
    emit_diagnostic(
        diagnostic_kind="chat_greeting",
        target_id=target_id,
        target_name=target_name,
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
