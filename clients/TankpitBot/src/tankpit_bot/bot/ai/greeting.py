"""One-shot HELLO greeting for any human known to be on the map.

User ruling 2026-07-31 (superseding the 2026-07-30 encounter trigger):
"hello can run anytime... as long as the other player is on the map
logged in. you dont have to be near them." Chat is global, so the
HELLO fires for any map-fresh enemy human the bot has not yet
greeted, wherever they are — the stand-off GREET VISIT ("we want to
see them") is a separate obligation with its own latch
(``visited_tank_ids``, ``hunt_acquire._greeting_approach``). The two
were briefly one latch, and the first human-opponent sim soak proved
that wrong: an early long-range HELLO burned the shared latch and the
visit never happened.

"On the map logged in" is approximated by map freshness (the registry
``timestamp_ms`` within the map-open cooldown — 0x4C map opens and
the global 0x2E sync refresh it for every tank actually in the game).
A departed player can linger in MapData (the Yuppler-ghost finding,
2026-07-30), so a hello may occasionally go to a ghost — one wasted
chat, bounded by the latch.

Flood-mute discipline ([[chat-messages]], sniff-20260729-214411: after
8 rapid sends the server silently swallowed chat for the rest of the
session): exactly one greeting per greeted tank id, latched in
``ai_state["greeted_tank_ids"]``, never retried on silence.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import make_chat_command
from tankpit_bot.protocol.chat import CHAT_HELLO
from tankpit_bot.protocol.naming import is_human_name
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)


def attach_human_greeting(ctx: DecideCtx, decision: TickDecisionDict) -> TickDecisionDict:
    """Attach a one-shot HELLO when an ungreeted human is on the map.

    Scans the registry for alive, map-fresh enemy humans not yet in
    the per-id greeted map and greets the nearest one —
    distance never gates the hello (chat is global), it only breaks
    ties when several humans are ungreeted at once. The greeting never
    displaces a planned secondary command — those ticks skip, and the
    unchanged latch retries on the next tick.

    Args:
        ctx: Decision context (registry lookup + self position).
        decision: The decision leaving the arbitrator this tick.

    Returns:
        The decision unchanged, or with the HELLO chat attached as
        ``secondary_command`` and the id latched in ``greeted_tank_ids``.
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
        # Deliberately NO position gate here. User ruling 2026-07-31:
        # "hello can run anytime... as long as the other player is on
        # the map logged in. you dont have to be near them." A human
        # still at the login-roster (0, 0) default gets the HELLO the
        # moment their identity broadcast lands; the distance below
        # only orders who is greeted first, and every ungreeted human
        # is greeted eventually. has_known_position gates targeting
        # and the stand-off visit, never the chat.
        if str(tank["tank_id"]) in state["greeted_tank_ids"]:
            continue
        if ctx.timestamp_ms - tank["timestamp_ms"] > ctx.config["map_open_cooldown_ms"]:
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
        updated_ai_state=AIStateDict(
            **{
                **state,
                "greeted_tank_ids": {
                    **state["greeted_tank_ids"],
                    str(target_id): ctx.timestamp_ms,
                },
            }
        ),
        desired_equipment=decision["desired_equipment"],
        secondary_command=chat,
    )


__all__ = [
    "attach_human_greeting",
]
