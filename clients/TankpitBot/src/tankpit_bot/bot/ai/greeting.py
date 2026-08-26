"""One-shot HELLO greeting on arrival in front of a human.

User ruling 2026-08-14 (superseding the 2026-07-31 greet-from-anywhere
rule, which itself superseded the 2026-07-30 encounter trigger):
"he's supposed to say hello AFTER teleporting to the human, when he's
ready to engage, not way before." The HELLO is the face-to-face
opener of an engagement — it fires on the tick the human stands in
the bot's visible viewport, once per id. The stand-off GREET VISIT
("we want to see them") keeps its own latch (``visited_tank_ids``,
``hunt_acquire._greeting_approach``); the two were briefly one latch
and the first human-opponent sim soak proved that wrong.

Map freshness still bounds the scan (registry ``timestamp_ms`` within
the map-open cooldown), and the viewport gate makes the Yuppler-ghost
class of wasted hellos (2026-07-30: departed players lingering in
MapData) structurally rare — a ghost is never IN the viewport.

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
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


def _nearest_ungreeted_viewport_human(
    ctx: DecideCtx,
    state: AIStateDict,
) -> tuple[int, str]:
    """Find the nearest map-fresh enemy human in view awaiting a hello.

    The ARRIVAL gate (user ruling 2026-08-14, superseding the
    2026-07-31 greet-from-anywhere rule): the hello is the
    face-to-face opener, so a candidate must stand in the visible
    viewport — the tick after the teleport that closed on them, when
    the bot is genuinely ready to engage.

    Args:
        ctx: Decision context (registry lookup + self position).
        state: The decision's updated AI state (greeted latch map).

    Returns:
        ``(tank_id, name)`` of the greeting target, or ``(-1, "")``
        when no candidate qualifies this tick.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
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
        if not (left <= tank["x"] <= right and top <= tank["y"] <= bottom):
            continue
        if str(tank["tank_id"]) in state["greeted_tank_ids"]:
            continue
        if ctx.timestamp_ms - tank["timestamp_ms"] > ctx.config["map_intel_horizon_ms"]:
            continue
        dist = abs(tank["x"] - sx) + abs(tank["y"] - sy)
        if target_id == -1 or dist < best_dist:
            target_id = tank["tank_id"]
            target_name = tank["name"]
            best_dist = dist
    return (target_id, target_name)


def attach_human_greeting(ctx: DecideCtx, decision: TickDecisionDict) -> TickDecisionDict:
    """Attach a one-shot HELLO when an ungreeted human is on the map.

    Scans the registry for alive, map-fresh enemy humans not yet in
    the per-id greeted map, IN THE VISIBLE VIEWPORT, and greets the
    nearest one. The viewport gate is the arrival law (user ruling
    2026-08-14: "he's supposed to say hello AFTER teleporting to the
    human, when he's ready to engage, not way before" — superseding
    the 2026-07-31 greet-from-anywhere ruling): the hello is the
    face-to-face opener of an engagement, so it fires on the tick the
    bot stands in front of them, not from across the map. The
    greeting never displaces a planned secondary command — those
    ticks skip, and the unchanged latch retries on the next tick.

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
    target_id, target_name = _nearest_ungreeted_viewport_human(ctx, state)
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
