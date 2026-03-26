"""Tick loop orchestrator for the bot.

Runs the sync-decide-execute cycle on each server tick:
1. SYNC: drain JavaScript WebSocket messages to refresh world state.
2. KILLS: drain Deactivation kills and add to AI killed_tank_ids.
3. FEEDBACK: check protocol-level CombatHit for hit/miss detection.
4. DECIDE: run the AI strategy to pick a command and equipment config.
5. EXECUTE: apply equipment toggles and dispatch the command.
6. WAIT: sleep for one server tick (TICK_RATE_MS).
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.bot import ai_strategy, executor, world_sync
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.states import transition_to
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sniffer.world_state import (
    check_and_clear_combat_hit,
    drain_killed_tank_ids,
    get_inventory_state,
    get_terrain_map,
)

log = get_logger(__name__)


def run_tick_loop(bot: Bot, page: PageProtocol) -> None:
    """Run the main tick loop: sync, decide, execute, repeating forever.

    Args:
        bot: Bot instance.
        page: Playwright page for waiting between ticks.
    """
    world_sync.install_ws_hook(bot)

    while True:
        _tick_once(bot)
        page.wait_for_timeout(TICK_RATE_MS)


def _tick_once(bot: Bot) -> None:
    """Execute one sync-decide-execute cycle.

    Args:
        bot: Bot instance.
    """
    # 1. SYNC: drain JS messages to refresh world state
    world_sync.drain_js_messages(bot)

    # 2. Close map if it was opened last tick (miss→map_open→close sequence)
    if bot._map_is_open:
        bot.close_map()

    # 3. Read current state
    world = bot.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        return

    # Reset to IDLE so the AI can take fresh decisions
    bot._state_data = transition_to(bot._state_data, "IDLE")

    # 4. KILLS: merge protocol-level kills into AI killed_tank_ids.
    # Corpses stay at their death position, so we filter by tank ID.
    bot._ai_state = _merge_protocol_kills(bot._ai_state)

    # 5. FEEDBACK: check protocol-level combat outcome
    combat_feedback = _get_combat_feedback(bot)

    inventory = get_inventory_state()
    terrain = get_terrain_map()
    now = get_current_time_ms()

    # 5. DECIDE: run AI strategy with combat feedback
    decision = ai_strategy.decide(
        world,
        self_state,
        bot._ai_state,
        inventory,
        now,
        terrain,
        combat_feedback,
    )

    # 6. Persist updated AI state
    bot._ai_state = decision["updated_ai_state"]

    # 7. EXECUTE: apply equipment + dispatch command
    executor.execute(bot, decision)


def _merge_protocol_kills(ai_state: AIStateDict) -> AIStateDict:
    """Merge Deactivation kills from the protocol into AI killed_tank_ids.

    The sniffer tracks kills via the Deactivation protocol message.
    This drains those kills and adds them to the AI's killed_tank_ids
    dict so the evaluators filter out dead tanks (corpses stay at
    their death position on the map).

    Args:
        ai_state: Current AI state.

    Returns:
        Updated AI state with new kills merged, or unchanged if none.
    """
    new_kills = drain_killed_tank_ids()
    if not new_kills:
        return ai_state
    now = get_current_time_ms()
    merged = dict(ai_state["killed_tank_ids"])
    for tank_id in new_kills:
        merged[str(tank_id)] = now
        log.info("AI: kill registered from protocol (tank_id=%d)", tank_id)
    return AIStateDict(**{**ai_state, "killed_tank_ids": merged})


def _get_combat_feedback(bot: Bot) -> CombatFeedback:
    """Get combat feedback from protocol messages.

    Uses the sniffer's CombatHit tracking (set by protocol decoder when
    we receive a CombatHit where we are the attacker).

    Only checks when a shot was fired last tick (last_shot_target_id != -1).

    Args:
        bot: Bot instance with AI state.

    Returns:
        Combat feedback: "hit" if shot connected, "miss" if not, "" if
        no shot was pending.
    """
    if bot._ai_state["last_shot_target_id"] == -1:
        return ""
    got_hit = check_and_clear_combat_hit()
    if got_hit:
        return "hit"
    return "miss"


__all__ = [
    "run_tick_loop",
]
