"""Shared context, inventory, and decision builders for the mode tests."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.tick_loop_types import (
    TickDecisionDict,
    make_tick_decision,
)
from tankpit_bot.bot.types import (
    BotCommand,
)
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
)
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


def _make_ctx(*, fuel: int = 1200, dual_count: int = 30, radar_count: int = 30) -> DecideCtx:
    """Create a focused DecideCtx for durable mode tests.

    Args:
        fuel: Current fuel amount.
        dual_count: Dual-shot count.
        radar_count: Extra-radar count.

    Returns:
        Decision context for testing mode predicates.
    """
    world, self_state = make_world(fuel=fuel)
    ai_state = make_scanned_ai_state()
    inventory = make_inventory(default_count=30, dual_count=dual_count)
    inventory["extra_radars"]["count"] = radar_count
    return DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")


def _make_hold_inventory(
    dual_count: int = 25,
    homing_count: int = 25,
) -> InventoryState:
    """Build an inventory for the hold-decision equipment checks."""
    item = InventoryItem(count=25, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=InventoryItem(count=dual_count, enabled=True),
        missile_shots=item,
        homing_shots=InventoryItem(count=homing_count, enabled=True),
        extra_radars=item,
    )


def _make_decision(
    command: BotCommand,
    ai_state: AIStateDict | None = None,
    *,
    secondary_command: BotCommand | None = None,
) -> TickDecisionDict:
    """Build a minimal :class:`TickDecisionDict` for counter tests.

    Args:
        command: Primary command for the decision.
        ai_state: Optional AI state override; defaults to
            :func:`make_initial_ai_state`.
        secondary_command: Optional secondary command.

    Returns:
        A :class:`TickDecisionDict` suitable for exercising
        :func:`apply_dispatch_counters`.
    """
    state = ai_state if ai_state is not None else make_initial_ai_state()
    return make_tick_decision(
        command=command,
        behavior=make_behavior_score("HUNT", 100, 0, 0, "manual_hold"),
        updated_ai_state=state,
        desired_equipment=[],
        secondary_command=secondary_command,
    )
