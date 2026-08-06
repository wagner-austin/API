"""Committed-intent layer: the collect plan as a first-class object.

The bot re-derives its decision every tick, so any input flicker used
to silently replace the current plan mid-execution — the s8-2 receipt
(run bot-20260730-025337, 03:00:00): an escape hop landed ON its
target and the next derivation selected a fresh teleport TO THE TILE
THE TANK WAS STANDING ON, burning a map-open tick, because nothing
asked whether the committed plan's purpose was already served.

This module is the single owner of collect-plan SEMANTICS. The plan's
persistent representation is the resource lock already carried by
``AIStateDict`` (``resource_target_kind/x/y`` — no new state, no
migration); what lives here is the interpretation:

* :func:`current_collect_plan` — read the held plan, typed.
* :func:`plan_completes_here` — "is the purpose served from where the
  tank stands?" One action (a pickup, or a single blessed-under-fire
  step) finishes it.
* :func:`validate_collect_plan` — per-tick validity against the live
  world (target still exists and is pursuable), run at ``DecideCtx``
  construction.
* :func:`release_collect_plan` — the ONLY sanctioned way cascade code
  drops a plan. Every release emits a ``plan_released`` diagnostic
  with a specific reason code, so plan churn is measurable in the
  events stream instead of silent ([[flag-triage-20260729]] s8-2;
  [[committed-intent]]).

Raw field mutation stays in :mod:`tankpit_bot.bot.ai.context`
(``set_resource_target`` / ``clear_resource_target``); this layer
adds meaning, not a parallel mechanism. Phase 2 extends the same
shape to hunt plans (close/pursuit) per [[committed-intent]].
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.bot.ai.context import clear_resource_target
from tankpit_bot.bot.ai.equipment import is_container_pursuable
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.state.types import WorldStateDict

CollectPlanKind = Literal["fuel", "equipment"]

COLLECT_PLAN_KINDS: tuple[CollectPlanKind, ...] = ("fuel", "equipment")
"""Closed vocabulary of collect-plan kinds.

Matches the resource-lock kinds ``set_resource_target`` writes; the
empty string means "no plan" and never appears inside a plan dict.
"""

PlanReleaseReason = Literal[
    "tank_at_capacity",
    "superior_candidate",
    "not_executable",
    "landing_scan_reset",
    "walk_for_fuel_override",
    "target_gone",
    "target_not_pursuable",
    "kind_invalid",
    "unservable",
]

PLAN_RELEASE_REASONS: tuple[PlanReleaseReason, ...] = (
    "tank_at_capacity",
    "superior_candidate",
    "not_executable",
    "landing_scan_reset",
    "walk_for_fuel_override",
    "target_gone",
    "target_not_pursuable",
    "kind_invalid",
    "unservable",
)
"""Closed vocabulary of plan-release reason codes.

``tank_at_capacity`` is a completion (the fuel plan's purpose is
served); every other code is an invalidation. ``unservable`` is the
structural release (2026-08-05): the locked container has no legal
teleport landing AND no fresh ferry on its own water body, so no
lane — walk, hop, or ride — can ever serve it and no move-failed
mark will ever arrive (run bot-20260804-234008 held such a lock for
11 minutes). The vocabulary is closed on purpose: a new release site
means a new documented reason here, not an invented string — churn
analysis groups the ``plan_released`` events by this field.
"""

PLAN_SERVE_REACH = 1
"""Manhattan distance at or below which a plan completes from here.

The established auto-pick convention ([[fuel-system]]: pickup lands
ON or cardinally adjacent; the displaced-landing gate uses the same
``lock_dist <= 1``). At this range the lock continuation is a single
action — a pickup dispatch, or the one blessed-under-fire step (user
movement law 2026-07-30: "walking is 1 tick ... you only take one
hit") — so re-deriving instead of continuing can only waste the tick
the s8-2 receipt paid.
"""


class CollectPlanDict(TypedDict):
    """One committed collect plan: harvest a specific container.

    Attributes:
        kind: Resource kind, ``"fuel"`` or ``"equipment"``.
        target_x: Target container X.
        target_y: Target container Y.
    """

    kind: CollectPlanKind
    target_x: int
    target_y: int


def _require_plan_kind(data: JSONObject, key: str) -> CollectPlanKind:
    """Validate and return a collect-plan kind field.

    Args:
        data: JSON object holding the field.
        key: Field name to read.

    Returns:
        The validated plan kind.

    Raises:
        JSONTypeError: If the value is not a supported plan kind.
    """
    raw = require_str(data, key)
    for kind in COLLECT_PLAN_KINDS:
        if raw == kind:
            return kind
    raise JSONTypeError(f"{key} must be one of {COLLECT_PLAN_KINDS}, got {raw!r}")


def encode_collect_plan(plan: CollectPlanDict) -> JSONObject:
    """Serialize a collect plan to a JSON object.

    Args:
        plan: Plan to serialize.

    Returns:
        JSON object with the plan's fields.
    """
    return {
        "kind": plan["kind"],
        "target_x": plan["target_x"],
        "target_y": plan["target_y"],
    }


def decode_collect_plan(data: JSONObject) -> CollectPlanDict:
    """Validate and deserialize a collect plan from a JSON object.

    Args:
        data: JSON object with the plan's fields.

    Returns:
        The validated plan.

    Raises:
        JSONTypeError: If a field is missing, mistyped, or the kind is
            not in :data:`COLLECT_PLAN_KINDS`.
    """
    return CollectPlanDict(
        kind=_require_plan_kind(data, "kind"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
    )


def current_collect_plan(ai_state: AIStateDict) -> CollectPlanDict | None:
    """Read the committed collect plan held in the AI state.

    Args:
        ai_state: AI state carrying the persistent lock fields.

    Returns:
        The held plan, or ``None`` when no resource lock is held. A
        lock kind outside the closed vocabulary also reads as ``None``
        — :func:`validate_collect_plan` releases it with
        ``kind_invalid`` at context construction, so cascade code
        never sees one.
    """
    kind = ai_state["resource_target_kind"]
    for plan_kind in COLLECT_PLAN_KINDS:
        if kind == plan_kind:
            return CollectPlanDict(
                kind=plan_kind,
                target_x=ai_state["resource_target_x"],
                target_y=ai_state["resource_target_y"],
            )
    return None


def plan_completes_here(plan: CollectPlanDict, self_x: int, self_y: int) -> bool:
    """Return True when the plan's purpose is served from this tile.

    Args:
        plan: The committed plan.
        self_x: Bot X.
        self_y: Bot Y.

    Returns:
        True when the target is within :data:`PLAN_SERVE_REACH` — the
        continuation is a single action, so the plan must be finished,
        never re-derived away (s8-2, [[flag-triage-20260729]]).
    """
    distance = abs(plan["target_x"] - self_x) + abs(plan["target_y"] - self_y)
    return distance <= PLAN_SERVE_REACH


def release_collect_plan(
    ai_state: AIStateDict,
    *,
    reason: PlanReleaseReason,
) -> AIStateDict:
    """Drop any held collect plan, making the release visible.

    The single sanctioned release path: every held plan that gets
    dropped emits a ``plan_released`` diagnostic naming the plan and
    the specific reason, so plan churn shows up in the events stream
    instead of vanishing silently (the pre-intent releases were
    log-text only). Releasing when no plan is held is a no-op by
    SEMANTICS, not a fallback: sites like the landing-scan reset
    clear unconditionally, and "abandoning nothing" is nothing.

    Args:
        ai_state: AI state possibly holding a plan.
        reason: Release reason code (:data:`PLAN_RELEASE_REASONS`).

    Returns:
        AI state with the lock fields zeroed (unchanged content when
        no plan was held).
    """
    plan = current_collect_plan(ai_state)
    if plan is None:
        return ai_state
    emit_diagnostic(
        diagnostic_kind="plan_released",
        plan_kind=plan["kind"],
        target_x=plan["target_x"],
        target_y=plan["target_y"],
        reason=reason,
    )
    return clear_resource_target(ai_state)


def validate_collect_plan(
    ai_state: AIStateDict,
    world: WorldStateDict,
) -> AIStateDict:
    """Release a committed plan whose target no longer holds up.

    Runs once per tick at ``DecideCtx`` construction, applying the
    SAME pursuability predicate as candidate selection (kind match,
    failed pickups) so the cascade only ever sees plans it could have
    selected. A drained or vanished container releases the plan so
    the tick re-derives from live registry truth.

    Args:
        ai_state: AI state with the persistent plan fields.
        world: Filtered world state with container positions.

    Returns:
        AI state with a no-longer-valid plan released (reason
        ``target_gone`` / ``target_not_pursuable`` / ``kind_invalid``),
        or unchanged when the plan holds — or when none is held.
    """
    kind = ai_state["resource_target_kind"]
    if kind == "":
        return ai_state
    plan = current_collect_plan(ai_state)
    if plan is None:
        emit_diagnostic(
            diagnostic_kind="plan_released",
            plan_kind=kind,
            target_x=ai_state["resource_target_x"],
            target_y=ai_state["resource_target_y"],
            reason="kind_invalid",
        )
        return clear_resource_target(ai_state)
    target = world["containers"].get(f"{plan['target_x']},{plan['target_y']}")
    if target is None:
        return release_collect_plan(ai_state, reason="target_gone")
    if not is_container_pursuable(target, want_fuel=plan["kind"] == "fuel"):
        return release_collect_plan(ai_state, reason="target_not_pursuable")
    return ai_state


__all__ = [
    "COLLECT_PLAN_KINDS",
    "PLAN_RELEASE_REASONS",
    "PLAN_SERVE_REACH",
    "CollectPlanDict",
    "CollectPlanKind",
    "PlanReleaseReason",
    "current_collect_plan",
    "decode_collect_plan",
    "encode_collect_plan",
    "plan_completes_here",
    "release_collect_plan",
    "validate_collect_plan",
]
