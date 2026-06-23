"""Universal decision invariants.

Each ``check_*`` function inspects one :class:`TickDecisionDict` and
returns an :class:`InvariantViolation` when the decision violates a
property no bot may ever produce. These properties are CORRECTNESS
gates -- they catch decisions that are definitionally wrong (target
self, radar with zero inventory, teleport off the map), not decisions
that are merely poor strategy. Strategic-quality checks live in
:mod:`tests.scenarios._strategic_invariants`.

Invariant functions are intentionally pure: they take a decision plus
the state it was produced from, and return either ``None`` (passes)
or an :class:`InvariantViolation` (fails). Tests run every invariant
on every decision they produce; corpus replay applies them on every
tick of every captured session.

Adding a new invariant: write one ``check_*`` function and append it
to :func:`check_all_universal_invariants`. The :class:`InvariantName`
literal union grows by one entry so each violation carries a stable,
machine-readable name. Failures include a descriptive ``detail``
string so triage agents (human or AI) can pinpoint the cause without
re-running the case.

Properties that the production type system already proves
unreachable (e.g. unknown ``cmd_type`` or ``behavior.mode``
strings -- both are :data:`Literal` unions) are deliberately NOT
included here: a runtime check that can never fire is dead code by
the project's strictness rules.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.inventory import InventoryState
from tankpit_bot.state.types import SelfStateDict, WorldStateDict


def _target_coords(command: BotCommand) -> tuple[int, int] | None:
    """Return ``(target_x, target_y)`` for commands that carry them.

    Args:
        command: Any :data:`BotCommand` variant.

    Returns:
        ``(target_x, target_y)`` when the command kind is ``shoot``,
        ``teleport``, ``move``, ``pickup_fuel``, or
        ``pickup_equipment`` (the variants that declare coordinate
        fields). ``None`` for ``radar`` / ``map_open``, which carry
        no target.
    """
    match command:
        case {"cmd_type": "shoot", "target_x": int(x), "target_y": int(y)}:
            return (x, y)
        case {"cmd_type": "teleport", "target_x": int(x), "target_y": int(y)}:
            return (x, y)
        case {"cmd_type": "move", "target_x": int(x), "target_y": int(y)}:
            return (x, y)
        case {"cmd_type": "pickup_fuel", "target_x": int(x), "target_y": int(y)}:
            return (x, y)
        case {"cmd_type": "pickup_equipment", "target_x": int(x), "target_y": int(y)}:
            return (x, y)
    return None


#: Map tiles are 0-255 on each axis. Decisions targeting outside this
#: range cannot be executed.
TILE_MIN: int = 0
TILE_MAX: int = 255


InvariantName = Literal[
    "decision_target_on_map",
    "decision_does_not_target_self",
    "decision_does_not_teleport_to_origin_sentinel",
    "decision_secondary_does_not_duplicate_primary",
]


class InvariantViolation(TypedDict):
    """Structured record of an invariant failure.

    Attributes:
        invariant: Stable name of the invariant the decision failed.
        detail: Human-readable explanation, including the offending
            field values, that lets a triage agent (human or AI)
            pinpoint the failure without re-running the case.
    """

    invariant: InvariantName
    detail: str


def _make_violation(invariant: InvariantName, detail: str) -> InvariantViolation:
    """Construct a typed :class:`InvariantViolation`.

    Args:
        invariant: Stable name of the invariant that failed.
        detail: Human-readable failure description.

    Returns:
        A fully-typed :class:`InvariantViolation`.
    """
    return InvariantViolation(invariant=invariant, detail=detail)


def check_target_on_map(decision: TickDecisionDict) -> InvariantViolation | None:
    """Reject decisions whose target coordinates fall off the 256x256 map.

    Applies to commands that carry ``target_x`` / ``target_y``
    (``shoot``, ``teleport``, ``move``). Commands without coordinates
    (``radar``, ``map_open``, ``none``) pass unconditionally.

    Args:
        decision: Decision under test.

    Returns:
        ``None`` when target is on-map or absent, otherwise an
        :class:`InvariantViolation` with the offending coordinates.
    """
    coords = _target_coords(decision["command"])
    if coords is None:
        return None
    target_x, target_y = coords
    if not (TILE_MIN <= target_x <= TILE_MAX and TILE_MIN <= target_y <= TILE_MAX):
        return _make_violation(
            "decision_target_on_map",
            f"target_x={target_x}, target_y={target_y} is off the {TILE_MIN}..{TILE_MAX} map",
        )
    return None


def check_does_not_target_self(
    decision: TickDecisionDict,
    self_state: SelfStateDict,
) -> InvariantViolation | None:
    """Reject decisions whose ``shoot`` command lands on the bot's own tile.

    Self-fire would deduct fuel and produce no benefit. Only checked
    for ``shoot`` because ``teleport``/``move`` to own tile is a
    legitimate no-op in edge cases and gets filtered downstream.

    Args:
        decision: Decision under test.
        self_state: Bot's own state at decision time.

    Returns:
        ``None`` if the shot doesn't land on self, otherwise an
        :class:`InvariantViolation`.
    """
    match decision["command"]:
        case {"cmd_type": "shoot", "target_x": int(target_x), "target_y": int(target_y)}:
            if target_x == self_state["x"] and target_y == self_state["y"]:
                return _make_violation(
                    "decision_does_not_target_self",
                    f"shoot at ({target_x},{target_y}) which is self's own tile",
                )
    return None


# The ``check_does_not_radar_with_zero_inventory`` invariant was
# removed deliberately. The bot's ``radar`` command in the AI layer is
# the SAME command for both variants -- the server routes it to the
# free built-in 5x5 scan when ``extra_radars=0`` and to the
# inventory-consuming extended scan when ``extra_radars>0``. The
# foraging mode (``bot/ai/forage.py:147``) explicitly logs the 5x5
# fallback and treats it as a valid scan, so a radar command with
# zero extras is correct behaviour, not a violation. Adding it back
# would block the bot's foraging policy.


def check_does_not_teleport_to_origin_sentinel(
    decision: TickDecisionDict,
) -> InvariantViolation | None:
    """Reject ``teleport`` decisions targeting (0, 0) -- the deactivation sentinel.

    The wire format uses ``(0, 0)`` as the default for tanks that have
    no wire-confirmed position. Teleporting there is always a bug.

    Args:
        decision: Decision under test.

    Returns:
        ``None`` when target is not (0,0) or command isn't teleport,
        otherwise an :class:`InvariantViolation`.
    """
    match decision["command"]:
        case {"cmd_type": "teleport", "target_x": int(target_x), "target_y": int(target_y)}:
            if target_x == 0 and target_y == 0:
                return _make_violation(
                    "decision_does_not_teleport_to_origin_sentinel",
                    "teleport target is (0, 0); this is the unsynced-tank "
                    "sentinel and never a valid teleport target",
                )
    return None


def check_secondary_does_not_duplicate_primary(
    decision: TickDecisionDict,
) -> InvariantViolation | None:
    """Reject decisions whose secondary command duplicates the primary.

    The secondary slot is for follow-up actions (e.g. radar after
    teleport). A secondary identical to the primary is always wrong:
    either both fire and one is wasted, or the executor's dedup
    silently drops it.

    Args:
        decision: Decision under test.

    Returns:
        ``None`` when no secondary, secondaries are absent, or
        primary != secondary; otherwise an
        :class:`InvariantViolation`.
    """
    secondary = decision["secondary_command"]
    if secondary is None:
        return None
    primary = decision["command"]
    if primary == secondary:
        return _make_violation(
            "decision_secondary_does_not_duplicate_primary",
            f"secondary_command duplicates primary command {primary}",
        )
    return None


def check_all_universal_invariants(
    decision: TickDecisionDict,
    self_state: SelfStateDict,
    inventory: InventoryState,
) -> list[InvariantViolation]:
    """Run every universal invariant and return all failures.

    Args:
        decision: Decision under test.
        self_state: Bot's own state at decision time (needed for the
            self-target check).
        inventory: Bot's inventory. Accepted for signature symmetry
            with future invariants; no current invariant inspects it.

    Returns:
        List of :class:`InvariantViolation` records; empty when the
        decision passes every check.
    """
    del inventory  # reserved for future inventory-aware invariants
    violations: list[InvariantViolation] = []
    for violation in (
        check_target_on_map(decision),
        check_does_not_target_self(decision, self_state),
        check_does_not_teleport_to_origin_sentinel(decision),
        check_secondary_does_not_duplicate_primary(decision),
    ):
        if violation is not None:
            violations.append(violation)
    return violations


def assert_no_violations(
    decision: TickDecisionDict,
    self_state: SelfStateDict,
    inventory: InventoryState,
    world: WorldStateDict | None = None,
) -> None:
    """Assert the decision satisfies every universal invariant.

    Pytest assertion wrapper around
    :func:`check_all_universal_invariants` for use in scenario
    tests. The ``world`` parameter is accepted for symmetry with
    future invariants that need it, even though no current
    invariant inspects ``world``.

    Args:
        decision: Decision under test.
        self_state: Bot's own state at decision time.
        inventory: Bot's inventory at decision time.
        world: Bot's world state at decision time. Reserved for
            future invariants that need world context (currently
            unused).

    Raises:
        AssertionError: When one or more invariants fail; the message
            enumerates every violation.
    """
    del world  # reserved
    violations = check_all_universal_invariants(decision, self_state, inventory)
    if not violations:
        return
    lines = [f"  - {v['invariant']}: {v['detail']}" for v in violations]
    raise AssertionError("Universal decision invariants failed:\n" + "\n".join(lines))


__all__ = [
    "TILE_MAX",
    "TILE_MIN",
    "InvariantName",
    "InvariantViolation",
    "assert_no_violations",
    "check_all_universal_invariants",
    "check_does_not_target_self",
    "check_does_not_teleport_to_origin_sentinel",
    "check_secondary_does_not_duplicate_primary",
    "check_target_on_map",
]
