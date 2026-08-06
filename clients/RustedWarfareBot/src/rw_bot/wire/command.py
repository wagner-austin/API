"""Orders the planner sends to the agent.

The outbound half of the wire contract. Each order is one flat JSON object on
one line, in exactly the shape ``rwbot.agent.CommandRecord`` accepts — the
agent rejects anything else loudly rather than skipping it, so this encoder
exists to make malformed lines unrepresentable rather than merely unlikely.

Seven order verbs are defined here, of the eight lines the agent can
dispatch: move a unit to a world position, attack-move it there so it
engages what it meets, attack a named target, have a builder place a
structure, have a building produce a unit, fire a unit's own action by its
interned key, and fire such an action at a chosen ground point. Every one
addresses a unit by its engine identity, never by roster position — position
renumbers whenever anything is built or dies. The eighth line, the reflex
posture, addresses a *type* rather than a unit, which is why it lives in
:mod:`rw_bot.wire.posture` rather than among the orders.

Placing and producing are separate verbs because the engine keeps them
separate. A structure goes where the planner chooses and travels there as a
build waypoint; a unit rolls out of the building that made it and is dispatched
by the action's own key, with no position to carry
([[mechanics-build-actions]]).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from rw_bot import RwBotError

_BLANK_TYPE = "RW-CMD-001"
_NOT_FINITE = "RW-CMD-002"
_SELF_TARGET = "RW-CMD-003"
_BAD_ACTION = "RW-CMD-004"


class CommandError(RwBotError):
    """An order could not be encoded because it is not well formed.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending field.
    """


class MoveOrder(TypedDict):
    """Send one unit to a world position.

    Attributes:
        kind: Discriminator, always ``"move"``.
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.
    """

    kind: Literal["move"]
    unit_id: int
    x: float
    y: float


class AttackMoveOrder(TypedDict):
    """Send one unit to a world position, engaging whatever it meets en route.

    The verb every raid and forward rally needs: a plain move walks a unit
    *past* enemies to its point, and an attack names one target and nothing
    else. The engine's own double-right-click encodes this as a move command
    with one flag set ([[community-play-strategies]]).

    Attributes:
        kind: Discriminator, always ``"attack_move"``.
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.
    """

    kind: Literal["attack_move"]
    unit_id: int
    x: float
    y: float


class BuildOrder(TypedDict):
    """Have one builder place a structure at a world position.

    Attributes:
        kind: Discriminator, always ``"build"``.
        unit_id: Engine identity of the builder.
        type_name: Registry name of the structure, e.g. ``"landFactory"``.
        x: Placement world x.
        y: Placement world y.
    """

    kind: Literal["build"]
    unit_id: int
    type_name: str
    x: float
    y: float


class ProduceOrder(TypedDict):
    """Have one building produce a unit.

    Carries no position. The unit appears at the building that made it, so
    there is nothing for the planner to choose, and the agent rejects a produce
    order that carries a coordinate rather than ignoring it.

    Attributes:
        kind: Discriminator, always ``"produce"``.
        unit_id: Engine identity of the producing building.
        type_name: Registry name of the unit, e.g. ``"c_tank"``.
    """

    kind: Literal["produce"]
    unit_id: int
    type_name: str


class AttackOrder(TypedDict):
    """Send one unit to attack another.

    Attributes:
        kind: Discriminator, always ``"attack"``.
        unit_id: Engine identity of the attacker.
        target_id: Engine identity of the unit to attack.
    """

    kind: Literal["attack"]
    unit_id: int
    target_id: int


class AbilityOrder(TypedDict):
    """Have one unit use an action that concerns no type.

    The verb the tech tree was waiting for. The land factory's tier-two
    upgrade converts into nothing -- it flips a flag on the same building
    and unlocks the heavy roster -- so it cannot be named by ``produce``'s
    type or placed by ``build``'s position. It arrives on the option stream
    as ``produces:""`` with the engine's interned key name, and this order
    fires that key back. It fired the engine's per-action index before, and
    the index is not a selector: every action on a unit answers the same
    figure, so four probes running the "unlock" dispatched was the rally
    point ([[mechanics-build-actions]]).

    Attributes:
        kind: Discriminator, always ``"ability"``.
        unit_id: Engine identity of the unit whose action it is.
        key: The action's interned key name, exactly as the option carried
            it (``c_2``, ``u_builder``, ...).
    """

    kind: Literal["ability"]
    unit_id: int
    key: str


class TargetedAbilityOrder(TypedDict):
    """Have one unit fire an action at a chosen ground point.

    The finisher's verb. The nuke launcher's launch is declared
    ``fireTurretXAtGround``: the engine aims the silo turret at the point
    the command carries, so the point is the whole decision -- and the
    plain ability verb can never express it, because that verb sends the
    unit's own position, which for a production action nothing reads
    ([[mechanics-build-actions]]). Same dispatch, same key, with the point
    chosen by the planner instead of defaulted by the agent.

    A verb of its own rather than an optional field on ``ability``,
    matching the wire's rule that every verb carries exactly its own
    fields: an ability that takes no point refuses one, and an ability
    fired at the ground cannot leave the point out.

    Attributes:
        kind: Discriminator, always ``"ability_at"``.
        unit_id: Engine identity of the unit whose action it is.
        key: The action's interned key name, exactly as the option carried
            it.
        x: Target world x.
        y: Target world y.
    """

    kind: Literal["ability_at"]
    unit_id: int
    key: str
    x: float
    y: float


def attack_order(*, unit_id: int, target_id: int) -> AttackOrder:
    """Build a validated attack order.

    Carries no position, and that is the point of the verb rather than an
    omission. A move sends a unit to where the target stood; this names the
    target itself, so the engine follows it as it moves.

    Args:
        unit_id: Engine identity of the attacker.
        target_id: Engine identity of the unit to attack.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-003`` when a unit is ordered to attack itself,
            which the engine accepts and then cannot act on.
    """
    if unit_id == target_id:
        raise CommandError(
            _SELF_TARGET,
            f"unit {unit_id} cannot be ordered to attack itself",
        )
    return AttackOrder(kind="attack", unit_id=unit_id, target_id=target_id)


def ability_order(*, unit_id: int, key: str) -> AbilityOrder:
    """Build a validated ability order.

    Args:
        unit_id: Engine identity of the unit whose action it is.
        key: The action's interned key name, from the option stream.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-004`` when the key is blank or carries a
            character the flat wire format cannot -- a keyless action cannot
            be dispatched, and the agent would refuse the order.
    """
    _require_action_key(key)
    return AbilityOrder(kind="ability", unit_id=unit_id, key=key)


def targeted_ability_order(*, unit_id: int, key: str, x: float, y: float) -> TargetedAbilityOrder:
    """Build a validated targeted ability order.

    Args:
        unit_id: Engine identity of the unit whose action it is.
        key: The action's interned key name, from the option stream.
        x: Target world x.
        y: Target world y.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-004`` when the key is blank or carries a
            character the flat wire format cannot, ``RW-CMD-002`` when a
            coordinate is not finite.
    """
    _require_action_key(key)
    require_finite(x, "x")
    require_finite(y, "y")
    return TargetedAbilityOrder(kind="ability_at", unit_id=unit_id, key=key, x=x, y=y)


def encode_ability(order: AbilityOrder) -> str:
    """Render an ability order as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return f'{{"kind":"ability","unit_id":{order["unit_id"]},"key":"{order["key"]}"}}'


def encode_targeted_ability(order: TargetedAbilityOrder) -> str:
    """Render a targeted ability order as one wire line.

    The key is emitted unescaped, which is safe because
    :func:`targeted_ability_order` already refused any key the flat format
    cannot carry.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return (
        f'{{"kind":"ability_at","unit_id":{order["unit_id"]},'
        f'"x":{order["x"]!r},"y":{order["y"]!r},"key":"{order["key"]}"}}'
    )


def encode_ack() -> str:
    """Render the sample acknowledgement.

    Not an order, and carries no subject: it is about the exchange rather than
    about a unit. In lockstep the agent holds the simulation after each sample
    until this arrives, so that the planner's think time cannot decide which
    frame its orders land on ([[policy-determinism]]).

    Returns:
        One JSON object, without a trailing newline.
    """
    return '{"kind":"ack"}'


def encode_attack(order: AttackOrder) -> str:
    """Render an attack order as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return f'{{"kind":"attack","unit_id":{order["unit_id"]},"target_id":{order["target_id"]}}}'


def move_order(*, unit_id: int, x: float, y: float) -> MoveOrder:
    """Build a validated move order.

    Args:
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-002`` when a coordinate is not finite.
    """
    require_finite(x, "x")
    require_finite(y, "y")
    return MoveOrder(kind="move", unit_id=unit_id, x=x, y=y)


def attack_move_order(*, unit_id: int, x: float, y: float) -> AttackMoveOrder:
    """Build a validated attack-move order.

    Args:
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-002`` when a coordinate is not finite.
    """
    require_finite(x, "x")
    require_finite(y, "y")
    return AttackMoveOrder(kind="attack_move", unit_id=unit_id, x=x, y=y)


def encode_attack_move(order: AttackMoveOrder) -> str:
    """Render an attack-move order as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return (
        f'{{"kind":"attack_move","unit_id":{order["unit_id"]},'
        f'"x":{order["x"]!r},"y":{order["y"]!r}}}'
    )


def build_order(*, unit_id: int, type_name: str, x: float, y: float) -> BuildOrder:
    """Build a validated build order.

    Args:
        unit_id: Engine identity of the builder.
        type_name: Registry name of the structure to place.
        x: Placement world x.
        y: Placement world y.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-001`` when the type name is blank or carries a
            character the wire format cannot, ``RW-CMD-002`` when a coordinate
            is not finite.
    """
    require_type_name(type_name, "build")
    require_finite(x, "x")
    require_finite(y, "y")
    return BuildOrder(kind="build", unit_id=unit_id, type_name=type_name, x=x, y=y)


def produce_order(*, unit_id: int, type_name: str) -> ProduceOrder:
    """Build a validated produce order.

    Args:
        unit_id: Engine identity of the producing building.
        type_name: Registry name of the unit to produce.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-001`` when the type name is blank or carries a
            character the wire format cannot.
    """
    require_type_name(type_name, "produce")
    return ProduceOrder(kind="produce", unit_id=unit_id, type_name=type_name)


def encode_move(order: MoveOrder) -> str:
    """Render a move order as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return f'{{"kind":"move","unit_id":{order["unit_id"]},"x":{order["x"]!r},"y":{order["y"]!r}}}'


def encode_build(order: BuildOrder) -> str:
    """Render a build order as one wire line.

    The type name is emitted unescaped, which is safe because
    :func:`build_order` already refused any name the flat format cannot carry.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return (
        f'{{"kind":"build","unit_id":{order["unit_id"]},'
        f'"x":{order["x"]!r},"y":{order["y"]!r},"type":"{order["type_name"]}"}}'
    )


def encode_produce(order: ProduceOrder) -> str:
    """Render a produce order as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return f'{{"kind":"produce","unit_id":{order["unit_id"]},"type":"{order["type_name"]}"}}'


def require_type_name(type_name: str, verb: str) -> None:
    """Reject a unit-type name the flat wire format cannot carry.

    Checked when the order is built rather than when it is encoded, so a
    malformed order cannot exist to be sent. Registry names come from the
    engine's own vocabulary and contain none of these characters; one that did
    would otherwise produce a line the agent's strict parser rejects at the far
    end of a socket, rather than here.

    Args:
        type_name: The name to check.
        verb: Order verb, for the message.

    Raises:
        CommandError: ``RW-CMD-001`` when the name is blank or unencodable.
    """
    if type_name.strip() == "":
        raise CommandError(_BLANK_TYPE, f"a {verb} order needs a unit-type name")
    for forbidden in ('"', "\\", "\n", "\r"):
        if forbidden in type_name:
            raise CommandError(
                _BLANK_TYPE,
                f"unit-type name {type_name!r} contains a character the wire format does not carry",
            )


def _require_action_key(key: str) -> None:
    """Reject an action key the flat wire format cannot carry.

    Shared by the two ability verbs, so a key is legal on one exactly when
    it is legal on the other.

    Args:
        key: The action's interned key name.

    Raises:
        CommandError: ``RW-CMD-004`` when the key is blank or unencodable.
    """
    if key.strip() == "":
        raise CommandError(
            _BAD_ACTION,
            "an ability order needs the option stream's action key",
        )
    for forbidden in ('"', "\\", "\n", "\r"):
        if forbidden in key:
            raise CommandError(
                _BAD_ACTION,
                f"action key {key!r} contains a character the wire format does not carry",
            )


def require_finite(value: float, field: str) -> None:
    """Reject a coordinate JSON cannot carry.

    Args:
        value: The coordinate.
        field: Field name, for the message.

    Raises:
        CommandError: ``RW-CMD-002`` when the value is not finite.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise CommandError(_NOT_FINITE, f"order field {field!r} must be finite, got {value}")


__all__ = [
    "AbilityOrder",
    "AttackMoveOrder",
    "AttackOrder",
    "BuildOrder",
    "CommandError",
    "MoveOrder",
    "ProduceOrder",
    "TargetedAbilityOrder",
    "ability_order",
    "attack_move_order",
    "attack_order",
    "build_order",
    "encode_ability",
    "encode_ack",
    "encode_attack",
    "encode_attack_move",
    "encode_build",
    "encode_move",
    "encode_produce",
    "encode_targeted_ability",
    "move_order",
    "produce_order",
    "require_finite",
    "require_type_name",
    "targeted_ability_order",
]
