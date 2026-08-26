"""Typed message builders for the BotScenario harness.

Each function returns a real :data:`tankpit_bot.protocol.BinaryMessage`
TypedDict ready to feed into :meth:`BotScenario.ingest`. The
constructors carry sensible defaults so scenarios can be terse, but
every wire field is overridable.

This module is the canonical place to add helpers for new message
kinds. Test files MUST NOT inline message construction with magic
numbers -- if you find yourself writing ``MovementResponseDict(...)``
in a test, lift that into a builder here so other tests can reuse it
and the wire-protocol surface stays small and inspectable.
"""

from __future__ import annotations

from tankpit_bot.protocol import (
    ChatMessageDict,
    DeactivationDict,
    EnemyDetectionDict,
    InventoryDict,
    MovementResponseDict,
    RadarResultDict,
    ShootEventDict,
    TankExitDict,
    TankInfoDict,
    TankRemoveDict,
    TankStatusSyncDict,
    ViewportEntityDict,
    ViewportUpdateDict,
)

#: Default weapon byte for ``shot()`` helper -- dual shot, the bot's
#: combat default at non-zero ranges.
DEFAULT_WEAPON_DUAL: int = 1

#: Default team for ``enemy()`` and friends. Differs from
#: :data:`tests.scenarios._harness.DEFAULT_SELF_TEAM` so the
#: enemy-vs-self check fires.
DEFAULT_ENEMY_TEAM: int = 1


def tank_info(
    tank_id: int,
    name: str,
    team: int = DEFAULT_ENEMY_TEAM,
    persistent_tank_id: int = 0,
    decoration_state: bytes = bytes(4),
) -> TankInfoDict:
    """Build a 0x21 ``TankInfo`` message for the registry.

    Args:
        tank_id: Tank id (must match downstream wire messages).
        name: Player name.
        team: Team id (0-3).
        persistent_tank_id: Cross-session tank identity (``0`` when
            test doesn't care).
        decoration_state: Packed award bytes; undecorated (4 zero bytes) for tests.

    Returns:
        A :class:`TankInfoDict` ready for ``BotScenario.ingest``.
    """
    return TankInfoDict(
        msg_type=0x21,
        tank_id=tank_id,
        team=team,
        name=name,
        decoration_state=decoration_state,
        persistent_tank_id=persistent_tank_id,
    )


def movement_response(
    tank_id: int,
    x: int,
    y: int,
    team: int = DEFAULT_ENEMY_TEAM,
    direction: int = 0,
    damage_state: int = 0,
    rank: int = 1,
    lb_score: int = 0,
    carrying: int = 0,
) -> MovementResponseDict:
    """Build a 0x3D ``MovementResponse`` for tank-position update.

    Args:
        tank_id: Tank id.
        x: Tile X.
        y: Tile Y.
        team: Team id.
        direction: Sprite direction. 0-31 alive; 32-33 corpse.
        damage_state: Damage tier (0-3).
        rank: Military rank.
        lb_score: Leaderboard score.
        carrying: Obstacle-carry flag.

    Returns:
        A :class:`MovementResponseDict` ready for ingestion.
    """
    return MovementResponseDict(
        msg_type=0x3D,
        team=team,
        tank_id=tank_id,
        x=x,
        y=y,
        direction=direction,
        damage_state=damage_state,
        rank=rank,
        lb_score=lb_score,
        carrying=carrying,
    )


def tank_remove(tank_id: int) -> TankRemoveDict:
    """Build a 0x58 ``TankRemove`` -- tracking removal, NOT death.

    The dispatcher treats this as "stop broadcasting per-tank updates".
    For an actual kill, use :func:`deactivation` instead.

    Args:
        tank_id: Tank id to remove from the registry.

    Returns:
        A :class:`TankRemoveDict` ready for ingestion.
    """
    return TankRemoveDict(msg_type=0x58, tank_id=tank_id)


def tank_exit(
    tank_id: int,
    team: int = DEFAULT_ENEMY_TEAM,
    was_silent: bool = False,
    was_eliminated: bool = False,
) -> TankExitDict:
    """Build a 0x29 ``TankExit`` announcement.

    Args:
        tank_id: Tank id that exited.
        team: Team id.
        was_silent: True when the server suppresses the announcement
            text.
        was_eliminated: True when the exit is "eliminated from the
            game" rather than "left".

    Returns:
        A :class:`TankExitDict` ready for ingestion.
    """
    return TankExitDict(
        msg_type=0x29,
        team=team,
        tank_id=tank_id,
        was_silent=was_silent,
        was_eliminated=was_eliminated,
    )


def deactivation(
    victim_id: int,
    killer_id: int,
    promo_eligible: bool = False,
    status: int = 0,
    is_mine_kill: bool = False,
) -> DeactivationDict:
    """Build a 0x41 ``Deactivation`` -- the real kill signal.

    Args:
        victim_id: Tank id that was killed.
        killer_id: Tank id of the killer (encodes mine team if
            ``is_mine_kill``).
        promo_eligible: True when the kill earns the killer extra
            promo points.
        status: Wire status byte.
        is_mine_kill: True when the killer is a mine, not a tank.

    Returns:
        A :class:`DeactivationDict` ready for ingestion.
    """
    return DeactivationDict(
        msg_type=0x41,
        status=status,
        victim_id=victim_id,
        promo_eligible=promo_eligible,
        killer_id=killer_id,
        is_mine_kill=is_mine_kill,
    )


def shoot_event(
    shooter_id: int,
    source_x: int,
    source_y: int,
    target_x: int,
    target_y: int,
    team: int = DEFAULT_ENEMY_TEAM,
    aim_x: int | None = None,
    aim_y: int | None = None,
    weapon: int = DEFAULT_WEAPON_DUAL,
) -> ShootEventDict:
    """Build a 0x53 ``ShootEvent`` -- a shot fired on the wire.

    When ``aim_x`` / ``aim_y`` are ``None`` they default to ``(target_x,
    target_y)`` (straight-shot weapons). Pass them explicitly to test
    homing-redirect cases where aim diverges from impact.

    Args:
        shooter_id: Tank id that fired.
        source_x: Shooter tile X.
        source_y: Shooter tile Y.
        target_x: Resolved impact tile X.
        target_y: Resolved impact tile Y.
        team: Team id of the shooter.
        aim_x: Barrel-aim X (defaults to ``target_x``).
        aim_y: Barrel-aim Y (defaults to ``target_y``).
        weapon: Weapon byte (0=single, 1=dual, 2=missile, 3=homing).

    Returns:
        A :class:`ShootEventDict` ready for ingestion.
    """
    effective_aim_x = target_x if aim_x is None else aim_x
    effective_aim_y = target_y if aim_y is None else aim_y
    return ShootEventDict(
        msg_type=0x53,
        team=team,
        shooter_id=shooter_id,
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
        target_y=target_y,
        aim_x=effective_aim_x,
        aim_y=effective_aim_y,
        weapon=weapon,
    )


def inventory_sync(
    armor_shields: int = 0,
    dual_shots: int = 25,
    missile_shots: int = 0,
    homing_shots: int = 25,
    extra_radars: int = 25,
    armor_enabled: bool = False,
    dual_enabled: bool = True,
    missile_enabled: bool = False,
    homing_enabled: bool = True,
    radar_enabled: bool = True,
    show: bool = True,
    alternate: bool = False,
) -> InventoryDict:
    """Build a 0x49 ``Inventory`` absolute-sync.

    Slot order matches the bot's inventory: armor, dual, missile,
    homing, radar. Keyword-only construction so tests stay readable
    and the slot order can never silently shift.

    Args:
        armor_shields: Armor shield slot count.
        dual_shots: Dual-shot slot count.
        missile_shots: Missile slot count.
        homing_shots: Homing-missile slot count.
        extra_radars: Extra-radar slot count (the bot's radar pool).
        armor_enabled: Armor on/off flag.
        dual_enabled: Dual-shot on/off.
        missile_enabled: Missile on/off.
        homing_enabled: Homing on/off.
        radar_enabled: Radar on/off.
        show: Inventory-display flag.
        alternate: Alternate-display flag.

    Returns:
        An :class:`InventoryDict` ready for ingestion.
    """
    return InventoryDict(
        msg_type=0x49,
        show=show,
        alternate=alternate,
        counts=[
            armor_shields,
            dual_shots,
            missile_shots,
            homing_shots,
            extra_radars,
        ],
        enabled=[
            armor_enabled,
            dual_enabled,
            missile_enabled,
            homing_enabled,
            radar_enabled,
        ],
    )


def viewport_update(
    viewport_left: int,
    viewport_top: int,
    entities: list[ViewportEntityDict] | None = None,
) -> ViewportUpdateDict:
    """Build a 0x5A ``ViewportUpdate`` establishing the viewport origin.

    Every real session receives one on join and on every teleport
    landing — it is the authoritative viewport record consumers like
    the aim clamp and the greeting encounter gate read. An empty
    ``entities`` list is a legal "nothing visible" patch (the
    reset-then-apply sweep clears visible-layer beliefs inside the
    bounds).

    Args:
        viewport_left: Viewport left edge (self x - 8 centers the
            bot, matching the live client).
        viewport_top: Viewport top edge.
        entities: Visible-layer entity entries; ``None`` for an empty
            patch.

    Returns:
        A :class:`ViewportUpdateDict` ready for ingestion.
    """
    return ViewportUpdateDict(
        msg_type=0x5A,
        viewport_left=viewport_left,
        viewport_top=viewport_top,
        entities=entities if entities is not None else [],
    )


def chat_message(
    sender_id: int,
    message_id: int = 41,
    x: int | None = None,
    y: int | None = None,
) -> ChatMessageDict:
    """Build an inbound 0x4D ``ChatMessage`` broadcast.

    A non-self sender lands in ``chat_seen_tank_ids`` — the
    human-consent contract's chat signal (2026-07-30). The default
    ``message_id`` 41 is the HELLO preset the bot itself greets with.

    Args:
        sender_id: Chatting tank's id.
        message_id: Preset chat message ID (E[] table index).
        x: Sender-reported X tile, or ``None`` for a coordinate-less
            frame.
        y: Sender-reported Y tile, or ``None``.

    Returns:
        A :class:`ChatMessageDict` ready for ingestion.
    """
    return ChatMessageDict(
        msg_type=0x4D,
        sender_id=sender_id,
        message_type=message_id,
        x=x,
        y=y,
    )


def self_status_sync(
    fuel: int,
    tank_id: int,
    team: int = 2,
    rank: int = 1,
    damage_state: int = 0,
    lb_score: int = 0,
    promo_state: int = 0,
    promo_bar_lit: bool = True,
) -> TankStatusSyncDict:
    """Build a fuel-bearing 0x2E ``TankStatusSync`` for the self tank.

    The long (fuel-carrying) form is per-recipient — always the self
    tank — and is the wire path that CONFIRMS incoming damage: the
    dispatcher folds the fuel delta into the damage book
    (``confirm_incoming_damage``), which is what feeds the
    engagement-break's measured incoming rate.

    Args:
        fuel: New absolute fuel level.
        tank_id: The self tank's id.
        team: Team id (the wire subtype byte).
        rank: Military rank.
        damage_state: Damage tier (0-3).
        lb_score: Leaderboard score.
        promo_state: Promotion-progress counter.
        promo_bar_lit: The promotion bar's colour byte — lit on
            70,313 of 70,532 archived long-form bodies, so lit is the
            default and the dark form is the one a test opts into.

    Returns:
        A :class:`TankStatusSyncDict` ready for ingestion.
    """
    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=team,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        lb_score=lb_score,
        promo_state=promo_state,
        promo_bar_lit=promo_bar_lit,
        fuel=fuel,
    )


def radar_result(found: bool, detection_type: int = 0) -> RadarResultDict:
    """Build an 0x46 ``RadarResult`` (the boolean detect-found ack).

    Args:
        found: True when the server reports the radar revealed
            something.
        detection_type: Server-side detection-kind byte.

    Returns:
        A :class:`RadarResultDict` ready for ingestion.
    """
    return RadarResultDict(
        msg_type=0x46,
        detection_type=detection_type,
        found=found,
    )


def enemy_detection(
    tank_id: int,
    x: int,
    y: int,
    team: int = DEFAULT_ENEMY_TEAM,
    rank: int = 1,
) -> EnemyDetectionDict:
    """Build an 0x48 ``EnemyDetection`` -- enemy-scan reveal.

    Args:
        tank_id: Enemy tank id revealed by the scan.
        x: Revealed tile X.
        y: Revealed tile Y.
        team: Team id.
        rank: Military rank.

    Returns:
        An :class:`EnemyDetectionDict` ready for ingestion.
    """
    return EnemyDetectionDict(
        msg_type=0x48,
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
    )


__all__ = [
    "DEFAULT_ENEMY_TEAM",
    "DEFAULT_WEAPON_DUAL",
    "chat_message",
    "deactivation",
    "enemy_detection",
    "inventory_sync",
    "movement_response",
    "radar_result",
    "self_status_sync",
    "shoot_event",
    "tank_exit",
    "tank_info",
    "tank_remove",
    "viewport_update",
]
