"""Typed world state decoded from the agent's NDJSON stream.

The stream is a sequence of records discriminated by ``kind``. A ``frame``
record opens a sample and declares how many ``entity`` and ``pool`` records
follow; the entity records carry the visible roster and the pool records the
visible resource pools. Folding them back into whole samples is this module's
job.

The declared counts are checked rather than trusted. A sample that promises
three entities and delivers two is a truncated capture — the ordinary result of
reading a stream while the agent is still writing it — and silently yielding the
short sample would let a planner make decisions on a roster it cannot see all
of.

Nothing here reads a file. Decoding is a pure function of the lines it is
given, which is what lets the same code serve a live tail and an archived
replay corpus without a branch between them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.validation import (
    require_bool,
    require_finite_float,
    require_int,
    require_non_empty_str,
)
from rw_bot.wire.ndjson import parse_object

KIND_FRAME: Final = "frame"
"""``kind`` value opening a sample."""

KIND_ENTITY: Final = "entity"
"""``kind`` value of an entity record inside a sample."""

KIND_POOL: Final = "pool"
"""``kind`` value of a resource-pool record inside a sample."""

KIND_OPTION: Final = "option"
"""``kind`` value of a build-option record inside a sample."""

_UNKNOWN_KIND = "RW-WIRE-001"
_RECORD_BEFORE_FRAME = "RW-WIRE-002"
_COUNT_MISMATCH = "RW-WIRE-003"
_FRAME_MISMATCH = "RW-WIRE-004"
_POOL_COUNT_MISMATCH = "RW-WIRE-005"
_OPTION_COUNT_MISMATCH = "RW-WIRE-006"


class WireError(RwBotError):
    """The record sequence did not form whole samples.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending record.
    """


class Entity(TypedDict):
    """One entity the local player owns at a given frame.

    Attributes:
        index: Position in the owned roster, as the agent enumerated it. Useful
            for reading a single sample and nothing else: it renumbers whenever
            anything is built or dies, so it is not an addressing handle.
        unit_id: The engine's own object identity, assigned once at construction
            and used by the engine for network identity. This is the handle an
            order is dispatched against.
        type_name: Readable unit-type name, e.g. ``"builder"``. The same string
            the type registry accepts when building.
        class_name: Engine class of the entity, obfuscated and pinned to the
            recorded build.
        x: World x coordinate.
        y: World y coordinate.
        team: Owning team number. Present for every visible entity, friend or
            not.
        mine: Whether the local player owns it. The stream carries enemies too,
            so a consumer that skips this check would credit an opponent's
            buildings to itself.
        hostile: Whether the engine considers this entity's owner an enemy of
            the local player. Not the negation of ``mine``: an ally's units are
            neither, and so are the neutral team's. Read from the engine's own
            alliance comparison rather than derived here, because a planner that
            treats every unowned unit as a threat cannot cross its own ally's
            territory.
        movement: Engine name of the layer this entity travels on, e.g.
            ``"LAND"``, ``"AIR"``, ``"HOVER"``. The engine keeps a separate
            connectivity grid per layer, so this says which grid ``group``
            belongs to.
        group: Connectivity component this entity stands in, on its own layer.
            Two things can reach each other exactly when their components match
            and both are non-negative — a negative is the engine's way of
            saying the point has no component at all
            ([[mechanics-movement-layers]]).
        hp: Current health.
        max_hp: Health at full.
        complete: Whether construction has finished. A building joins the roster
            the moment construction starts, so presence is not completion — and
            an unfinished factory never advances its production queue.
        queued: Units this entity has queued for production, zero for anything
            that makes nothing. A production order changes no roster until the
            unit is finished, so this is the only immediate evidence that the
            engine accepted one.
    """

    index: int
    unit_id: int
    type_name: str
    class_name: str
    x: float
    y: float
    team: int
    mine: bool
    hostile: bool
    movement: str
    group: int
    hp: float
    max_hp: float
    complete: bool
    queued: int


class ResourcePool(TypedDict):
    """One resource pool the local player can currently see.

    Pools are terrain rather than units: they appear in no entity list, and a
    planner reading only the roster cannot see them at all. They matter because
    an extractor is the one structure the engine refuses to place anywhere else.

    Both coordinate systems are carried because they answer different
    questions. The tile coordinate identifies the pool — integral, fixed for the
    life of the map, and the unit the engine's own placement check works in. The
    world point is where a build order has to be addressed.

    Attributes:
        index: Position in the sample's pool list. Enumeration order only.
        tile_x: Tile column.
        tile_y: Tile row.
        x: World x of the tile's centre.
        y: World y of the tile's centre.
        group_land: Connectivity component of this tile on the **land** layer,
            or a negative when it has none. Compare against a land unit's
            ``group`` to decide whether it can walk here at all. Land
            specifically: every builder in the base game travels on land, and
            naming the layer keeps a mismatched comparison from looking like an
            answer ([[mechanics-movement-layers]]).
    """

    index: int
    tile_x: int
    tile_y: int
    x: float
    y: float
    group_land: int


class BuildOption(TypedDict):
    """One thing an owned unit can make.

    The engine treats placing a building and producing a unit as the same
    mechanism: a builder's actions yield buildings, a factory's yield units, and
    one command verb dispatches either. What differs is only which unit has the
    action, so this is the table that answers "who can make X" — a question no
    stat dump answers and that the bot has twice guessed wrong.

    Attributes:
        index: Position in the sample's option list. Enumeration order only.
        unit_id: Engine identity of the unit that can make it. This is what an
            order is addressed to, so no second lookup is needed.
        produces: Type name it makes, in the same vocabulary a plan uses.
        action: The engine's selector index for this action. Distinguishes two
            actions on one unit that produce the same type.
        placed: Whether the thing is put at a position the planner chooses. A
            structure is; a unit rolls out of the building that made it. This
            decides which verb orders it, and it is the engine's own
            distinction rather than a guess from the type's speed.
        available: Whether the unit may use it right now. An action that exists
            but is unavailable is a wait; one that does not exist at all is a
            dead plan entry, and the two need different answers.
    """

    index: int
    unit_id: int
    produces: str
    action: int
    placed: bool
    available: bool


class Sample(TypedDict):
    """One coherent observation of the world.

    Attributes:
        frame: The engine's frame counter at the moment of the read.
        clock_ms: The engine's millisecond clock at the same moment.
        credits: The current player's credits, floored to whole currency. The
            engine spends in whole units, so a planner comparing against a unit
            price wants the floor: 99 credits does not buy a 100-credit
            structure.
        entities: Every visible entity, in the order the agent enumerated it.
            Includes entities the local player does not own; check
            :attr:`Entity.mine`.
        pools: Every resource pool currently visible, in scan order. Fog-filtered
            by the engine's own per-tile test, so this grows as the map is
            explored rather than listing the whole map from the first frame.
        options: Everything the player's own units can currently make, one entry
            per producible type per unit.
    """

    frame: int
    clock_ms: int
    credits: int
    entities: tuple[Entity, ...]
    pools: tuple[ResourcePool, ...]
    options: tuple[BuildOption, ...]


def decode_samples(lines: Sequence[str]) -> tuple[Sample, ...]:
    """Fold a run of NDJSON lines into whole samples.

    Blank lines are skipped, which is what a file ending in a newline produces.

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Every complete sample, in stream order.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        WireError: ``RW-WIRE-001`` on an unknown ``kind``, ``RW-WIRE-002`` when
            an entity or pool precedes any frame, ``RW-WIRE-003`` when a
            sample's entity count disagrees with its declared count,
            ``RW-WIRE-004`` when a record's frame disagrees with the sample it
            falls in, ``RW-WIRE-005`` when the pool count disagrees.
    """
    samples: list[Sample] = []
    frame: int = 0
    clock_ms: int = 0
    declared_entities: int = 0
    declared_pools: int = 0
    declared_options: int = 0
    credits: int = 0
    entities: list[Entity] = []
    pools: list[ResourcePool] = []
    options: list[BuildOption] = []
    started = False

    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        kind = require_non_empty_str(record, "kind")

        if kind == KIND_FRAME:
            if started:
                samples.append(
                    _close(
                        frame,
                        clock_ms,
                        credits,
                        declared_entities,
                        declared_pools,
                        declared_options,
                        entities,
                        pools,
                        options,
                    )
                )
            frame = require_int(record, "frame")
            clock_ms = require_int(record, "clock_ms")
            declared_entities = require_int(record, "visible")
            declared_pools = require_int(record, "pools")
            declared_options = require_int(record, "options")
            credits = require_int(record, "credits")
            entities = []
            pools = []
            options = []
            started = True
            continue

        if kind == KIND_OPTION:
            _require_inside_sample(started, kind, frame, record)
            options.append(
                BuildOption(
                    index=require_int(record, "index"),
                    unit_id=require_int(record, "unit_id"),
                    produces=require_non_empty_str(record, "produces"),
                    action=require_int(record, "action"),
                    placed=require_bool(record, "placed"),
                    available=require_bool(record, "available"),
                )
            )
            continue

        if kind == KIND_POOL:
            _require_inside_sample(started, kind, frame, record)
            pools.append(
                ResourcePool(
                    index=require_int(record, "index"),
                    tile_x=require_int(record, "tile_x"),
                    tile_y=require_int(record, "tile_y"),
                    x=require_finite_float(record, "x"),
                    y=require_finite_float(record, "y"),
                    group_land=require_int(record, "group_land"),
                )
            )
            continue

        if kind == KIND_ENTITY:
            _require_inside_sample(started, kind, frame, record)
            entities.append(
                Entity(
                    index=require_int(record, "index"),
                    unit_id=require_int(record, "id"),
                    type_name=require_non_empty_str(record, "type"),
                    class_name=require_non_empty_str(record, "class"),
                    x=require_finite_float(record, "x"),
                    y=require_finite_float(record, "y"),
                    team=require_int(record, "team"),
                    mine=require_bool(record, "mine"),
                    hostile=require_bool(record, "hostile"),
                    movement=require_non_empty_str(record, "movement"),
                    group=require_int(record, "group"),
                    hp=require_finite_float(record, "hp"),
                    max_hp=require_finite_float(record, "max_hp"),
                    complete=require_bool(record, "complete"),
                    queued=require_int(record, "queued"),
                )
            )
            continue

        raise WireError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")

    if started:
        samples.append(
            _close(
                frame,
                clock_ms,
                credits,
                declared_entities,
                declared_pools,
                declared_options,
                entities,
                pools,
                options,
            )
        )
    return tuple(samples)


def _require_inside_sample(
    started: bool,
    kind: str,
    frame: int,
    record: Mapping[str, str | int | float | bool],
) -> None:
    """Check that a record falls inside the sample it claims to.

    Args:
        started: Whether a frame record has opened a sample.
        kind: The record's ``kind``, for the message.
        frame: The open sample's frame counter.
        record: The record being placed.

    Raises:
        WireError: ``RW-WIRE-002`` when no sample is open, ``RW-WIRE-004`` when
            the record's own frame disagrees with the open one.
    """
    if not started:
        raise WireError(
            _RECORD_BEFORE_FRAME,
            f"a {kind} record appeared before any frame record; the stream does "
            "not begin at a sample boundary",
        )
    reported = require_int(record, "frame")
    if reported != frame:
        raise WireError(
            _FRAME_MISMATCH,
            f"{kind} reports frame {reported} inside the sample for frame {frame}; "
            "the records have been interleaved",
        )


def _close(
    frame: int,
    clock_ms: int,
    credits: int,
    declared_entities: int,
    declared_pools: int,
    declared_options: int,
    entities: list[Entity],
    pools: list[ResourcePool],
    options: list[BuildOption],
) -> Sample:
    """Finish a sample, checking it against its own declared counts.

    Args:
        frame: The sample's frame counter.
        clock_ms: The sample's millisecond clock.
        credits: The sample's credit balance.
        declared_entities: The entity count the frame record promised.
        declared_pools: The pool count the frame record promised.
        declared_options: The option count the frame record promised.
        entities: The entity records actually seen.
        pools: The pool records actually seen.
        options: The option records actually seen.

    Returns:
        The completed sample.

    Raises:
        WireError: ``RW-WIRE-003`` when the entity counts disagree,
            ``RW-WIRE-005`` when the pool counts do, ``RW-WIRE-006`` when the
            option counts do.
    """
    if len(entities) != declared_entities:
        raise WireError(
            _COUNT_MISMATCH,
            f"frame {frame} declared {declared_entities} visible entities but carried "
            f"{len(entities)}; the capture is truncated or interleaved",
        )
    if len(pools) != declared_pools:
        raise WireError(
            _POOL_COUNT_MISMATCH,
            f"frame {frame} declared {declared_pools} visible resource pools but carried "
            f"{len(pools)}; the capture is truncated or interleaved",
        )
    if len(options) != declared_options:
        raise WireError(
            _OPTION_COUNT_MISMATCH,
            f"frame {frame} declared {declared_options} build options but carried "
            f"{len(options)}; the capture is truncated or interleaved",
        )
    return Sample(
        frame=frame,
        clock_ms=clock_ms,
        credits=credits,
        entities=tuple(entities),
        pools=tuple(pools),
        options=tuple(options),
    )


def encode_sample(sample: Sample) -> tuple[str, ...]:
    """Render a sample back to NDJSON lines.

    Round-trips with :func:`decode_samples`, which is what makes a decoded
    corpus re-emittable as a fixture.

    Args:
        sample: The sample to encode.

    Returns:
        One frame line, then one line per entity, then one line per pool — the
        order the agent writes them in, which is what makes the round trip
        byte-for-byte rather than merely equivalent.
    """
    frame = sample["frame"]
    lines = [
        f'{{"kind":"{KIND_FRAME}","frame":{frame},'
        f'"clock_ms":{sample["clock_ms"]},"visible":{len(sample["entities"])},'
        f'"pools":{len(sample["pools"])},"options":{len(sample["options"])},'
        f'"credits":{sample["credits"]}}}'
    ]
    for entity in sample["entities"]:
        name = _escape(entity["class_name"])
        type_name = _escape(entity["type_name"])
        lines.append(
            f'{{"kind":"{KIND_ENTITY}","frame":{frame},"index":{entity["index"]},'
            f'"id":{entity["unit_id"]},"type":"{type_name}",'
            f'"class":"{name}","x":{entity["x"]!r},"y":{entity["y"]!r},'
            f'"team":{entity["team"]},"mine":{str(entity["mine"]).lower()},'
            f'"hostile":{str(entity["hostile"]).lower()},'
            f'"movement":"{entity["movement"]}","group":{entity["group"]},'
            f'"hp":{entity["hp"]!r},"max_hp":{entity["max_hp"]!r},'
            f'"complete":{str(entity["complete"]).lower()},'
            f'"queued":{entity["queued"]}}}'
        )
    for pool in sample["pools"]:
        lines.append(
            f'{{"kind":"{KIND_POOL}","frame":{frame},"index":{pool["index"]},'
            f'"tile_x":{pool["tile_x"]},"tile_y":{pool["tile_y"]},'
            f'"x":{pool["x"]!r},"y":{pool["y"]!r},'
            f'"group_land":{pool["group_land"]}}}'
        )
    for option in sample["options"]:
        produces = _escape(option["produces"])
        lines.append(
            f'{{"kind":"{KIND_OPTION}","frame":{frame},"index":{option["index"]},'
            f'"unit_id":{option["unit_id"]},"produces":"{produces}",'
            f'"action":{option["action"]},'
            f'"placed":{str(option["placed"]).lower()},'
            f'"available":{str(option["available"]).lower()}}}'
        )
    return tuple(lines)


def _escape(text: str) -> str:
    """Escape a string for inclusion in a JSON string literal.

    Engine class names contain nothing that needs escaping, so this never fires
    in practice. It is here because :func:`encode_sample` claims to round-trip,
    and a claim that holds only for the values seen so far is not a round trip.

    Args:
        text: The raw string.

    Returns:
        The escaped string, without surrounding quotes.
    """
    out: list[str] = []
    for char in text:
        if char in ('"', "\\"):
            out.append("\\" + char)
        elif char in ("\n", "\r", "\t"):
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)


__all__ = [
    "KIND_ENTITY",
    "KIND_FRAME",
    "KIND_OPTION",
    "KIND_POOL",
    "BuildOption",
    "Entity",
    "ResourcePool",
    "Sample",
    "WireError",
    "decode_samples",
    "encode_sample",
]
