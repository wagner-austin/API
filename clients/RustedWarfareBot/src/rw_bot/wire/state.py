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

_UNKNOWN_KIND = "RW-WIRE-001"
_RECORD_BEFORE_FRAME = "RW-WIRE-002"
_COUNT_MISMATCH = "RW-WIRE-003"
_FRAME_MISMATCH = "RW-WIRE-004"
_POOL_COUNT_MISMATCH = "RW-WIRE-005"


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
        hp: Current health.
        max_hp: Health at full.
    """

    index: int
    unit_id: int
    type_name: str
    class_name: str
    x: float
    y: float
    team: int
    mine: bool
    hp: float
    max_hp: float


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
    """

    index: int
    tile_x: int
    tile_y: int
    x: float
    y: float


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
    """

    frame: int
    clock_ms: int
    credits: int
    entities: tuple[Entity, ...]
    pools: tuple[ResourcePool, ...]


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
    credits: int = 0
    entities: list[Entity] = []
    pools: list[ResourcePool] = []
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
                        entities,
                        pools,
                    )
                )
            frame = require_int(record, "frame")
            clock_ms = require_int(record, "clock_ms")
            declared_entities = require_int(record, "visible")
            declared_pools = require_int(record, "pools")
            credits = require_int(record, "credits")
            entities = []
            pools = []
            started = True
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
                    hp=require_finite_float(record, "hp"),
                    max_hp=require_finite_float(record, "max_hp"),
                )
            )
            continue

        raise WireError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")

    if started:
        samples.append(
            _close(frame, clock_ms, credits, declared_entities, declared_pools, entities, pools)
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
    entities: list[Entity],
    pools: list[ResourcePool],
) -> Sample:
    """Finish a sample, checking it against its own declared counts.

    Args:
        frame: The sample's frame counter.
        clock_ms: The sample's millisecond clock.
        credits: The sample's credit balance.
        declared_entities: The entity count the frame record promised.
        declared_pools: The pool count the frame record promised.
        entities: The entity records actually seen.
        pools: The pool records actually seen.

    Returns:
        The completed sample.

    Raises:
        WireError: ``RW-WIRE-003`` when the entity counts disagree,
            ``RW-WIRE-005`` when the pool counts do.
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
    return Sample(
        frame=frame,
        clock_ms=clock_ms,
        credits=credits,
        entities=tuple(entities),
        pools=tuple(pools),
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
        f'"pools":{len(sample["pools"])},'
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
            f'"hp":{entity["hp"]!r},"max_hp":{entity["max_hp"]!r}}}'
        )
    for pool in sample["pools"]:
        lines.append(
            f'{{"kind":"{KIND_POOL}","frame":{frame},"index":{pool["index"]},'
            f'"tile_x":{pool["tile_x"]},"tile_y":{pool["tile_y"]},'
            f'"x":{pool["x"]!r},"y":{pool["y"]!r}}}'
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
    "KIND_POOL",
    "Entity",
    "ResourcePool",
    "Sample",
    "WireError",
    "decode_samples",
    "encode_sample",
]
