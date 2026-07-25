"""Typed world state decoded from the agent's NDJSON stream.

The stream is a sequence of records discriminated by ``kind``. A ``frame``
record opens a sample and declares how many ``entity`` records follow; those
entity records carry the sample's owned roster. Folding them back into whole
samples is this module's job.

The declared count is checked rather than trusted. A sample that promises three
entities and delivers two is a truncated capture — the ordinary result of
reading a stream while the agent is still writing it — and silently yielding the
short sample would let a planner make decisions on a roster it cannot see all
of.

Nothing here reads a file. Decoding is a pure function of the lines it is
given, which is what lets the same code serve a live tail and an archived
replay corpus without a branch between them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.validation import require_finite_float, require_int, require_non_empty_str
from rw_bot.wire.ndjson import parse_object

KIND_FRAME: Final = "frame"
"""``kind`` value opening a sample."""

KIND_ENTITY: Final = "entity"
"""``kind`` value of a record inside a sample."""

_UNKNOWN_KIND = "RW-WIRE-001"
_ENTITY_BEFORE_FRAME = "RW-WIRE-002"
_COUNT_MISMATCH = "RW-WIRE-003"
_FRAME_MISMATCH = "RW-WIRE-004"


class WireError(RwBotError):
    """The record sequence did not form whole samples.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending record.
    """


class Entity(TypedDict):
    """One entity the local player owns at a given frame.

    Attributes:
        index: Position in the owned roster, as the agent enumerated it. This is
            the handle an order is dispatched against.
        class_name: Engine class of the entity, obfuscated and pinned to the
            recorded build.
        x: World x coordinate.
        y: World y coordinate.
    """

    index: int
    class_name: str
    x: float
    y: float


class Sample(TypedDict):
    """One coherent observation of the world.

    Attributes:
        frame: The engine's frame counter at the moment of the read.
        clock_ms: The engine's millisecond clock at the same moment.
        entities: The owned roster, in the order the agent enumerated it.
    """

    frame: int
    clock_ms: int
    entities: tuple[Entity, ...]


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
            an entity precedes any frame, ``RW-WIRE-003`` when a sample's entity
            count disagrees with its declared count, ``RW-WIRE-004`` when an
            entity's frame disagrees with the sample it falls in.
    """
    samples: list[Sample] = []
    frame: int = 0
    clock_ms: int = 0
    declared: int = 0
    entities: list[Entity] = []
    started = False

    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        kind = require_non_empty_str(record, "kind")

        if kind == KIND_FRAME:
            if started:
                samples.append(_close(frame, clock_ms, declared, entities))
            frame = require_int(record, "frame")
            clock_ms = require_int(record, "clock_ms")
            declared = require_int(record, "owned")
            entities = []
            started = True
            continue

        if kind == KIND_ENTITY:
            if not started:
                raise WireError(
                    _ENTITY_BEFORE_FRAME,
                    "an entity record appeared before any frame record; the stream "
                    "does not begin at a sample boundary",
                )
            entity_frame = require_int(record, "frame")
            if entity_frame != frame:
                raise WireError(
                    _FRAME_MISMATCH,
                    f"entity reports frame {entity_frame} inside the sample for frame "
                    f"{frame}; the records have been interleaved",
                )
            entities.append(
                Entity(
                    index=require_int(record, "index"),
                    class_name=require_non_empty_str(record, "class"),
                    x=require_finite_float(record, "x"),
                    y=require_finite_float(record, "y"),
                )
            )
            continue

        raise WireError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")

    if started:
        samples.append(_close(frame, clock_ms, declared, entities))
    return tuple(samples)


def _close(frame: int, clock_ms: int, declared: int, entities: list[Entity]) -> Sample:
    """Finish a sample, checking it against its own declared count.

    Args:
        frame: The sample's frame counter.
        clock_ms: The sample's millisecond clock.
        declared: The entity count the frame record promised.
        entities: The entity records actually seen.

    Returns:
        The completed sample.

    Raises:
        WireError: ``RW-WIRE-003`` when the counts disagree.
    """
    if len(entities) != declared:
        raise WireError(
            _COUNT_MISMATCH,
            f"frame {frame} declared {declared} owned entities but carried "
            f"{len(entities)}; the capture is truncated or interleaved",
        )
    return Sample(frame=frame, clock_ms=clock_ms, entities=tuple(entities))


def encode_sample(sample: Sample) -> tuple[str, ...]:
    """Render a sample back to NDJSON lines.

    Round-trips with :func:`decode_samples`, which is what makes a decoded
    corpus re-emittable as a fixture.

    Args:
        sample: The sample to encode.

    Returns:
        One frame line followed by one line per entity.
    """
    frame = sample["frame"]
    lines = [
        f'{{"kind":"{KIND_FRAME}","frame":{frame},'
        f'"clock_ms":{sample["clock_ms"]},"owned":{len(sample["entities"])}}}'
    ]
    for entity in sample["entities"]:
        name = _escape(entity["class_name"])
        lines.append(
            f'{{"kind":"{KIND_ENTITY}","frame":{frame},"index":{entity["index"]},'
            f'"class":"{name}","x":{entity["x"]!r},"y":{entity["y"]!r}}}'
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
    "Entity",
    "Sample",
    "WireError",
    "decode_samples",
    "encode_sample",
]
