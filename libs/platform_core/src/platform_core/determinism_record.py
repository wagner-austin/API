"""What determinism a run had, recorded so a later run can be compared to it.

Seeding an RNG makes the same numbers come out of the sampler. It says
nothing about the order a machine accumulates a reduction in, and
floating-point addition is not associative, so the same seed on the same
hardware can still produce a different result on every run.

Two things follow. Determinism has to be TURNED ON -- few stacks do it by
default. And it has to be RECORDED: "whatever this version happened to
default to" is not a specification, cannot be written into a provenance
block, and cannot be compared against a later run.

This module is the RECORD, deliberately not the pinning. Pinning is
stack-specific -- torch writes cuDNN and cuBLAS settings, a BLAS-bound job
writes a thread count, an arbitrary-precision one writes a mantissa width --
and it lives with each stack. It sits in ``platform_core`` rather than beside
any one stack's pinner because most of this monorepo's research is not torch:
gradient boosting, transliteration and metabolomics pull no torch at all, and
a record only one stack could fill would make every other stack's runs
compare as though the question did not apply.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_str,
)

#: Value spelling for a pinned boolean setting. Settings are strings because
#: the record must hold any stack's vocabulary -- a torch run pins
#: ``cudnn_benchmark``, a BLAS-bound one pins a thread count, an
#: arbitrary-precision one pins a mantissa width -- and a union of every
#: stack's value types would grow forever while comparing no better.
TRUE = "true"
FALSE = "false"

#: The stack name for a run that pinned nothing. "Nothing was pinned" is a
#: fact about a run and must be recordable, because a run whose determinism
#: is unknown and a run deliberately left free are the same thing to a later
#: comparison, and both differ from a pinned one.
UNPINNED_STACK = "none"


class DeterminismRecord(TypedDict):
    """What determinism was in force, and which stack put it there.

    Attributes:
        stack: What pinned these settings, e.g. ``"torch"``, or
            :const:`UNPINNED_STACK` when nothing did. Part of the record
            rather than inferred from the setting names, because two stacks
            may pin settings that share a name and mean different things.
        settings: The pinned settings as ``(name, value)`` pairs, sorted by
            name. Sorted at construction so two records describing the same
            posture are equal and render identically regardless of the order
            a producer emitted them in.
    """

    stack: str
    settings: tuple[tuple[str, str], ...]


def determinism_record(stack: str, settings: Mapping[str, str]) -> DeterminismRecord:
    """Build a record, putting the settings in canonical order.

    Args:
        stack: What pinned the settings.
        settings: The pinned settings by name.

    Returns:
        The record, with settings sorted by name.

    Raises:
        ValueError: When ``stack`` is empty. A record that cannot say what
            pinned it is not comparable with one that can -- and
            :const:`UNPINNED_STACK` is how "nothing did" is spelled, so an
            empty string carries no meaning the vocabulary lacks.
    """
    if stack == "":
        raise ValueError("stack must name what pinned these settings, or be UNPINNED_STACK")
    return DeterminismRecord(stack=stack, settings=tuple(sorted(settings.items())))


def encode_determinism_record(record: DeterminismRecord) -> JSONObject:
    """Encode a record for a run record or a structured log field.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying the stack and the settings as a nested
        object. Nested rather than flattened so a setting can never collide
        with the ``stack`` key, whatever a future stack decides to name one.
    """
    return {
        "stack": record["stack"],
        "settings": dict(record["settings"]),
    }


def decode_determinism_record(value: JSONValue) -> DeterminismRecord:
    """Validate a JSON value as a determinism record.

    Args:
        value: The value to validate, typically from a stored run record.

    Returns:
        The validated record, with settings in canonical order.

    Raises:
        JSONTypeError: When ``value`` is not an object, the stack is absent
            or empty, ``settings`` is absent or not an object, or any
            setting value is not a string. A record that cannot say what
            pinned it, or that carries a setting whose value has to be
            guessed at, is not comparable with one that can.
    """
    obj = narrow_json_to_dict(value)
    stack = require_str(obj, "stack")
    if stack == "":
        raise JSONTypeError("Field 'stack' must name what pinned these settings")
    raw = require_dict(obj, "settings")
    settings: dict[str, str] = {}
    for name, setting in raw.items():
        if not isinstance(setting, str):
            raise JSONTypeError(f"Setting {name!r} must be a string, got {type(setting).__name__}")
        settings[name] = setting
    return determinism_record(stack, settings)


def render_determinism_record(record: DeterminismRecord) -> str:
    """Render a record as one stable comparison key.

    Args:
        record: The record to render.

    Returns:
        The stack and its settings in canonical order, so two runs with the
        same posture render byte-identically and a difference is legible
        without reading two nested objects side by side.
    """
    body = ",".join(f"{name}={value}" for name, value in record["settings"])
    return f"{record['stack']}[{body}]"


__all__ = [
    "FALSE",
    "TRUE",
    "UNPINNED_STACK",
    "DeterminismRecord",
    "decode_determinism_record",
    "determinism_record",
    "encode_determinism_record",
    "render_determinism_record",
]
