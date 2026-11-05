"""A weight edit written by a language model, as a document a run can apply.

Weight-space programming (Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight-Space Programming", MIT CSAIL, 2026-11-02) has a general-purpose
language model read a network's parameters and write new ones, with no
optimiser. What it writes has to reach the network somehow, and this is that
somehow: a short program of operations on named parameters, each of which a
reader can check by hand against the weights it was written for.

THREE OPERATIONS, ALL ON SLICES ALONG ONE AXIS. An operation names a
parameter, an axis, and the indices it selects along that axis -- rows when
the axis is 0, columns when it is 1 -- and treats each selected slice as a
vector across the other axis. A one-dimensional parameter (a bias, a gain)
has only axis 0, and each of its elements is a slice of width one.

* ``set`` replaces each selected slice. ``values`` holds the replacements
  end to end, one slice after another, so its length is the number of
  indices times the slice width.
* ``scale`` multiplies each selected slice by ``values[0]``.
* ``project_out`` removes, from each selected slice, its component along the
  direction ``values``: ``s <- s - (s . d) d`` for the unit vector ``d``.
  This is the operation the post's Experiment 8 edit was made of, and the one
  Arditi et al. (2024) remove refusal with.

WHY ``written_against`` IS REQUIRED. An edit is written by reading one exact
set of weights. Applied to any others it is a different edit -- a row index
that meant "the unit that tracks velocity" in one trained policy means
nothing in another -- so the program names the weights it was read from, and
the applier refuses it anywhere else.
"""

from __future__ import annotations

import math
from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

#: Every operation a program may hold, in the order the module docstring
#: describes them.
WEIGHT_OPS: tuple[Literal["set", "scale", "project_out"], ...] = (
    "set",
    "scale",
    "project_out",
)


class WeightOp(TypedDict):
    """One operation on the slices of one parameter.

    Attributes:
        op: Which operation, from :data:`WEIGHT_OPS`.
        parameter: The dotted parameter path it writes.
        axis: 0 to select rows (or the elements of a vector), 1 to select
            columns.
        indices: The selected slices, distinct and non-negative.
        values: The replacements, the factor, or the direction, by ``op``.
    """

    op: Literal["set", "scale", "project_out"]
    parameter: str
    axis: int
    indices: tuple[int, ...]
    values: tuple[float, ...]


class WeightProgram(TypedDict):
    """A whole edit: what it was written against, and its operations in order.

    Attributes:
        program_id: A name for the edit, carried onto the run's record.
        written_against: The digest of the weights the writer read, as
            :func:`~model_trainer.core.services.model.editing.verify.weights_digest`
            renders it.
        ops: The operations, applied first to last.
    """

    program_id: str
    written_against: str
    ops: tuple[WeightOp, ...]


def as_weight_op_name(raw: str) -> Literal["set", "scale", "project_out"]:
    """Narrow a string to an operation name, or refuse it.

    Args:
        raw: The name as read.

    Returns:
        The operation name.

    Raises:
        JSONTypeError: If the name is not one of :data:`WEIGHT_OPS`.
    """
    for name in WEIGHT_OPS:
        if raw == name:
            return name
    raise JSONTypeError(f"Field 'op' must be one of {', '.join(WEIGHT_OPS)}, got {raw!r}")


def _require_nonempty_str(obj: JSONObject, key: str) -> str:
    """Read a string field that must not be empty.

    Args:
        obj: The object to read.
        key: The field.

    Returns:
        The string.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def _require_indices(obj: JSONObject) -> tuple[int, ...]:
    """Read the selected indices: at least one, distinct, none negative.

    Args:
        obj: The operation object.

    Returns:
        The indices, in the order written.

    Raises:
        JSONTypeError: If the list is missing or empty, holds a non-integer
            or a negative index, or names one index twice. A repeat is
            refused rather than applied twice: ``scale`` would square its
            factor on that slice and ``set`` would keep only the second
            replacement, and neither is what a writer listing a row once per
            reason meant.
    """
    indices = tuple(narrow_json_to_int(raw) for raw in require_list(obj, "indices"))
    if not indices:
        raise JSONTypeError("Field 'indices' must select at least one slice")
    if min(indices) < 0:
        raise JSONTypeError(f"Field 'indices' must be non-negative, got {min(indices)}")
    if len(set(indices)) != len(indices):
        raise JSONTypeError("Field 'indices' must not name a slice twice")
    return indices


def _require_values(obj: JSONObject) -> tuple[float, ...]:
    """Read the operation's numbers: at least one, every one finite.

    Args:
        obj: The operation object.

    Returns:
        The values.

    Raises:
        JSONTypeError: If the list is missing or empty, or holds a
            non-number, an infinity or a NaN.
    """
    values = tuple(narrow_json_to_float(raw) for raw in require_list(obj, "values"))
    if not values:
        raise JSONTypeError("Field 'values' must hold at least one number")
    if not all(math.isfinite(value) for value in values):
        raise JSONTypeError("Field 'values' must hold only finite numbers")
    return values


def decode_weight_op(obj: JSONObject) -> WeightOp:
    """Decode one operation, refusing any that cannot be applied as written.

    Checks that need the parameter's shape (an index past the end, a
    direction of the wrong length) belong to the applier, which has the
    parameter; everything decidable from the document alone is decided here.

    Args:
        obj: The object to decode.

    Returns:
        The validated operation.

    Raises:
        JSONTypeError: If a field is missing or mistyped, the axis is not 0
            or 1, the indices or values are malformed, a ``scale`` carries
            other than one factor, a ``set`` carries a number of values that
            does not divide evenly among its indices, or a ``project_out``
            direction has zero length.
    """
    op = as_weight_op_name(require_str(obj, "op"))
    axis = require_int(obj, "axis")
    if axis not in (0, 1):
        raise JSONTypeError(f"Field 'axis' must be 0 or 1, got {axis}")
    indices = _require_indices(obj)
    values = _require_values(obj)
    if op == "scale" and len(values) != 1:
        raise JSONTypeError(f"a scale carries exactly one factor, got {len(values)} values")
    if op == "set" and len(values) % len(indices) != 0:
        raise JSONTypeError(
            f"a set of {len(indices)} slices carries {len(values)} values, "
            f"which is not a whole number of slices"
        )
    if op == "project_out" and math.fsum(value * value for value in values) == 0.0:
        raise JSONTypeError("a project_out direction must not be the zero vector")
    return WeightOp(
        op=op,
        parameter=_require_nonempty_str(obj, "parameter"),
        axis=axis,
        indices=indices,
        values=values,
    )


def encode_weight_op(op: WeightOp) -> JSONObject:
    """Encode one operation.

    Args:
        op: The operation.

    Returns:
        The JSON-serializable form.
    """
    indices: list[JSONValue] = list(op["indices"])
    values: list[JSONValue] = list(op["values"])
    return {
        "op": op["op"],
        "parameter": op["parameter"],
        "axis": op["axis"],
        "indices": indices,
        "values": values,
    }


def decode_weight_program(obj: JSONObject) -> WeightProgram:
    """Decode a whole program.

    Args:
        obj: The object to decode.

    Returns:
        The validated program.

    Raises:
        JSONTypeError: If a field is missing or mistyped, an identifier is
            empty, the program holds no operation, or any operation is
            refused by :func:`decode_weight_op`.
    """
    ops = tuple(decode_weight_op(narrow_json_to_dict(raw)) for raw in require_list(obj, "ops"))
    if not ops:
        raise JSONTypeError("Field 'ops' must hold at least one operation")
    return WeightProgram(
        program_id=_require_nonempty_str(obj, "program_id"),
        written_against=_require_nonempty_str(obj, "written_against"),
        ops=ops,
    )


def encode_weight_program(program: WeightProgram) -> JSONObject:
    """Encode a whole program.

    Args:
        program: The program.

    Returns:
        The JSON-serializable form.
    """
    ops: list[JSONValue] = [encode_weight_op(op) for op in program["ops"]]
    return {
        "program_id": program["program_id"],
        "written_against": program["written_against"],
        "ops": ops,
    }


__all__ = [
    "WEIGHT_OPS",
    "WeightOp",
    "WeightProgram",
    "as_weight_op_name",
    "decode_weight_op",
    "decode_weight_program",
    "encode_weight_op",
    "encode_weight_program",
]
