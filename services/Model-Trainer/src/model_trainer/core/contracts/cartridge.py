"""The shape of a trained key-value prefix, and what it was trained against.

A cartridge is a block of key and value vectors prepended to every attention
layer's cache, trained by gradient descent while the base model stays frozen.
Its parameters are meaningless without the geometry they were cut to: a tensor
of the right total size but the wrong head count is silently a different
object, and loading one into a model it does not fit produces either a shape
error at best or wrong attention at worst.

WHY GEOMETRY IS DISCOVERED RATHER THAN CONFIGURED. The head count that matters
is the KEY-VALUE head count, which under grouped-query attention is smaller
than the attention head count -- Llama-3.1-8B has 32 attention heads and 8
key-value heads. Reading it from a config would mean knowing that GPT-2 spells
its shape ``n_layer``/``n_head``/``n_embd`` while Llama spells it
``num_hidden_layers``/``num_key_value_heads``/``hidden_size``, and adding a
branch per architecture. Measured 2026-09-03 against transformers 4.46.3: a
one-token forward with ``use_cache=True`` returns a cache whose layer-zero key
has shape ``(batch, kv_heads, seq, head_dim)`` -- 2 for a Llama configured
with 8 attention heads and 2 key-value heads, and 4 for a GPT-2 with 4 heads.
So the model reports its own geometry, correctly and without a per-architecture
table. The reading is ``cartridge_slots.discover_geometry``.
"""

from __future__ import annotations

from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.json_utils import JSONObject, JSONValue, require_int
from typing_extensions import TypedDict

CARTRIDGE_MANIFEST_NAME = "cartridge.json"
"""Filename of the geometry manifest inside a saved cartridge directory.

JSON rather than folded into the tensor file, so what a cartridge IS can be
read without loading torch -- by a person, by a packaging step, or by an
inventory that must not import a deep-learning stack to list what it holds.
"""

CARTRIDGE_WEIGHTS_NAME = "cartridge.pt"
"""Filename of the tensor blocks inside a saved cartridge directory."""


class CartridgeGeometry(TypedDict):
    """The shape a cartridge's parameters were cut to.

    Every field is required and every field is positive. There is no partial
    geometry: a cartridge whose head count is unknown cannot be attached to
    anything, so a half-populated record would only defer the failure.

    Attributes:
        num_layers: Attention layers the prefix spans. One key block and one
            value block per layer.
        num_kv_heads: Key-value heads per layer, which under grouped-query
            attention is FEWER than the attention head count. This is the
            number the cache is shaped by, not the number the model advertises
            as its head count.
        head_dim: Width of a single head's key or value vector.
        num_slots: Prefix positions. The cartridge's capacity, and the only
            field a caller chooses; the other three are the model's.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_slots: int


def encode_cartridge_geometry(geometry: CartridgeGeometry) -> JSONObject:
    """Encode a geometry to a JSON object.

    Args:
        geometry: Geometry to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "num_layers": geometry["num_layers"],
        "num_kv_heads": geometry["num_kv_heads"],
        "head_dim": geometry["head_dim"],
        "num_slots": geometry["num_slots"],
    }


def _require_positive(obj: JSONObject, key: str) -> int:
    """Read a required field that must be a positive count.

    Zero is refused as firmly as a negative. A zero-slot cartridge holds no
    parameters, a zero-layer one attaches to nothing, and both would train
    without error while learning nothing -- a silent no-op is the failure this
    is here to prevent.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing or is not an integer.
        AppError: With ``CARTRIDGE_GEOMETRY_INVALID`` if the value is zero or
            negative.
    """
    value = require_int(obj, key)
    if value > 0:
        return value
    raise AppError(
        ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID,
        (
            f"cartridge geometry field {key!r} must be a positive count, got {value}; "
            f"a cartridge with no {key.removeprefix('num_')} holds no trainable "
            f"parameters and would train to completion having learned nothing"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID),
    )


def decode_cartridge_geometry(value: JSONValue) -> CartridgeGeometry:
    """Decode and validate a geometry read back from disk.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated geometry.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing or
            mistyped.
        AppError: With ``CARTRIDGE_GEOMETRY_INVALID`` if any count is not
            positive.
    """
    if not isinstance(value, dict):
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID,
            (
                f"cartridge geometry must be a JSON object, got "
                f"{type(value).__name__}; the file naming this cartridge's shape "
                f"is not the file that was written"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID),
        )
    return CartridgeGeometry(
        num_layers=_require_positive(value, "num_layers"),
        num_kv_heads=_require_positive(value, "num_kv_heads"),
        head_dim=_require_positive(value, "head_dim"),
        num_slots=_require_positive(value, "num_slots"),
    )


def trainable_parameter_count(geometry: CartridgeGeometry) -> int:
    """Count the scalars a cartridge of this geometry trains.

    One key block and one value block per layer, hence the factor of two. This
    is the whole of what a cartridge run updates: the base model contributes
    nothing to this number, which is the point of the method.

    Args:
        geometry: The shape to count.

    Returns:
        Total trainable scalars.
    """
    return (
        2
        * geometry["num_layers"]
        * geometry["num_kv_heads"]
        * geometry["num_slots"]
        * geometry["head_dim"]
    )


__all__ = [
    "CARTRIDGE_MANIFEST_NAME",
    "CARTRIDGE_WEIGHTS_NAME",
    "CartridgeGeometry",
    "decode_cartridge_geometry",
    "encode_cartridge_geometry",
    "trainable_parameter_count",
]
