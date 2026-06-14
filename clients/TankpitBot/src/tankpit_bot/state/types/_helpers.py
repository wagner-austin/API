"""Internal helpers shared by world-state submodules.

Keeps the once-duplicated ``_decode_dict_field_X`` shapes in one place so
new entity collections (future viewports, tiles, etc.) inherit the same
strict iterate-and-decode behavior without copy-paste drift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from platform_core.json_utils import JSONObject, JSONValue

_T = TypeVar("_T")


def decode_entity_dict(
    raw: JSONValue,
    decoder: Callable[[JSONObject], _T],
) -> dict[str, _T]:
    """Decode a ``{"x,y": entity}`` dict using ``decoder`` per entry.

    The world-state JSON shape stores tanks, containers, mines, and
    terrain as ``dict[str, EntityDict]``. Each value is decoded by a
    type-specific decoder; this helper handles the outer dict iteration
    and skips entries that are not JSON objects (mirroring the
    permissive shape the previous per-entity helpers used).

    Args:
        raw: Raw JSON value, expected to be an object.
        decoder: Decoder function applied to each inner object.

    Returns:
        Mapping of preserved keys to decoded entities. Empty when
        ``raw`` is not a JSON object.
    """
    result: dict[str, _T] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decoder(v)
    return result


__all__ = [
    "decode_entity_dict",
]
