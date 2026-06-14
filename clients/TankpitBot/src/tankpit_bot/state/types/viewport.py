"""Viewport bounds TypedDict + encode/decode."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int
from typing_extensions import TypedDict


class ViewportStateDict(TypedDict):
    """Current viewport state.

    Attributes:
        left: Left edge X coordinate of viewport.
        top: Top edge Y coordinate of viewport.
        width: Visible viewport width in tiles (typically 16).
        height: Visible viewport height in tiles (typically 16).
    """

    left: int
    top: int
    width: int
    height: int


def encode_viewport_state(state: ViewportStateDict) -> JSONObject:
    """Encode ViewportStateDict to JSON-serializable dict.

    Args:
        state: ViewportStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "left": state["left"],
        "top": state["top"],
        "width": state["width"],
        "height": state["height"],
    }


def decode_viewport_state(data: JSONObject) -> ViewportStateDict:
    """Decode ViewportStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ViewportStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return ViewportStateDict(
        left=require_int(data, "left"),
        top=require_int(data, "top"),
        width=require_int(data, "width"),
        height=require_int(data, "height"),
    )


__all__ = [
    "ViewportStateDict",
    "decode_viewport_state",
    "encode_viewport_state",
]
