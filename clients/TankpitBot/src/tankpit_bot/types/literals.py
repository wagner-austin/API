"""Closed vocabularies for TypedDict fields, and dict conversion helpers.

Each vocabulary is a StrEnum whose members are the words the capture
files and probe records carry, so a decoder narrows untrusted text with
:func:`platform_core.members.require_member` and an encoder writes
``.value``.
"""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import JSONObject

# =============================================================================
# Vocabularies
# =============================================================================


class MessageDirection(StrEnum):
    """Which side of the wire a message or frame travelled.

    Sent messages carry our own commands (the same XOR cipher covers both
    directions); received messages are the server's stream.
    """

    SENT = "sent"
    RECEIVED = "received"


class InputType(StrEnum):
    """Which device a recorded probe input came from."""

    KEY = "key"
    MOUSE = "mouse"


class MouseButton(StrEnum):
    """Which mouse button a recorded probe click used."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class SentFrameOrigin(StrEnum):
    """Who put a sent frame on the wire: the bot, or the page's own client."""

    BOT_INJECTED = "bot_injected"
    PAGE_CLIENT = "page_client"
    UNKNOWN = "unknown"


# =============================================================================
# Dict Conversion Helpers
# =============================================================================


def str_dict_to_json(source: dict[str, str]) -> JSONObject:
    """Convert dict[str, str] to JSONObject for type safety.

    Args:
        source: Dict with string keys and values.

    Returns:
        JSONObject with same contents.
    """
    result: JSONObject = {}
    for key, value in source.items():
        result[key] = value
    return result


def int_dict_to_json(source: dict[str, int]) -> JSONObject:
    """Convert dict[str, int] to JSONObject for type safety.

    Args:
        source: Dict with string keys and int values.

    Returns:
        JSONObject with same contents.
    """
    result: JSONObject = {}
    for key, value in source.items():
        result[key] = value
    return result


def mixed_dict_to_json(source: dict[str, int | str]) -> JSONObject:
    """Convert dict[str, int | str] to JSONObject for type safety.

    Args:
        source: Dict with string keys and int or str values.

    Returns:
        JSONObject with same contents.
    """
    result: JSONObject = {}
    for key, value in source.items():
        result[key] = value
    return result


__all__ = [
    "InputType",
    "MessageDirection",
    "MouseButton",
    "SentFrameOrigin",
    "int_dict_to_json",
    "mixed_dict_to_json",
    "str_dict_to_json",
]
