"""Configuration types for sniffer and bot."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_int,
    require_str,
)


class SnifferConfig(TypedDict):
    """Configuration for the WebSocket sniffer.

    Attributes:
        target_url: URL to navigate to and capture WebSocket traffic.
        output_path: Path to save captured session data.
        headless: Whether to run browser in headless mode.
        capture_duration_ms: How long to capture traffic in milliseconds.
    """

    target_url: str
    output_path: str
    headless: bool
    capture_duration_ms: int


def encode_sniffer_config(config: SnifferConfig) -> JSONObject:
    """Encode SnifferConfig to JSON-serializable dict.

    Args:
        config: SnifferConfig to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "target_url": config["target_url"],
        "output_path": config["output_path"],
        "headless": config["headless"],
        "capture_duration_ms": config["capture_duration_ms"],
    }
    return result


def decode_sniffer_config(data: JSONObject) -> SnifferConfig:
    """Decode SnifferConfig from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SnifferConfig.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    headless_val = data.get("headless")
    if headless_val is None:
        raise JSONTypeError("Missing required field 'headless'")
    if not isinstance(headless_val, bool):
        actual_type = type(headless_val).__name__
        raise JSONTypeError(f"Field 'headless' must be a boolean, got {actual_type}")

    return SnifferConfig(
        target_url=require_str(data, "target_url"),
        output_path=require_str(data, "output_path"),
        headless=headless_val,
        capture_duration_ms=require_int(data, "capture_duration_ms"),
    )


class BotConfig(TypedDict):
    """Configuration for the TankpitBot.

    Attributes:
        ws_url: WebSocket URL to connect to.
        username: Tank username for the game.
        game_id: Game/map ID to join.
    """

    ws_url: str
    username: str
    game_id: str


def encode_bot_config(config: BotConfig) -> JSONObject:
    """Encode BotConfig to JSON-serializable dict.

    Args:
        config: BotConfig to encode.

    Returns:
        JSON-serializable dict representation.
    """
    result: JSONObject = {
        "ws_url": config["ws_url"],
        "username": config["username"],
        "game_id": config["game_id"],
    }
    return result


def decode_bot_config(data: JSONObject) -> BotConfig:
    """Decode BotConfig from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated BotConfig.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return BotConfig(
        ws_url=require_str(data, "ws_url"),
        username=require_str(data, "username"),
        game_id=require_str(data, "game_id"),
    )


__all__ = [
    "BotConfig",
    "SnifferConfig",
    "decode_bot_config",
    "decode_sniffer_config",
    "encode_bot_config",
    "encode_sniffer_config",
]
