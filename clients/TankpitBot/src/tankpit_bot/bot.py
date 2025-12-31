"""TankpitBot entry point.

The bot connects to Tankpit's WebSocket server and plays the game automatically.
Protocol details are discovered using the sniffer module.
"""

from __future__ import annotations

import sys

from tankpit_bot import _test_hooks


class BotError(Exception):
    """Base error for bot operations."""


class ProtocolNotDiscoveredError(BotError):
    """Raised when the game protocol has not been discovered yet."""


def main() -> None:
    """Entry point for tankpit-bot command.

    Currently displays usage instructions since the protocol must be
    captured first using the sniffer.
    """
    capture_path = _test_hooks.get_env("TANKPIT_CAPTURE")
    if capture_path is None:
        capture_path = "capture_session.json"

    output_lines = [
        "TankpitBot - Automated Tankpit.com player",
        "",
        "Before running the bot, you must first capture the game protocol:",
        "",
        "  1. Run the sniffer to capture WebSocket traffic:",
        "     make sniff",
        "",
        "  2. Log into tankpit.com in the browser window that opens",
        "",
        "  3. Join a game and play for a bit",
        "",
        "  4. The sniffer will save the captured protocol to:",
        f"     {capture_path}",
        "",
        "  5. Analyze the captured messages to understand the protocol",
        "",
        "Once the protocol is understood, this bot will be updated to:",
        "  - Connect to the WebSocket server",
        "  - Authenticate and join games",
        "  - Control tank movement and shooting",
        "  - Implement AI strategy",
        "",
    ]

    for line in output_lines:
        sys.stdout.write(line + "\n")


__all__ = [
    "BotError",
    "ProtocolNotDiscoveredError",
    "main",
]
