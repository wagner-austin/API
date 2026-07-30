"""Chat protocol: preset message table and the outbound chat command.

The game has exactly 65 predefined chat messages (JS ``E[0]``..``E[64]``,
wiki [[chat-messages]]); there is no free-text chat. A chat rides the
``m`` (0x6D) Hb command. Wire-verified against sniff-20260729-214411
(44 live sends decoded): every send uses the 6-byte serialized form
``[6,'m',message_id,x,y,flag]`` -- including messages with no position
semantics -- with ``x,y`` the sender's current tile and ``flag`` always
0. The server's broadcast comes back as the 0x4D ``M`` frame
(``decode_chat_message``): ``sender_id(2 LE) + message_id + x + y``.

Flood-mute contract (same capture): after 8 rapid sends the server
silently swallowed every later chat for the rest of the session -- no
error frame, no echo. Bot policy: chat rarely, never retry on silence.
"""

from __future__ import annotations

from tankpit_bot.protocol.commands import COMMAND_PREFIX

CMD_CHAT = 0x6D
"""Chat command byte (``'m'``, JS Hb class)."""

TYPE_CHAT = 6
"""Wire type byte for chat sends.

The plaintext send scheme mirrors the JS serialized frame: the type
byte equals the K-command's length byte, and the 6-byte Hb chat frame
therefore shares the value with :data:`TYPE_COMBAT`.
"""

CHAT_HELLO = 41
"""Message ID for "HELLO" (all-players filter, no target check)."""

CHAT_MESSAGES: dict[int, str] = {
    0: "Attack the red",
    1: "Attack the purple",
    2: "Attack the blue",
    3: "Attack the orange",
    4: "HELP - Enemy!",
    5: "Bring it!",
    6: "HELP - Fuel low!",
    7: "I'll help you",
    8: "Fuel detected here",
    9: "Equipment detected here",
    10: "Thanks",
    11: "No problem",
    12: "Base is here",
    13: "Enemy base is here",
    14: "Ferry located here",
    15: "Meet me",
    16: "Come get me!",
    17: "Follow me",
    18: "Buzz off!",
    19: "Out of the way",
    20: "Stop shooting",
    21: "Retreat",
    22: "Let's team up",
    23: "Make base",
    24: "Move obstacle",
    25: "Build bridge",
    26: "Plant mines",
    27: "Blow up mines",
    28: "Use the radar",
    29: "Charge",
    30: "Hold on",
    31: "Getting fuel",
    32: "Getting equipment",
    33: "Sure!",
    34: "No way!",
    35: "Congrats",
    36: "You've got mad skills!",
    37: "My bad",
    38: "That was mine",
    39: "Long time no see",
    40: "Be right back",
    41: "HELLO",
    42: "BYE",
    43: "Let's chill here for a while",
    44: "That was whack!",
    45: "Whatever",
    46: "Do your worst!",
    47: "Don't Cry.",
    48: "Is that the best you can do?",
    49: "My dog plays better than you!",
    50: "Nice try!",
    51: "Move obstacle off of ferry",
    52: "Don't follow me!",
    53: "I need equipment!",
    54: "I need fuel!",
    55: "I gotta go.",
    56: "Put that here.",
    57: "Lame!",
    58: "Who's your daddy?",
    59: "I rule!",
    60: "Check the bulletin board.",
    61: "Good job",
    62: "You're Welcome",
    63: "I'm playing TankPit, mom.",
    64: "I'm playing TankPit, dad.",
}
"""All 65 preset chat texts, keyed by message ID (JS ``E[]`` table)."""


def chat_message_text(message_id: int) -> str:
    """Return the preset text for a chat message ID.

    Args:
        message_id: Chat message ID (0-64 for known presets).

    Returns:
        The preset text, or ``"unknown_<id>"`` for an ID outside the
        E[] table (forward-compat against server-side additions).
    """
    text = CHAT_MESSAGES.get(message_id)
    if text is None:
        return f"unknown_{message_id}"
    return text


def build_chat_command(message_id: int, x: int, y: int) -> bytes:
    """Build a chat command ready to send (with length header).

    Format: ``[len_lo, len_hi] + ! + 0x06 + 0x6D + id + x + y + 0x00``
    (9 bytes total) -- the plaintext image of the page client's
    6-byte Hb frame, byte-identical after the send path's XOR pass to
    every send in sniff-20260729-214411.

    Args:
        message_id: Preset chat message ID (0-64).
        x: Sender's current X tile (0-255).
        y: Sender's current Y tile (0-255).

    Returns:
        Framed command bytes ready to send via WebSocket.
    """
    body = bytes(
        [
            COMMAND_PREFIX,
            TYPE_CHAT,
            CMD_CHAT,
            message_id & 0xFF,
            x & 0xFF,
            y & 0xFF,
            0,
        ]
    )
    length = len(body)
    return bytes([length & 0xFF, (length >> 8) & 0xFF]) + body


__all__ = [
    "CHAT_HELLO",
    "CHAT_MESSAGES",
    "CMD_CHAT",
    "TYPE_CHAT",
    "build_chat_command",
    "chat_message_text",
]
