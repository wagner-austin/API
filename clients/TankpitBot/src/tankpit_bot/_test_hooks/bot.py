"""Bot dispatch and buffered-message-source protocols.

``BotProtocol`` is the narrow command-dispatch surface that
:mod:`tankpit_bot.bot.executor` and :mod:`tankpit_bot.bot.world_sync`
consume. The tick loop itself uses the concrete ``Bot`` class; only the
inner consumers receive the protocol so tests can substitute focused
fakes without bringing along the full Bot machinery.

``BufferedMessageSourceProtocol`` is the still narrower surface a
draining loop needs: a single mutable list of base64-encoded payloads.
"""

from __future__ import annotations

from typing import Protocol


class BufferedMessageSourceProtocol(Protocol):
    """Interface for objects that buffer received protocol payloads.

    Attributes:
        _cdp_message_buffer: Frames captured since the last drain.
        xor_table: The SESSION's XOR table, or None until its magic is
            captured. Frames cannot be decoded without it
            ([[session-state-deglobalisation]]).
    """

    _cdp_message_buffer: list[str]
    xor_table: bytes | None


__all__ = [
    "BufferedMessageSourceProtocol",
]
