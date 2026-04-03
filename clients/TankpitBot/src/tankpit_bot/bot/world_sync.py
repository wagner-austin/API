"""World state synchronization for the tick loop.

Drains the CDP message buffer each tick and feeds messages through
the protocol decoder to keep world state fresh. Single path: CDP
Network.webSocketFrameReceived events are buffered by Bot, drained here.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import BotProtocol
from tankpit_bot.sniffer.decoders import process_received_message

log = get_logger(__name__)


def drain_messages(bot: BotProtocol) -> int:
    """Drain CDP message buffer and decode to update world state.

    Args:
        bot: Bot instance with CDP message buffer.

    Returns:
        Number of messages drained and decoded.
    """
    msgs = bot._cdp_message_buffer
    bot._cdp_message_buffer = []

    for payload in msgs:
        process_received_message(payload)

    if msgs:
        log.debug("SYNC: %d messages", len(msgs))

    return len(msgs)


__all__ = [
    "drain_messages",
]
