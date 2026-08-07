"""World state synchronization for the tick loop.

Drains the CDP message buffer each tick and feeds messages through
the protocol decoder to keep world state fresh. Single path: CDP
Network.webSocketFrameReceived events are buffered by Bot, drained here.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.sniffer.decoders import process_received_message

log = get_logger(__name__)


def drain_messages(bot: BufferedMessageSourceProtocol) -> int:
    """Drain CDP message buffer and decode to update world state.

    Frames buffered before the session's magic arrives cannot be
    decoded at all — the XOR table is derived from that magic. They are
    dropped, and the drain reports zero. The previous global decoder
    instead XOR'd them against a ``None`` table, which returned the body
    UNDECODED and dispatched the garbage into world state as if it were
    real ([[session-state-deglobalisation]]).

    Args:
        bot: Bot instance with CDP message buffer and session XOR table.

    Returns:
        Number of messages drained and decoded. Zero while the session
        has no XOR table yet, with the buffer left intact so the frames
        decode on a later tick.
    """
    xor_table = bot.xor_table
    if xor_table is None:
        return 0

    msgs = bot._cdp_message_buffer
    bot._cdp_message_buffer = []

    for payload in msgs:
        process_received_message(payload, xor_table)

    if msgs:
        log.debug("SYNC: %d messages", len(msgs))

    return len(msgs)


__all__ = [
    "drain_messages",
]
