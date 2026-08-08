"""Replay-layer dependency injection: the decode entry point.

Lives here rather than in the process-wide ``_test_hooks`` package for a
layering reason. The hook must name :class:`WorldService` — a replay owns
its world and the decoder has to be told which one to write into — and
``_test_hooks`` sits BELOW ``sniffer``, because
``sniffer/world_service.py`` depends on the terrain and clock seams it
provides. Naming ``WorldService`` from there closes an import cycle
through ``state`` (measured 2026-08-07).

``action_lab`` already owns a package-local ``_test_hooks`` module for
the same reason, so this is the established shape rather than a new one
([[session-state-deglobalisation]] step 8).
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_service import WorldService


class ProcessReceivedMessageProtocol(Protocol):
    """Protocol for decoding one received frame into a world service."""

    def __call__(self, ws: WorldService, payload: str, xor_table: bytes) -> None:
        """Decode a received message payload into ``ws``.

        Args:
            ws: The replay's world service; the decode lands here.
            payload: Base64-encoded WebSocket frame payload.
            xor_table: The replayed session's XOR table.
        """
        ...


def _real_process_received_message(ws: WorldService, payload: str, xor_table: bytes) -> None:
    """Real implementation — delegate straight to the sniffer decoder.

    No singleton lookup: the service arrives as an argument. This was
    the last ``get_world_service()`` reach on the replay path.

    Args:
        ws: The replay's world service; the decode lands here.
        payload: Base64-encoded WebSocket frame payload.
        xor_table: The replayed session's XOR table.
    """
    process_received_message(ws, payload, xor_table)


#: Replay-time decode entry point. Tests replace this attribute via
#: save-and-restore to inject state mid-replay (for example a
#: ``self_state`` the capture never carried) without faking the whole
#: decoder stack.
process_received_message_hook: ProcessReceivedMessageProtocol = _real_process_received_message


__all__ = [
    "ProcessReceivedMessageProtocol",
    "_real_process_received_message",
    "process_received_message_hook",
]
