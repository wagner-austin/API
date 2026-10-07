"""Read one capture as a sequence of server ticks, for the conformance replay.

The real server processes client commands once per tick
(:data:`~tankpit_bot.protocol.commands.TICK_RATE_MS`) and answers each
tick with one batch, which reaches the capture as a run of WebSocket
messages a few milliseconds apart. So a capture reads as alternation:
commands the client sent, then the batch that answered them, then more
commands. :func:`read_replay` recovers exactly that pairing: every
received run is one :class:`ReplayTick`, holding the commands sent since
the previous run.

The response-shape differ windows a capture by wall clock instead
(:data:`~tankpit_bot.analysis.response_shapes.WINDOW_MS`), which is why
roughly 3,000 of its divergences were a slow answer spilling into the
next command's window ([[capture-differ]]). Pairing by batch has no such
spill: a command's answer is the batch after it, whenever that lands.

The capture also names the field it played. The lobby lists every room
with its field image (``+`` text frames) and confirms the one joined
(``=<room>|...``); the last confirmed join of a listed room is the field.
"""

from __future__ import annotations

from typing import NamedTuple

from tankpit_bot.analysis.response_shapes import decode_received_frame
from tankpit_bot.analysis.types import DecodedFrameDict
from tankpit_bot.parser import is_room_info_text
from tankpit_bot.parser_messages import parse_room_info
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict, decode_client_command
from tankpit_bot.types.literals import MessageDirection

BURST_GAP_MS = 500
"""Received messages closer than this, with no command between, are one batch.

A tick's batch lands as several messages within a few milliseconds of
each other, and consecutive batches are a tick (2,000 ms) apart, so a
quarter of a tick separates them with room to spare on either side.
"""

_ROOM_INFO_PREFIX = b"+"
_JOIN_CONFIRM_PREFIX = b"="


class ReplayTick(NamedTuple):
    """One server tick as the capture recorded it.

    Attributes:
        timestamp_ms: When the batch's last message was captured.
        commands: The commands the client sent since the previous batch,
            in send order; empty for a tick the client sent nothing into.
        received: The batch's decoded binary messages, in capture order.
    """

    timestamp_ms: int
    commands: tuple[ClientCommandDict, ...]
    received: tuple[BinaryMessage, ...]


class ReplayCapture(NamedTuple):
    """A capture read as ticks.

    Attributes:
        field_image: The joined room's field image (``field01.gif``), or
            None when the capture never confirmed a join of a listed room.
        ticks: Every batch, in capture order.
    """

    field_image: str | None
    ticks: tuple[ReplayTick, ...]


def _frame_time(frame: DecodedFrameDict) -> int:
    """Sort key: one frame's capture time.

    Args:
        frame: The frame to order.

    Returns:
        Its capture time in milliseconds.
    """
    return frame["timestamp_ms"]


def _sent_command(frame: DecodedFrameDict) -> ClientCommandDict | None:
    """Decode a sent frame's command, if it carries one.

    The lobby shares the socket, so only ``!``-prefixed frames are
    commands. A ``!`` frame whose body will not decode is not passed
    over: the tick it was sent into would then be replayed without it
    and compared as if the client had sent less than it did. Every one
    of the 163,688 commands in the 451 captures of ``runs/bot`` and
    ``runs/probe`` decodes (2026-10-07), so such a frame is a fault in
    the capture or in :func:`~tankpit_bot.sim.commands.decode_client_command`,
    and it stops the read.

    Args:
        frame: One sent frame.

    Returns:
        The command, or None when the frame is not a command.

    Raises:
        DecodeError: If a ``!`` frame's body is not a decodable command.
    """
    if frame["msg_type"] != COMMAND_PREFIX:
        return None
    return decode_client_command(frame["body"])


class _RoomLedger:
    """The lobby's room list and the joins confirmed against it."""

    def __init__(self) -> None:
        self.images: dict[str, str] = {}
        self.joined: str | None = None

    def observe(self, raw: bytes) -> None:
        """Note a received frame if it is a room listing or a join.

        A binary frame can begin with either prefix byte; it decodes to
        text that names no listed room, so it is never mistaken for one.

        Args:
            raw: The frame as captured.
        """
        prefix = raw[:1]
        if prefix not in (_ROOM_INFO_PREFIX, _JOIN_CONFIRM_PREFIX):
            return
        text = raw[1:].decode("utf-8", errors="replace")
        if prefix == _ROOM_INFO_PREFIX:
            if is_room_info_text(text):
                info = parse_room_info(text)
                self.images[info["room_id"]] = info["image"]
            return
        room_id = text.split("|", 1)[0]
        if room_id in self.images:
            self.joined = room_id

    def field_image(self) -> str | None:
        """The joined room's field image, if a listed room was joined."""
        if self.joined is None:
            return None
        return self.images[self.joined]


class _TickBuilder:
    """Accumulates commands and received runs into ticks."""

    def __init__(self) -> None:
        self.ticks: list[ReplayTick] = []
        self._pending: list[ClientCommandDict] = []
        self._run: list[BinaryMessage] = []
        self._run_end = 0

    def command(self, command: ClientCommandDict) -> None:
        """Note a sent command, which closes any open received run."""
        self._close()
        self._pending.append(command)

    def received(self, timestamp_ms: int, message: BinaryMessage) -> None:
        """Note a received message, opening a new run after a gap."""
        if self._run and timestamp_ms - self._run_end > BURST_GAP_MS:
            self._close()
        self._run.append(message)
        self._run_end = timestamp_ms

    def finish(self) -> tuple[ReplayTick, ...]:
        """Close the last run and return every tick.

        Commands sent after the last batch were never answered in the
        capture, so they form no tick.
        """
        self._close()
        return tuple(self.ticks)

    def _close(self) -> None:
        if not self._run:
            return
        self.ticks.append(
            ReplayTick(
                timestamp_ms=self._run_end,
                commands=tuple(self._pending),
                received=tuple(self._run),
            )
        )
        self._pending = []
        self._run = []


def read_replay(frames: list[DecodedFrameDict]) -> ReplayCapture:
    """Read a capture's frames as ticks and find the field it played.

    Args:
        frames: Every frame of one capture, both directions, from
            :func:`tankpit_bot.analysis.scan.decode_session_frames`.

    Returns:
        The capture as ticks, with its field image.

    Raises:
        DecodeError: If a sent ``!`` frame's body is not a decodable
            command.
    """
    rooms = _RoomLedger()
    builder = _TickBuilder()
    for frame in sorted(frames, key=_frame_time):
        if frame["direction"] is MessageDirection.SENT:
            command = _sent_command(frame)
            if command is not None:
                builder.command(command)
            continue
        rooms.observe(frame["raw"])
        message = decode_received_frame(frame)
        if message is not None:
            builder.received(frame["timestamp_ms"], message)
    return ReplayCapture(field_image=rooms.field_image(), ticks=builder.finish())


__all__ = ["BURST_GAP_MS", "ReplayCapture", "ReplayTick", "read_replay"]
