"""The planner's connection to a running agent.

Reads world samples as the agent pushes them and sends orders back over the
same line channel. The agent listens and this connects, because the game is the
long-lived process: a planner can attach, exit, and reattach without disturbing
the match (wiki: issuing-orders).

The channel transports and validates. It chooses nothing — a caller reads
samples, decides, and hands back orders — which is the same split the agent
holds on its own side.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot import RwBotError
from rw_bot.control import _test_hooks
from rw_bot.validation import require_int
from rw_bot.wire.command import (
    AttackOrder,
    BuildOrder,
    MoveOrder,
    ProduceOrder,
    encode_ack,
    encode_attack,
    encode_build,
    encode_move,
    encode_produce,
)
from rw_bot.wire.ndjson import parse_object
from rw_bot.wire.state import Sample, decode_samples

DEFAULT_HOST = "127.0.0.1"
DEFAULT_TIMEOUT_S = 30.0

_STREAM_ENDED = "RW-CHAN-001"


class ChannelError(RwBotError):
    """The agent channel ended or could not carry a message.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description.
    """


class AgentChannel:
    """One connection to a running agent.

    Attributes:
        _connection: The open line channel.
    """

    def __init__(self, connection: _test_hooks.Connection) -> None:
        self._connection = connection

    def next_sample(self) -> Sample:
        """Read lines until one whole world sample has arrived.

        A sample is a frame record followed by the entity and pool records it
        declares, so this accumulates until both counts are satisfied rather
        than returning a partial world. A planner acting on a world it cannot
        fully see is the failure this prevents.

        Returns:
            The next complete sample.

        Raises:
            ChannelError: ``RW-CHAN-001`` when the stream ends mid-sample or
                before one arrives.
            NdjsonError: When a line does not parse.
            WireError: When the records do not form a valid sample.
            DecodeError: When a record is missing a field or mistyped.
        """
        lines: list[str] = []
        while True:
            line = self._connection.read_line()
            if line == "":
                raise ChannelError(
                    _STREAM_ENDED,
                    "the agent closed the channel before a whole sample arrived; "
                    f"{len(lines)} line(s) were pending",
                )
            lines.append(line)
            samples = _complete_or_none(lines)
            if samples is not None:
                return samples

    def send_move(self, order: MoveOrder) -> None:
        """Send one move order.

        Args:
            order: The order to send.

        Raises:
            OSError: When the write fails.
        """
        self._connection.send_line(encode_move(order))

    def send_build(self, order: BuildOrder) -> None:
        """Send one build order.

        Args:
            order: The order to send.

        Raises:
            OSError: When the write fails.
        """
        self._connection.send_line(encode_build(order))

    def send_produce(self, order: ProduceOrder) -> None:
        """Send one produce order.

        Args:
            order: The order to send.

        Raises:
            OSError: When the write fails.
        """
        self._connection.send_line(encode_produce(order))

    def send_attack(self, order: AttackOrder) -> None:
        """Send one attack order.

        Args:
            order: The order to send.

        Raises:
            OSError: When the write fails.
        """
        self._connection.send_line(encode_attack(order))

    def send_ack(self) -> None:
        """Tell the agent this sample is finished with.

        Harmless when the agent is not in lockstep -- it parses the line and
        releases a barrier nobody is waiting on. Sending it unconditionally is
        what keeps the planner from having to know which mode the agent is in.

        Raises:
            OSError: When the write fails.
        """
        self._connection.send_line(encode_ack())

    def close(self) -> None:
        """Release the connection."""
        self._connection.close()


def open_channel(
    port: int,
    host: str = DEFAULT_HOST,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> AgentChannel:
    """Connect to an agent.

    Args:
        port: Port the agent listens on.
        host: Host to reach it on.
        timeout_s: Socket timeout in seconds.

    Returns:
        The open channel.

    Raises:
        OSError: When the agent is not listening.
    """
    return AgentChannel(_test_hooks.connect(host, port, timeout_s))


def _complete_or_none(lines: Sequence[str]) -> Sample | None:
    """Return the sample these lines form, or None while it is still partial.

    The frame record states how many entity and pool records follow, so
    completeness is known before parsing the whole thing. Decoding is left to
    :func:`~rw_bot.wire.state.decode_samples`, which re-checks both counts and
    rejects a mismatch — this only decides when to stop reading.

    Args:
        lines: Lines accumulated since the last complete sample.

    Returns:
        The sample once both declared counts are satisfied, otherwise None.

    Raises:
        NdjsonError: When a line does not parse.
        WireError: When the records cannot form a valid sample.
        DecodeError: When a record is missing a field or mistyped.
    """
    opening = parse_object(lines[0])
    declared = (
        require_int(opening, "visible")
        + require_int(opening, "pools")
        + require_int(opening, "options")
    )
    if len(lines) < declared + 1:
        return None
    return decode_samples(list(lines))[0]


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_TIMEOUT_S",
    "AgentChannel",
    "ChannelError",
    "open_channel",
]
