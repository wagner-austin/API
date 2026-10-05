"""One tick's outgoing wire, one batch per connection.

The server used to fill a single ``messages`` list, which was correct
with one connection and meant nothing with two: a per-recipient receipt
(a 0x52 refusal, a 0x4B placement, a 0x67 gain, a fuel sync) and a
broadcast (a 0x45 detonation, a corpse's 0x58, a 0x4D chat) were
appended the same way, so the list could not say who was entitled to
which. The routers now say it, and this is where they say it to.

There are exactly three ways a message reaches a connection, and each
is one method:

* :meth:`TickOutbox.to` — a RECEIPT, owed to one tank's connection
  alone. A tank nobody is connected for (a roster bot, the scripted
  opponent, a ghost) has no batch, and its receipt goes nowhere, which
  is the truth: the server does not answer a connection that does not
  exist.
* :meth:`TickOutbox.broadcast` — a FIELD event every connection sees
  identically.
* :meth:`TickOutbox.narrate` — an outcome every connection sees, but
  each through its own eyes: the pure narrators of
  :mod:`tankpit_bot.sim.narrate` are called once per observer, so the
  action is resolved once and described N times
  ([[recipient-policy]]).

With one connection all three reduce to appending to its one batch, in
the order they are called, which is why the single-client wire is
unchanged byte for byte.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import SimError


class TickOutbox:
    """The batches one tick owes its connections, in emission order.

    Batches are keyed by the connected tank's id and kept in the order
    the connections were admitted, so iterating them is deterministic.
    """

    def __init__(self, recipients: Sequence[int]) -> None:
        """Open an empty batch for each connected tank.

        Args:
            recipients: The connected tanks' ids, in admission order.

        Raises:
            SimError: If an id appears twice — two batches for one
                connection would deliver every broadcast twice.
        """
        self._batches: dict[int, list[BinaryMessage]] = {}
        for recipient in recipients:
            self.admit(recipient)

    def admit(self, recipient: int) -> None:
        """Open a batch for a connection that joined since this opened.

        Args:
            recipient: The newly connected tank's id.

        Raises:
            SimError: If the tank already has a batch.
        """
        if recipient in self._batches:
            raise SimError(f"tank {recipient} already has a batch in this tick's outbox")
        self._batches[recipient] = []

    def to(self, recipient: int, message: BinaryMessage) -> None:
        """Deliver a receipt to one tank's connection, if it has one.

        Args:
            recipient: The tank the receipt is owed to.
            message: The receipt.
        """
        batch = self._batches.get(recipient)
        if batch is not None:
            batch.append(message)

    def broadcast(self, messages: Sequence[BinaryMessage]) -> None:
        """Deliver field events to every connection, identically.

        Args:
            messages: The events, in emission order.
        """
        for batch in self._batches.values():
            batch.extend(messages)

    def narrate(self, narrator: Callable[[int], list[BinaryMessage]]) -> None:
        """Describe one resolved outcome to each connection in turn.

        Args:
            narrator: The pure narration of the outcome for one
                observer id. It is called once per connection and must
                not mutate the world — the outcome is already resolved.
        """
        for observer_id, batch in self._batches.items():
            batch.extend(narrator(observer_id))

    def batch(self, recipient: int) -> list[BinaryMessage]:
        """One connection's batch, for its own per-connection holders.

        The viewport, rank and award holders append to a list they are
        handed; this is that list.

        Args:
            recipient: A connected tank's id.

        Returns:
            The live batch (appending to it appends to the outbox).

        Raises:
            SimError: If the tank has no connection.
        """
        batch = self._batches.get(recipient)
        if batch is None:
            raise SimError(f"tank {recipient} has no connection to batch for")
        return batch

    def batches(self) -> dict[int, list[BinaryMessage]]:
        """Every connection's finished batch, in admission order.

        Returns:
            The batches keyed by connected tank id.
        """
        return self._batches


__all__ = ["TickOutbox"]
