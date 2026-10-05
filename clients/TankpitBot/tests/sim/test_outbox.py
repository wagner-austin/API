"""TickOutbox: the three ways a message reaches a connection."""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.types import BinaryMessage, SyncDict, TankRemoveDict
from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.outbox import TickOutbox


def _remove(tank_id: int) -> TankRemoveDict:
    """A 0x58 naming one tank — a message whose content says who it is."""
    return TankRemoveDict(msg_type=0x58, tank_id=tank_id)


def test_batches_open_empty_in_admission_order() -> None:
    """Each connection starts with nothing, and iteration is join order."""
    outbox = TickOutbox((20, 9))
    outbox.admit(31)

    assert list(outbox.batches()) == [20, 9, 31]
    assert outbox.batches() == {20: [], 9: [], 31: []}


def test_a_receipt_reaches_its_connection_alone() -> None:
    """``to`` appends to one batch and leaves every other untouched."""
    outbox = TickOutbox((9, 20))

    outbox.to(20, SyncDict(msg_type=0x3F))

    assert outbox.batches() == {9: [], 20: [SyncDict(msg_type=0x3F)]}


def test_a_receipt_for_an_unconnected_tank_goes_nowhere() -> None:
    """A roster bot's refusal is owed to no connection, so none gets it."""
    outbox = TickOutbox((9,))

    outbox.to(500, SyncDict(msg_type=0x3F))

    assert outbox.batches() == {9: []}


def test_a_broadcast_reaches_every_connection_identically() -> None:
    """``broadcast`` extends each batch with the same messages, in order."""
    outbox = TickOutbox((9, 20))

    outbox.broadcast([_remove(11), _remove(12)])

    assert outbox.batches() == {9: [_remove(11), _remove(12)], 20: [_remove(11), _remove(12)]}


def test_narration_is_called_once_per_observer_with_its_own_id() -> None:
    """``narrate`` hands each connection the narrator's answer for it."""
    outbox = TickOutbox((9, 20))
    asked: list[int] = []

    def narrator(observer_id: int) -> list[BinaryMessage]:
        asked.append(observer_id)
        return [_remove(observer_id)]

    outbox.narrate(narrator)

    assert asked == [9, 20]
    assert outbox.batches() == {9: [_remove(9)], 20: [_remove(20)]}


def test_the_three_modes_interleave_in_call_order() -> None:
    """One connection's batch is the calls that reached it, in order."""
    outbox = TickOutbox((9,))

    outbox.broadcast([_remove(11)])
    outbox.to(9, SyncDict(msg_type=0x3F))
    outbox.narrate(lambda observer_id: [_remove(observer_id)])

    assert outbox.batch(9) == [_remove(11), SyncDict(msg_type=0x3F), _remove(9)]


def test_a_batch_is_live() -> None:
    """Appending to ``batch`` appends to the outbox — the holders rely on it."""
    outbox = TickOutbox((9,))

    outbox.batch(9).append(_remove(11))

    assert outbox.batches() == {9: [_remove(11)]}


def test_admitting_a_connection_twice_is_refused() -> None:
    """Two batches for one tank would deliver every broadcast twice."""
    outbox = TickOutbox((9,))

    with pytest.raises(SimError, match="tank 9 already has a batch"):
        outbox.admit(9)
    with pytest.raises(SimError, match="tank 20 already has a batch"):
        TickOutbox((20, 20))


def test_the_batch_of_an_unconnected_tank_is_refused() -> None:
    """Asking for a connection's own batch where none exists is a bug."""
    outbox = TickOutbox((9,))

    with pytest.raises(SimError, match="tank 500 has no connection"):
        outbox.batch(500)
