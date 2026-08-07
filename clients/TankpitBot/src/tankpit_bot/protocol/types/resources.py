"""Resource message payloads: fuel, inventory, and equipment.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.resources` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class FuelGainDict(TypedDict):
    """Fuel gain event (D message). fuel_total is the new absolute fuel level.

    ``flag`` is the raw wire byte at offset 2; ``is_free`` derives from
    it (``flag == 0``). Corpus 2026-07-21 (295 samples): 294 bodies
    carry 0, one carries 0x2B — the byte is a value, not a boolean, so
    the encoder needs it verbatim for byte-identical round-trips.
    """

    msg_type: Literal[0x44]
    fuel_total: int
    is_free: bool
    flag: int


class FuelDepositDict(TypedDict):
    """Fuel deposit event (d message). fuel_total is the new absolute fuel level."""

    msg_type: Literal[0x64]
    fuel_total: int


class InventoryDict(TypedDict):
    """Inventory display (I message)."""

    msg_type: Literal[0x49]
    show: bool
    alternate: bool
    counts: list[int]
    enabled: list[bool]


class EquipmentGainDict(TypedDict):
    """Equipment gain (g message)."""

    msg_type: Literal[0x67]
    show_message: bool
    gained: list[int]


class EquipmentToggleDict(TypedDict):
    """Equipment toggle (t message)."""

    msg_type: Literal[0x74]
    enabled: list[bool]


__all__ = [
    "EquipmentGainDict",
    "EquipmentToggleDict",
    "FuelDepositDict",
    "FuelGainDict",
    "InventoryDict",
]
