"""Radar message payloads: scan results and the entities they carry.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.radar` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class RadarResultDict(TypedDict):
    """Radar scan result (F message)."""

    msg_type: Literal[0x46]
    detection_type: int
    found: bool


class EnemyDetectionDict(TypedDict):
    """Enemy detection (H message)."""

    msg_type: Literal[0x48]
    tank_id: int
    x: int
    y: int
    rank: int
    team: int


class RadarContainerDict(TypedDict):
    """Container entry in radar scan result.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        volume: Fuel volume (0-32767), or -1 for equipment.
    """

    x: int
    y: int
    volume: int


class RadarMineDict(TypedDict):
    """Mine entry in radar scan result.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team that placed the mine (0=red, 1=purple, 2=blue, 3=orange).
    """

    x: int
    y: int
    team: int


class RadarMineClearDict(TypedDict):
    """Mine-clear entry in radar scan result.

    An overlay entry whose value is >= 8 (255 in the JS dh detonation
    handler) — the server's statement that the tile has NO mine. The
    JS ch handler writes the value into ``tile.m`` raw; 255 is the
    canonical no-mine sentinel it uses everywhere else.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
    """

    x: int
    y: int


class RadarScanResultDict(TypedDict):
    """Radar scan result (0x4F, JS handler ``ch`` / V.O).

    The 0x4F body is a batch of per-tile writes — a delta sync of the
    scanned area, not an append-only reveal list. Cache entries set a
    tile's container layer (0 = tile now empty, N = fuel volume,
    65535 -> -1 = equipment); overlay entries set the mine layer
    (0-7 = mine with ``team = value & 3``, >= 8 = no mine). Corpus
    scan 2026-07-03 (199 sessions, 1817 bodies): 247 of 2093 cache
    entries were removals (value 0); every body arrived tunneled
    inside 0x2E.

    Attributes:
        msg_type: Message type (0x4F).
        containers: Container entries (volume 0 = authoritative removal).
        mines: Mine entries (overlay value 0-7).
        mine_clears: Tiles the server declared mine-free (overlay >= 8).
    """

    msg_type: Literal[0x4F]
    containers: list[RadarContainerDict]
    mines: list[RadarMineDict]
    mine_clears: list[RadarMineClearDict]


__all__ = [
    "EnemyDetectionDict",
    "RadarContainerDict",
    "RadarMineClearDict",
    "RadarMineDict",
    "RadarResultDict",
    "RadarScanResultDict",
]
