"""World message payloads: sync, viewport, terrain, and supervisor.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.world` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class SyncDict(TypedDict):
    """Sync/heartbeat (0x3F '?' message)."""

    msg_type: Literal[0x3F]


class CacheUpdateDict(TypedDict):
    """Cache-only tile patch (0x43 'C' message)."""

    msg_type: Literal[0x43]
    updates: list[tuple[int, int, int]]


class OverlayUpdateDict(TypedDict):
    """Overlay-only tile patch (0x40 '@' message)."""

    msg_type: Literal[0x40]
    updates: list[tuple[int, int, int]]


class ViewportEntityDict(TypedDict):
    """Single tile row in viewport update."""

    col: int
    row: int
    cache_value: int
    overlay_value: int
    terrain_type: int


class ViewportUpdateDict(TypedDict):
    """Viewport/map update (0x5A 'Z' message)."""

    msg_type: Literal[0x5A]
    viewport_left: int
    viewport_top: int
    entities: list[ViewportEntityDict]


class TerrainUpdateDict(TypedDict):
    """Terrain type update (0x4A 'J' message)."""

    msg_type: Literal[0x4A]
    updates: list[tuple[int, int, int]]


class SupervisorTextDict(TypedDict):
    """Free-form server text channel (0x3C '<' message).

    Trace-verified from tpclient.js wg.h (V['<']):
      ``wg.h(a) = new wg(p(a))`` -- a is the entire XOR-decoded body
      and ``p()`` is just byte-to-string conversion
      (``String.fromCharCode(a[i] & 255)``). The renderer prints:
      "Message from the Supervisor:\\n<message>\\n".

    Distinct from 0x52 CommandResult (V.R / xg) which carries a 3-byte
    error code; this is the server's freeform announcement channel.
    """

    msg_type: Literal[0x3C]
    message: str


class SupervisorDict(TypedDict):
    """Command failure response (0x52 'R' message).

    Trace-verified from tpclient.js xg.h (line 4317-4322):
      a[0] = reset_action (1=reset to idle)
      a[1] = close_map (1=close map view)
      a[2] = error_code (index into Gb[] error strings; 128+=custom text)
    """

    msg_type: Literal[0x52]
    reset_action: int
    close_map: int
    error_code: int


__all__ = [
    "CacheUpdateDict",
    "OverlayUpdateDict",
    "SupervisorDict",
    "SupervisorTextDict",
    "SyncDict",
    "TerrainUpdateDict",
    "ViewportEntityDict",
    "ViewportUpdateDict",
]
