"""The networked server's HTTP side: the browser client and what it fetches.

``tankpit-sim-serve`` takes players over WebSockets, and the same port
answers a plain HTTP ``GET`` too, so one route behind Traefik carries
both the game and the client that plays it. A request that asks to
upgrade is left to the WebSocket handshake; every other request is
answered here:

- ``/`` is the play page, ``play.html`` in the web root;
- ``/dist/<module>.js`` is one module of the built TypeScript client;
- ``/terrain/<room>`` is that room's static terrain, one class byte per
  tile, row by row from the north-west corner: 0 ground, 1 rock, 2
  water, the three classes the game's client samples its map into
  ([[terrain-system]]). It is read off the room's own
  :class:`~tankpit_bot.terrain.TerrainMap`, the terrain the sim plays
  on, so the client draws the field the server moves tanks over rather
  than a second classification of the image;
- ``/cipher-key`` is the static key every connection's XOR table is
  built from, which the client needs to read the 0x2E batches and to
  write its commands.

The web package carries neither the key nor any field: both reach the
browser from this server at run time, which keeps the TypeScript tree
free of anything taken from the game.

A server started without ``--web-root`` serves the terrain and the key
but no page; a request for the page then says so by name
(``SIM_WEB_NO_ROOT``), rather than answering an empty 404.
"""

from __future__ import annotations

import re
from http import HTTPStatus
from pathlib import Path
from typing import Final

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.capture.xor import require_static_key
from tankpit_bot.sim.net_room import NetRoom

FIELD_SPAN: Final[int] = 256
"""Tiles along each side of every field."""

PAGE_FILE: Final[str] = "play.html"
"""The play page, in the web root."""

CIPHER_KEY_PATH: Final[str] = "/cipher-key"
"""Where the static key is served."""

TERRAIN_GROUND: Final[int] = 0
TERRAIN_ROCK: Final[int] = 1
TERRAIN_WATER: Final[int] = 2
"""The class bytes of a terrain response, the game client's own three."""

_MODULE_PATH = re.compile(r"^/dist/([a-z_]+\.js)$")
_TERRAIN_PATH = re.compile(r"^/terrain/([A-Za-z0-9]+)$")


def is_upgrade(request: Request) -> bool:
    """Report whether a request asks for the WebSocket handshake.

    Args:
        request: The parsed request.

    Returns:
        True when its ``Upgrade`` header names ``websocket``.
    """
    return request.headers.get("Upgrade", "").lower() == "websocket"


def terrain_classes(terrain: TerrainMapProtocol) -> bytes:
    """Read a field's static terrain as one class byte per tile.

    Args:
        terrain: The room's terrain.

    Returns:
        ``FIELD_SPAN * FIELD_SPAN`` bytes, row by row: 0 ground, 1 rock,
        2 water.

    Raises:
        ValueError: If the terrain names a tile on the field with a class
            the game does not have (``SIM_WEB_TERRAIN``).
    """
    classes = {
        terrain.GROUND: TERRAIN_GROUND,
        terrain.ROCK: TERRAIN_ROCK,
        terrain.WATER: TERRAIN_WATER,
    }
    out = bytearray(FIELD_SPAN * FIELD_SPAN)
    for y in range(FIELD_SPAN):
        for x in range(FIELD_SPAN):
            value = terrain.get_terrain(x, y)
            if value not in classes:
                raise ValueError(f"SIM_WEB_TERRAIN: tile ({x}, {y}) is {value!r}, not a class")
            out[y * FIELD_SPAN + x] = classes[value]
    return bytes(out)


def _ok(content_type: str, body: bytes) -> Response:
    """A 200 response carrying a body.

    Modules and the page are sent ``no-store``: a redeploy changes them
    under the same names, and a stale module beside a fresh one is a
    client that cannot load.

    Args:
        content_type: The body's media type.
        body: The body.

    Returns:
        The response.
    """
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(body))
    headers["Cache-Control"] = "no-store"
    return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, body)


class WebPages:
    """What the server answers a browser that is not yet a player."""

    def __init__(self, web_root: Path | None, rooms: tuple[NetRoom, ...]) -> None:
        """Bind the pages to a built web package and the server's rooms.

        Args:
            web_root: The directory holding ``play.html`` and ``dist/``,
                or None for a server that serves no page.
            rooms: The rooms whose terrain may be asked for.
        """
        self._web_root = web_root
        self._rooms = {room.info["room_id"]: room for room in rooms}
        self._terrain: dict[str, bytes] = {}

    def answer(self, connection: ServerConnection, request: Request) -> Response | None:
        """Answer one HTTP request, or leave it to the WebSocket handshake.

        This is the listener's ``process_request``: returning None lets
        the handshake go on; a response ends the request with it.

        Args:
            connection: The connection the request arrived on.
            request: The parsed request.

        Returns:
            None for an upgrade, else the response.
        """
        if is_upgrade(request):
            return None
        path = request.path.split("?", 1)[0]
        if path == CIPHER_KEY_PATH:
            return _ok("text/plain; charset=us-ascii", require_static_key().encode("ascii"))
        terrain = _TERRAIN_PATH.match(path)
        if terrain is not None:
            return self._terrain_response(connection, terrain.group(1))
        module = _MODULE_PATH.match(path)
        if path == "/" or module is not None:
            return self._web_file(connection, "dist/" + module.group(1) if module else PAGE_FILE)
        return connection.respond(HTTPStatus.NOT_FOUND, f"SIM_WEB_NOT_FOUND: {path}\n")

    def _terrain_response(self, connection: ServerConnection, room_id: str) -> Response:
        """Answer a terrain request, reading the room's terrain once.

        Args:
            connection: The connection the request arrived on.
            room_id: The room asked for.

        Returns:
            The class bytes, or a 404 naming the room.
        """
        room = self._rooms.get(room_id)
        if room is None:
            return connection.respond(
                HTTPStatus.NOT_FOUND, f"SIM_WEB_NO_ROOM: this server hosts no room {room_id!r}\n"
            )
        if room_id not in self._terrain:
            self._terrain[room_id] = terrain_classes(room.server.terrain)
        return _ok("application/octet-stream", self._terrain[room_id])

    def _web_file(self, connection: ServerConnection, relative: str) -> Response:
        """Answer the page or one module from the web root.

        Args:
            connection: The connection the request arrived on.
            relative: The file's path under the web root.

        Returns:
            The file, or a 404 naming why it is not served.
        """
        if self._web_root is None:
            return connection.respond(
                HTTPStatus.NOT_FOUND,
                "SIM_WEB_NO_ROOT: this server was started without --web-root\n",
            )
        path = self._web_root / relative
        if not _test_hooks.path_exists(path):
            return connection.respond(
                HTTPStatus.NOT_FOUND, f"SIM_WEB_NOT_FOUND: {relative} is not in the web root\n"
            )
        content_type = "text/html; charset=utf-8" if relative == PAGE_FILE else "text/javascript"
        return _ok(content_type, _test_hooks.read_bytes_from(path, 0))


__all__ = [
    "CIPHER_KEY_PATH",
    "FIELD_SPAN",
    "PAGE_FILE",
    "TERRAIN_GROUND",
    "TERRAIN_ROCK",
    "TERRAIN_WATER",
    "WebPages",
    "is_upgrade",
    "terrain_classes",
]
