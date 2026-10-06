"""The sim server's HTTP side, over a real listener.

Every request goes to an ephemeral localhost port that serves the
WebSocket handler with :class:`WebPages` as its ``process_request``,
exactly as ``serve_rooms`` wires it, and is read back as the raw HTTP a
browser receives.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from tankpit_bot.capture.xor import require_static_key
from tankpit_bot.sim.net_host import NetHost
from tankpit_bot.sim.net_room import field_room_info, open_field_room
from tankpit_bot.sim.net_server import NetServer
from tankpit_bot.sim.net_web import (
    FIELD_SPAN,
    TERRAIN_GROUND,
    TERRAIN_ROCK,
    TERRAIN_WATER,
    WebPages,
    terrain_classes,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim._net_client import account_book, auth, plaintext, wire

PAGE = b"<!doctype html><title>play</title>"
MODULE = b"export const played = true;\n"


class HttpReply:
    """One HTTP response as the bytes a browser read."""

    def __init__(self, raw: bytes) -> None:
        """Split a raw response into its status, headers and body.

        Args:
            raw: Everything the server sent before closing.
        """
        head, _, self.body = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        self.status = int(lines[0].split(" ")[1])
        self.headers = {
            name.lower(): value for name, _, value in (line.partition(": ") for line in lines[1:])
        }


def _server() -> NetServer:
    """One open room on field01 over the two test accounts."""
    room = open_field_room(
        field_room_info("1", "Arena", "field01.gif", practice=False),
        layout="bot-20260706-223721",
        population_seed=7,
    )
    return NetServer(NetHost((room,), account_book()))


@pytest.fixture()
def web_root(tmp_path: Path) -> Path:
    """A built web package: the page and one module."""
    (tmp_path / "dist").mkdir()
    (tmp_path / "play.html").write_bytes(PAGE)
    (tmp_path / "dist" / "play.js").write_bytes(MODULE)
    return tmp_path


async def _get(pages: WebPages, server: NetServer, *paths: str) -> list[HttpReply]:
    """GET each path from a listener answering through the pages.

    Args:
        pages: The pages under test.
        server: The server whose handler takes upgrades.
        *paths: The request paths, each on its own connection.

    Returns:
        Each response, in order.
    """
    replies: list[HttpReply] = []
    async with serve(server.handle, "127.0.0.1", 0, process_request=pages.answer) as listening:
        bound: tuple[str, int] = next(iter(listening.sockets)).getsockname()
        for path in paths:
            reader, writer = await asyncio.open_connection("127.0.0.1", bound[1])
            writer.write(f"GET {path} HTTP/1.1\r\nHost: sim\r\n\r\n".encode("ascii"))
            await writer.drain()
            replies.append(HttpReply(await reader.read()))
            writer.close()
            await writer.wait_closed()
    return replies


async def test_the_page_and_its_modules_come_from_the_web_root(web_root: Path) -> None:
    """``/`` is play.html, a query string aside; ``/dist/<name>.js`` is a module."""
    server = _server()
    page, queried, module = await _get(
        WebPages(web_root, server.host.rooms), server, "/", "/?room=1", "/dist/play.js"
    )
    assert (page.status, page.headers["content-type"], page.body) == (
        200,
        "text/html; charset=utf-8",
        PAGE,
    )
    assert page.headers["cache-control"] == "no-store"
    assert queried.body == PAGE
    assert (module.status, module.headers["content-type"], module.body) == (
        200,
        "text/javascript",
        MODULE,
    )


async def test_what_is_not_served_is_refused_by_name(web_root: Path) -> None:
    """A module not built, a name outside the module pattern, a room not hosted."""
    server = _server()
    missing, outside, no_room = await _get(
        WebPages(web_root, server.host.rooms),
        server,
        "/dist/absent.js",
        "/dist/../play.html",
        "/terrain/9",
    )
    assert (missing.status, missing.body) == (
        404,
        b"SIM_WEB_NOT_FOUND: dist/absent.js is not in the web root\n",
    )
    assert (outside.status, outside.body) == (404, b"SIM_WEB_NOT_FOUND: /dist/../play.html\n")
    assert (no_room.status, no_room.body) == (
        404,
        b"SIM_WEB_NO_ROOM: this server hosts no room '9'\n",
    )


async def test_a_server_without_a_web_root_serves_no_page_but_still_its_terrain() -> None:
    """The page and modules say why they are absent; the runtime data is still served."""
    server = _server()
    page, module, key = await _get(
        WebPages(None, server.host.rooms), server, "/", "/dist/play.js", "/cipher-key"
    )
    refusal = b"SIM_WEB_NO_ROOT: this server was started without --web-root\n"
    assert [(page.status, page.body), (module.status, module.body)] == [(404, refusal)] * 2
    assert (key.status, key.body) == (200, require_static_key().encode("ascii"))
    assert key.headers["content-type"] == "text/plain; charset=us-ascii"


async def test_a_room_s_terrain_is_its_terrain_map_read_once(web_root: Path) -> None:
    """One class byte per tile, the same bytes on every request."""
    server = _server()
    expected = terrain_classes(server.host.rooms[0].server.terrain)
    first, second = await _get(
        WebPages(web_root, server.host.rooms), server, "/terrain/1", "/terrain/1"
    )
    assert (first.status, first.headers["content-type"]) == (200, "application/octet-stream")
    assert first.body == second.body == expected
    assert len(expected) == FIELD_SPAN * FIELD_SPAN
    assert set(expected) == {TERRAIN_GROUND, TERRAIN_ROCK, TERRAIN_WATER}


async def test_an_upgrade_still_reaches_the_game(web_root: Path) -> None:
    """The pages step aside for the WebSocket handshake; AUTH draws the room list."""
    server = _server()
    pages = WebPages(web_root, server.host.rooms)
    async with serve(server.handle, "127.0.0.1", 0, process_request=pages.answer) as listening:
        bound: tuple[str, int] = next(iter(listening.sockets)).getsockname()
        async with connect(f"ws://127.0.0.1:{bound[1]}/") as socket:
            await socket.send(wire(auth()))
            reply = await socket.recv(decode=False)
            assert plaintext([base64.b64encode(reply).decode("ascii")]) == [
                "+1|Arena|1|1,1,1,0,1,0,0|2|n|field01.gif|2026"
            ]


def test_terrain_classes_follow_the_game_s_three() -> None:
    """Ground 0, rock 1, water 2, row by row from the north-west corner."""
    terrain = InMemoryTerrainMap({(3, 0): "#", (0, 2): "W"})
    classes = terrain_classes(terrain)
    assert (classes[3], classes[2 * FIELD_SPAN], classes[1]) == (
        TERRAIN_ROCK,
        TERRAIN_WATER,
        TERRAIN_GROUND,
    )


def test_a_tile_outside_the_three_classes_is_refused() -> None:
    """A terrain that names a fourth class is a broken map, said by tile."""
    with pytest.raises(ValueError, match=r"SIM_WEB_TERRAIN: tile \(4, 1\) is '\?'"):
        terrain_classes(InMemoryTerrainMap({(4, 1): "?"}))
