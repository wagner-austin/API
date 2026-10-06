"""The browser client's session fixture is what the networked server sends.

``web/tests/sim_session.ts`` holds one whole session as the bytes on the
socket: the page client's AUTH, select, enter, enter-game and move, and
every message a real :class:`~tankpit_bot.sim.net_host.NetHost` answered,
through the join burst and two ticks of a walk on a practice room. The
TypeScript suite plays it through its wire client and checks what it draws
against where this server left the tank. This test records the session
again and requires the file to be exactly that, so a change to the server's
bytes cannot leave the browser client reading a stale copy.

The session is ciphered with a static key of this test's own making,
installed through ``_test_hooks.read_text`` for the key file alone, so the
fixture carries nothing of the game's key. To rewrite the file after a
deliberate change to the server, call :func:`write_fixture`.
"""

from __future__ import annotations

import base64
from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import ReadTextProtocol
from tankpit_bot.capture.xor import reset_static_key_cache
from tankpit_bot.resources import static_key_file_path
from tankpit_bot.sim.lobby import build_auth_frame
from tankpit_bot.sim.net_host import NetHost
from tankpit_bot.sim.net_room import NET_PLAYER_ID_BASE, NetRoom, field_room_info, open_field_room
from tests.sim._net_client import MAGIC, TOKEN, account_book, enter_game, move, payload, wire

FIXTURE = Path(__file__).resolve().parents[2] / "web" / "tests" / "sim_session.ts"

_KEY_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

TEST_KEY = "".join(_KEY_LETTERS[(index * 37) % len(_KEY_LETTERS)] for index in range(1000))
"""A 1000-character key of this test's own, as long as the game's: the
server refuses an envelope longer than its table. Letters and digits only,
so it sits in a TypeScript string as it is."""

ENTER = b"+1|2|128|128|web"
"""The enter frame the TypeScript client writes (``web/src/lobby.ts``)."""


def _with_test_key(real: ReadTextProtocol) -> ReadTextProtocol:
    """Read the static key file as :data:`TEST_KEY` and every other path as it is."""
    key_path = static_key_file_path()

    def read_text(path: Path) -> str:
        """The test key for the key file; the real contents otherwise."""
        return TEST_KEY if path == key_path else real(path)

    return read_text


def _walk_target(room: NetRoom, x: int, y: int) -> tuple[int, int]:
    """The first walkable tile two steps off the tank, inside its window."""
    for dx, dy in ((2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (-2, -1)):
        if room.server.terrain.is_passable(x + dx, y + dy):
            return x + dx, y + dy
    raise AssertionError(f"no walkable tile two steps from ({x}, {y})")


def record_session() -> str:
    """Play the session against a real host and render it as the TypeScript fixture.

    Returns:
        The fixture module's text.
    """
    room = open_field_room(
        field_room_info("1", "Practice", "field01.gif", practice=True),
        layout=None,
        population_seed=None,
    )
    host = NetHost((room,), account_book())
    connection = host.open()
    exchange: list[tuple[str, str]] = []

    def send(*bodies: bytes) -> None:
        """Send one message, keeping it and every reply the host answered with."""
        exchange.append(("client", wire(*bodies).hex()))
        for reply in host.receive(connection, payload(*bodies)):
            exchange.append(("server", base64.b64decode(reply).hex()))

    def tick() -> None:
        """Play one tick and keep the client's batch."""
        exchange.append(("server", base64.b64decode(host.tick()[connection]).hex()))

    send(build_auth_frame("1001", TOKEN, "0", MAGIC))
    send(b"*1")
    send(ENTER)
    send(enter_game())
    tick()
    tank = room.server.world["tanks"][NET_PLAYER_ID_BASE]
    target = _walk_target(room, tank["x"], tank["y"])
    send(move(*target))
    tick()
    tick()
    tank = room.server.world["tanks"][NET_PLAYER_ID_BASE]
    rows = "\n".join(f'    {{ from: "{side}", hex: "{data}" }},' for side, data in exchange)
    return (
        "/**\n"
        " * One session on the sim server's socket, as tests/sim/test_web_session.py\n"
        " * recorded it from a real NetHost: GENERATED, do not edit; that test\n"
        " * fails when the server's bytes change, and its write_fixture rewrites it.\n"
        " */\n\n"
        "/** One WebSocket message, who sent it, and its bytes as hex. */\n"
        "export interface SessionMessage {\n"
        '  readonly from: "client" | "server";\n'
        "  readonly hex: string;\n"
        "}\n\n"
        "export const SESSION = {\n"
        f'  key: "{TEST_KEY}",\n'
        f'  magic: "{MAGIC}",\n'
        f'  token: "{TOKEN}",\n'
        f"  tankId: {NET_PLAYER_ID_BASE},\n"
        f"  targetX: {target[0]},\n"
        f"  targetY: {target[1]},\n"
        f"  endX: {tank['x']},\n"
        f"  endY: {tank['y']},\n"
        "  exchange: [\n"
        f"{rows}\n"
        "  ] satisfies readonly SessionMessage[],\n"
        "};\n"
    )


def write_fixture() -> None:
    """Rewrite the fixture from the server as it is now."""
    _test_hooks.read_text = _with_test_key(_test_hooks.read_text)
    reset_static_key_cache()
    FIXTURE.write_text(record_session(), encoding="utf-8")


def test_the_fixture_is_the_session_the_server_plays() -> None:
    """Byte for byte: the browser client's fixture is today's server."""
    _test_hooks.read_text = _with_test_key(_test_hooks.read_text)
    reset_static_key_cache()
    assert FIXTURE.read_text(encoding="utf-8") == record_session()


def test_the_test_key_reaches_only_the_key_file(tmp_path: Path) -> None:
    """Every other path still reads its own contents."""
    other = tmp_path / "other.txt"
    other.write_text("own contents", encoding="utf-8")
    read_text = _with_test_key(_test_hooks.read_text)
    assert (read_text(static_key_file_path()), read_text(other)) == (TEST_KEY, "own contents")
