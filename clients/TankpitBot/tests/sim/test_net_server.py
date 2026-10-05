"""The networked sim server over a real WebSocket, and its command line.

Every socket test listens on an ephemeral localhost port and talks to it
with a real WebSocket client sending the page client's bytes
(:mod:`tests.sim._net_client`). Leaving ``serve`` closes the listener and
waits for every handler, so what a handler's ``finally`` did is settled
by the time the test reads the host.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close, CloseCode

from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sim.net_accounts import encode_account_book
from tankpit_bot.sim.net_room import NET_PLAYER_ID_BASE, NetError
from tankpit_bot.sim.net_server import (
    DEFAULT_BIND,
    DEFAULT_PORT,
    DEFAULT_ROOM,
    NetServer,
    ServeArgs,
    build_host,
    main,
    parse_serve_args,
    serve_rooms,
)
from tests.sim._net_client import (
    ACCOUNT,
    ENTER,
    OTHER_ACCOUNT,
    SELECT,
    auth,
    enter_game,
    plaintext,
    received_kinds,
    wire,
)


def _args(
    accounts: Path, *, ticks: int | None = 0, rooms: tuple[str, ...] = ("1:field01:n",)
) -> ServeArgs:
    """Flags for a server on an ephemeral port."""
    return ServeArgs(
        bind="127.0.0.1",
        port=0,
        accounts=accounts,
        rooms=rooms,
        ticks=ticks,
        tick_ms=0,
        layout="bot-20260706-223721",
        population_seed=7,
    )


@pytest.fixture()
def accounts(tmp_path: Path) -> Path:
    """The two test accounts, in an account file on disk."""
    path = tmp_path / "accounts.json"
    path.write_text(dump_json_str(encode_account_book((ACCOUNT, OTHER_ACCOUNT))), encoding="utf-8")
    return path


async def _received(socket: ClientConnection) -> str:
    """The next message the server sent, as the wire's base64 payload.

    Raises:
        AssertionError: If the server sent text; the wire is binary.
    """
    message = await socket.recv()
    if isinstance(message, str):
        raise AssertionError(f"the server sent a text message: {message!r}")
    return base64.b64encode(message).decode("ascii")


async def test_a_client_joins_and_plays_over_a_real_socket(accounts: Path) -> None:
    """AUTH, lobby, entry and the join burst, all over WebSocket bytes."""
    server = NetServer(build_host(_args(accounts)))
    async with serve(server.handle, "127.0.0.1", 0) as listening:
        bound: tuple[str, int] = next(iter(listening.sockets)).getsockname()
        async with connect(f"ws://127.0.0.1:{bound[1]}") as socket:
            await socket.send(wire(auth()))
            assert plaintext([await _received(socket)]) == [
                "+1|World (field01)|1|1,1,1,0,1,0,0|2|n|field01.gif|2026"
            ]
            await socket.send(wire(SELECT, ENTER))
            assert plaintext([await _received(socket), await _received(socket)]) == [
                "=1|Oct. 05, 2026|austin|3|9|9|9|9",
                "$1|0",
            ]
            await socket.send(wire(enter_game()))
            # The pong comes back after the server has read everything sent
            # before the ping, so the command is queued when the tick runs.
            await (await socket.ping())
            assert server.tick() == 1
            burst = received_kinds([await _received(socket)])
            assert burst[:2] == [0x21, 0x3E]
            assert sorted(server.host.rooms[0].server.world["tanks"]) == [NET_PLAYER_ID_BASE]
    assert server.host.connections == 0
    assert server.host.rooms[0].server.world["tanks"] == {}


async def test_a_text_message_ends_only_that_connection(accounts: Path) -> None:
    """The wire is binary; the handler raises, the library closes with 1011."""
    server = NetServer(build_host(_args(accounts)))
    async with serve(server.handle, "127.0.0.1", 0) as listening:
        bound: tuple[str, int] = next(iter(listening.sockets)).getsockname()
        async with connect(f"ws://127.0.0.1:{bound[1]}") as socket:
            await socket.send("hello")
            with pytest.raises(ConnectionClosedError) as closed:
                await socket.recv()
            assert closed.value.rcvd == Close(CloseCode.INTERNAL_ERROR, "")
    assert server.host.connections == 0


async def test_the_server_ticks_for_its_count_then_returns(accounts: Path) -> None:
    """A tick count is how a smoke run ends; nobody seated is still a tick."""
    server = NetServer(build_host(_args(accounts)))
    assert await server.tick_for(3, 0) == 3
    assert server.host.rooms[0].server.world["tick"] == 3
    assert await serve_rooms(server, _args(accounts, ticks=2)) == 2
    assert server.host.rooms[0].server.world["tick"] == 5


def test_the_command_line_serves_its_rooms_for_its_ticks(
    accounts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tankpit-sim-serve`` builds the rooms, listens, ticks and reports."""
    argv = ["--accounts", str(accounts), "--port", "0", "--ticks", "2", "--tick-ms", "0"]
    assert main([*argv, "--room", "1:field01:n"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "sim server: 2 ticks served"


def test_flags_default_to_one_practice_room_on_localhost(accounts: Path) -> None:
    """Only the account file must be named."""
    assert parse_serve_args(["--accounts", str(accounts)]) == ServeArgs(
        bind=DEFAULT_BIND,
        port=DEFAULT_PORT,
        accounts=accounts,
        rooms=(DEFAULT_ROOM,),
        ticks=None,
        tick_ms=TICK_RATE_MS,
        layout=None,
        population_seed=None,
    )


def test_every_flag_is_read(accounts: Path) -> None:
    """Rooms repeat; the rest name one value each."""
    argv = ["--bind", "0.0.0.0", "--port", "9000", "--accounts", str(accounts)]
    argv += ["--room", "1:field01:p", "--room", "5:field05:n", "--ticks", "4", "--tick-ms", "50"]
    argv += ["--layout", "bot-20260706-223721", "--population-seed", "3"]
    assert parse_serve_args(argv) == ServeArgs(
        bind="0.0.0.0",
        port=9000,
        accounts=accounts,
        rooms=("1:field01:p", "5:field05:n"),
        ticks=4,
        tick_ms=50,
        layout="bot-20260706-223721",
        population_seed=3,
    )


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        (["--port", "1"], "--accounts PATH names this server's account file"),
        (["--accounts"], "unknown flag or missing value at '--accounts'"),
        (["--player", "x"], "unknown flag or missing value at '--player'"),
    ],
)
def test_bad_flags_are_refused_by_name(argv: list[str], says: str) -> None:
    """Usage errors say which flag."""
    with pytest.raises(NetError, match=f"SIM_SERVE_USAGE: {says}"):
        parse_serve_args(argv)


def test_rooms_are_built_from_their_specs(accounts: Path) -> None:
    """A practice room and an open room, each on its own field."""
    host = build_host(_args(accounts, rooms=("1:field01:p", "5:field05:n")))
    assert [(r.info["room_id"], r.info["name"], r.info["mode_code"]) for r in host.rooms] == [
        ("1", "Practice", "p"),
        ("5", "World (field05)", "n"),
    ]


@pytest.mark.parametrize("spec", ["1:field01", "1:field01:x", ":field01:p"])
def test_a_room_spec_that_is_not_id_field_mode_is_refused(accounts: Path, spec: str) -> None:
    """Three parts, an id, and mode p or n."""
    with pytest.raises(NetError, match=r"SIM_SERVE_ROOM: .* is not ID:FIELD:MODE"):
        build_host(_args(accounts, rooms=(spec,)))
