"""What a page client puts on the networked server's socket, and how it reads the answers.

The frames are built by the production builders and ciphered with the
production cipher, and what the server sends back is read through the
production capture decoder, so these tests hold the server to the wire
the bot actually speaks rather than to a second copy of it.
"""

from __future__ import annotations

import base64

from tankpit_bot.analysis.response_shapes import decode_received_frame
from tankpit_bot.analysis.scan import decode_session_frames
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol.commands import (
    CMD_ENTER_GAME,
    CMD_MOVE,
    COMMAND_PREFIX,
    TYPE_MOVEMENT,
    TYPE_QUERY,
)
from tankpit_bot.sim.lobby import QUIT_BODY, build_auth_frame
from tankpit_bot.sim.net_accounts import MemoryAccountBook, NetAccountDict, token_digest
from tankpit_bot.types import CapturedMessage, CaptureSession
from tankpit_bot.types.literals import MessageDirection
from tankpit_bot.wire.helpers import pack16

MAGIC = "netmagic5uk3et4epiexu"
OTHER_MAGIC = "netmagic7qq2zz9wvbnma"
TOKEN = "token-of-austin"
OTHER_TOKEN = "token-of-kestrel"

ACCOUNT = NetAccountDict(
    account_id="1001",
    token_sha256=token_digest(TOKEN),
    name="austin",
    rank=3,
    game_start="Oct. 05, 2026",
    decorations=[1, 2, 0, 0, 0, 0, 0, 0, 0],
)
OTHER_ACCOUNT = NetAccountDict(
    account_id="1002",
    token_sha256=token_digest(OTHER_TOKEN),
    name="kestrel",
    rank=0,
    game_start="Oct. 05, 2026",
    decorations=[0] * 9,
)


def account_book() -> MemoryAccountBook:
    """The two test accounts, in a book of their own copies, so a test may record into it."""
    return MemoryAccountBook(
        tuple(
            NetAccountDict(
                account_id=account["account_id"],
                token_sha256=account["token_sha256"],
                name=account["name"],
                rank=account["rank"],
                game_start=account["game_start"],
                decorations=list(account["decorations"]),
            )
            for account in (ACCOUNT, OTHER_ACCOUNT)
        )
    )


def payload(*bodies: bytes) -> str:
    """Frame bodies as one base64 wire payload, length-prefixed as the client sends them.

    Args:
        *bodies: Frame bodies, lead byte included.

    Returns:
        The payload.
    """
    return base64.b64encode(wire(*bodies)).decode("ascii")


def wire(*bodies: bytes) -> bytes:
    """Frame bodies as the bytes of one WebSocket message.

    Args:
        *bodies: Frame bodies, lead byte included.

    Returns:
        The length-prefixed frames, concatenated.
    """
    return b"".join(pack16(len(body)) + body for body in bodies)


def auth(account: NetAccountDict = ACCOUNT, token: str = TOKEN, magic: str = MAGIC) -> bytes:
    """The page client's AUTH frame for an account.

    Args:
        account: The account to join as.
        token: The token presented.
        magic: The session magic.

    Returns:
        The frame body.
    """
    return build_auth_frame(account["account_id"], token, "0", magic)


SELECT = b"*1"
ENTER = b"+1|2|128|128|metadata"
QUIT = QUIT_BODY


def command(magic: str, *plain: int) -> bytes:
    """One command frame, ``!`` then the ciphered bytes, as the page puts it on the wire.

    Args:
        magic: The session magic whose table ciphers it.
        *plain: The command's plaintext bytes.

    Returns:
        The frame body.
    """
    return bytes([COMMAND_PREFIX]) + xor_decode_body(bytes(plain), build_session_xor_table(magic))


def enter_game(magic: str = MAGIC) -> bytes:
    """CMD_ENTER_GAME, which asks for the join burst."""
    return command(magic, TYPE_QUERY, CMD_ENTER_GAME)


def move(x: int, y: int, magic: str = MAGIC) -> bytes:
    """A move command to a tile."""
    return command(magic, TYPE_MOVEMENT, CMD_MOVE, x, y)


def received_kinds(payloads: list[str], magic: str = MAGIC) -> list[int | str]:
    """The message types a client reads out of what it received, through the production decoder.

    Args:
        payloads: The base64 payloads received, in order.
        magic: The session magic.

    Returns:
        Each decoded binary message's type, in order; plaintext frames
        are not binary messages and contribute nothing.
    """
    session = CaptureSession(
        session_id="net",
        start_timestamp_ms=0,
        end_timestamp_ms=None,
        base_url="ws://sim",
        messages=[
            CapturedMessage(
                timestamp_ms=index,
                direction=MessageDirection.RECEIVED,
                payload=received,
                ws_url="ws://sim",
            )
            for index, received in enumerate(payloads)
        ],
        magic=magic,
        game_log=[],
        tank_names={},
    )
    kinds: list[int | str] = []
    for frame in decode_session_frames(session):
        message = decode_received_frame(frame)
        if message is not None:
            kinds.append(message["msg_type"])
    return kinds


def plaintext(payloads: list[str]) -> list[str]:
    """The lobby text a client reads out of plaintext payloads.

    Args:
        payloads: Base64 payloads of plaintext frames.

    Returns:
        Each frame as text, in order.
    """
    return [
        body.decode("utf-8") for received in payloads for body in split_payload_frames(received)
    ]
