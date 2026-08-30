"""Law 0 — the lobby: the half of the connection before play starts.

The sim used to begin INSIDE a room. Its first frame was the join
burst, so the bot's entire join path — room discovery, select, join
confirm, room entry, the enter response, the autoscroll toggle, quit —
was exercised only against the live server. That is 1,571 archived
frames with no sim counterpart, and every one of them is a code path a
soak could not regress ([[session-state-deglobalisation]]).

Lobby frames are unlike everything else on this wire: PLAINTEXT,
un-XORed, and NOT wrapped in a 0x2E envelope. They are the only frames
the transport writes in the clear.

The exchange, read off 285 archived sessions (every one identical in
shape):

    --> 0x25  %AUTH !be <account>|<hash>|<stamp> <magic>
    <-- 0x2B  +1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026
    <-- 0x2B  +5|World (Desert)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026
    --> 0x2A  *5                      (select — the bot probes rooms)
    <-- 0x3D  =5|Sep. 25, 2012|Artax|4|9|9|9|9
    --> 0x2A  *1
    <-- 0x3D  =1|Jan. 08, 2013|Artax|1|9|9|9|9
    --> 0x2B  +1|2|128|128|<XOR'd metadata>
    <-- 0x24  $1|0
    ... play begins, everything from here is a 0x2E envelope

Two rooms are advertised in every session and the enter response is
``$1|0`` in all 286 archived entries — the bot always lands in
Practice. Room lists carry eight pipe-delimited fields and so do join
confirms.

Mid-session the same plaintext channel carries two more exchanges, and
BOTH are echoes: the autoscroll toggle (``A1``/``A0``, echoed back
verbatim — 38 and 37 in the archive) and quit (``-``, echoed at a
median 62 ms in 67 of the 74 sessions that carry one). The received
quit is the server acknowledging the client's own q-press, not a
server-initiated kick — which is what it looked like until the pairing
was measured.

The AUTH frame is the PAGE CLIENT's, not the bot's — our code only
reads it, to lift the session magic (``codec.extract_magic_from_auth``).
:class:`SimCDPSession` stands in for the page client, so the sim opens
its link by sending it, and the magic reaches the bot the way it does
live: off the wire.
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.parser import RoomInfo
from tankpit_bot.protocol.decoders import try_decode_plaintext_ack

AUTH_PREFIX = "%AUTH"
"""Lead of the page client's authentication frame (archive: 285/285)."""

ROOM_LIST_PREFIX = "+"
SELECT_PREFIX = "*"
ENTER_PREFIX = "+"
JOIN_CONFIRM_PREFIX = "="
ENTER_RESPONSE_PREFIX = "$"
QUIT_BODY = b"-"

ENTER_RESPONSE_CODE = 0
"""The second field of the enter response — ``$1|0`` in all 286 archived entries."""


class SimAccountDict(TypedDict):
    """The account fields a join confirm reports back.

    ``active_forces`` is the four trailing counts of ``=room|date|
    name|rank|orange|purple|blue|red`` -- tanks playing each color in
    the room. All 9s is a world with no humans in it: there are always
    9 bots per color (operator, 2026-08-28), so the archive's 9s and
    10s are the standing bots plus the odd human.
    """

    game_start: str
    name: str
    rank: int
    active_forces: tuple[int, int, int, int]


SIM_ROOMS: tuple[RoomInfo, ...] = (
    RoomInfo(
        room_id="1",
        name="Practice",
        field_id=1,
        game_modes="0,0,0,0,0,0,0",
        default_troop=2,
        mode_code="p",
        image="field01.gif",
        year="2026",
    ),
    RoomInfo(
        room_id="5",
        name="World (Desert)",
        field_id=5,
        game_modes="1,1,1,0,1,0,0",
        default_troop=2,
        mode_code="n",
        image="field05.gif",
        year="2026",
    ),
)
"""The advertised rooms, copied from the archive's own rows.

Two rooms in every archived session, and the bot resolves ``Practice``
out of them by name — a one-room list would not exercise that.
Practice is the only one the sim can actually seed: the mined container
atlas covers ``field01`` alone, so entering ``5`` yields real terrain
and an empty world."""


SIM_ACCOUNT: SimAccountDict = SimAccountDict(
    game_start="Jan. 08, 2013",
    name="red-9",
    rank=1,
    active_forces=(9, 9, 9, 9),
)
"""The account the sim's join confirms report.

``name`` matches the client tank's own wire name (``make_sim_tank``'s
practice shape for id 9) so the lobby and the room agree about who
just joined; ``rank`` matches the tank's. The game-start date and
active force counts are the archive's own — nothing downstream reads them, and the
only durable effect of a join confirm is that the room was accepted."""


def _room_list_frame(room: RoomInfo) -> bytes:
    """Render one advertised room as its ROOM_LIST body.

    Args:
        room: The advertised room.

    Returns:
        The plaintext frame body.
    """
    fields = "|".join(
        (
            room["room_id"],
            room["name"],
            str(room["field_id"]),
            room["game_modes"],
            str(room["default_troop"]),
            room["mode_code"],
            room["image"],
            room["year"],
        )
    )
    return f"{ROOM_LIST_PREFIX}{fields}".encode()


def _join_confirm_frame(room_id: str, account: SimAccountDict) -> bytes:
    """Render the join confirm for one selected room.

    Args:
        room_id: The room the client selected.
        account: The account whose standing the confirm reports.

    Returns:
        The plaintext frame body.
    """
    active_forces = "|".join(str(count) for count in account["active_forces"])
    return (
        f"{JOIN_CONFIRM_PREFIX}{room_id}|{account['game_start']}|"
        f"{account['name']}|{account['rank']}|{active_forces}"
    ).encode()


class SimLobby:
    """The pre-play protocol: room list, select, enter, toggles, quit.

    One instance per connection. It holds no world — the world is the
    :class:`~tankpit_bot.sim.server.SimServer`'s — and reports which
    room the client entered so the caller knows when play starts.
    """

    def __init__(
        self,
        account: SimAccountDict,
        rooms: tuple[RoomInfo, ...] = SIM_ROOMS,
    ) -> None:
        """Bind the lobby to an account and the rooms it advertises.

        Args:
            account: The account the join confirms report.
            rooms: The advertised rooms, in ROOM_LIST order.
        """
        self._account = account
        self._rooms = rooms
        self.entered_room_id: str | None = None
        """The room the client entered, or ``None`` before entry."""
        self.quit = False
        """Whether the client sent the plaintext quit frame."""

    def _room(self, room_id: str) -> RoomInfo | None:
        """Find one advertised room by id.

        Args:
            room_id: The id the client named.

        Returns:
            The room, or ``None`` when nothing advertises that id.
        """
        for room in self._rooms:
            if room["room_id"] == room_id:
                return room
        return None

    def handle_frame(self, body: bytes) -> list[bytes]:
        """Answer one plaintext client frame.

        Args:
            body: The frame body, including its lead byte.

        Returns:
            The server's plaintext reply frames, in order — empty when
            the frame needs no answer.
        """
        ack = try_decode_plaintext_ack(body)
        if ack is not None:
            # The server echoes the toggle back verbatim, un-XORed.
            # 0x41 and 0x43 are both overloaded with binary frames, so
            # the discrimination is the production predicate's, not a
            # local re-reading of the two bytes.
            return [body]
        if body == QUIT_BODY:
            # The server ECHOES the quit, exactly as it echoes the
            # autoscroll toggle. 67 of the 74 archived sessions that
            # carry a quit show ``--> <--`` at a median 62 ms; the
            # received frame is the acknowledgement of the client's own
            # q-press, not a server-initiated kick
            # ([[session-state-deglobalisation]]).
            self.quit = True
            return [body]
        text = body.decode("utf-8", errors="replace")
        if text.startswith(AUTH_PREFIX):
            return [_room_list_frame(room) for room in self._rooms]
        if text.startswith(SELECT_PREFIX):
            return self._select(text[1:].strip())
        if text.startswith(ENTER_PREFIX):
            return self._enter(text[1:])
        return []

    def _select(self, room_id: str) -> list[bytes]:
        """Answer a room selection with its join confirm.

        Args:
            room_id: The selected room's id.

        Returns:
            The join-confirm frame, or nothing for an unknown room.
        """
        if self._room(room_id) is None:
            return []
        return [_join_confirm_frame(room_id, self._account)]

    def _enter(self, fields: str) -> list[bytes]:
        """Answer a room-entry request and mark the client entered.

        The request is ``room|troop|preview_x|preview_y|metadata``; the
        metadata tail is XOR-encoded by the client and the server does
        not echo it, so only the room id is read here.

        Args:
            fields: The request body after the ``+`` prefix.

        Returns:
            The enter-response frame, or nothing for an unknown room.
        """
        room_id = fields.split("|", 1)[0]
        if self._room(room_id) is None:
            return []
        self.entered_room_id = room_id
        return [f"{ENTER_RESPONSE_PREFIX}{room_id}|{ENTER_RESPONSE_CODE}".encode()]


def build_auth_frame(account_id: str, token: str, stamp: str, magic: str) -> bytes:
    """Render the page client's AUTH frame.

    Shape from the archive (285/285 identical):
    ``%AUTH !be <account>|<hash>|<stamp> <magic>``. The trailing token
    IS the session magic — ``codec.extract_magic_from_auth`` lifts it
    from exactly here, which is how the bot learns the cipher live.

    Args:
        account_id: The account's numeric id.
        token: The session hash the real server issues.
        stamp: The frame's numeric stamp field.
        magic: The session magic this connection's cipher is built from.

    Returns:
        The plaintext frame body.
    """
    return f"{AUTH_PREFIX} !be {account_id}|{token}|{stamp} {magic}".encode()


__all__ = [
    "AUTH_PREFIX",
    "ENTER_PREFIX",
    "ENTER_RESPONSE_CODE",
    "ENTER_RESPONSE_PREFIX",
    "JOIN_CONFIRM_PREFIX",
    "QUIT_BODY",
    "ROOM_LIST_PREFIX",
    "SELECT_PREFIX",
    "SIM_ACCOUNT",
    "SIM_ROOMS",
    "SimAccountDict",
    "SimLobby",
    "build_auth_frame",
]
