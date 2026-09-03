"""Diff the real server's response shapes against the sim's.

Pair every SENT command with the self-caused messages the server
answered inside that command's window, reduce them to an ordered tuple
of tokens, and compare the distribution between the real archive and a
sim archive. A shape only the real server produces is a MISSING law; a
shape only the sim produces is an INVENTED one ([[capture-differ]]).

This is the durable form of the retired
``analysis_scripts/diff_server_laws.py``, recovered byte-exact from the
``source_git_blobs`` pin its wiki page carries — the retirement
convention exists so a one-shot miner's provenance survives its
deletion, and it did. The method (window law, token alphabet) is that
script's; the packaging, typing and tests are new, so the next reader
gets a module instead of a blob hash.

NOT carried over: the numeric law checks that rode the original's same
pass (teleport cost, the window-bound acceptance law). They answer a
different question from shape fidelity and belong to their own
validators; stating the omission so nobody assumes this covers them.

The session walk is not re-implemented — :mod:`analysis.scan` owns it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from tankpit_bot.analysis.response_shapes_types import (
    CAUSE_COMMAND_NEVER_SENT,
    CAUSE_LIVE_SILENT_WINDOW,
    CAUSE_SHAPE_NEVER_ASSEMBLED,
    CAUSE_SIM_ONLY,
    CAUSE_TOKEN_NEVER_EMITTED,
    VERDICT_INVENTED_LAW,
    VERDICT_MISSING_LAW,
    CommandShapesDict,
    CommandWindowDict,
    ResponseShapeDiffDict,
    ShapeCountDict,
    ShapeDivergenceDict,
)
from tankpit_bot.analysis.scan import scan_archive
from tankpit_bot.analysis.types import DecodedFrameDict, ScannedSessionDict
from tankpit_bot.protocol import try_decode_binary_message, try_decode_plaintext_ack
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.wire.helpers import DecodeError

#: A window closes at the next SENT command, or after this many
#: milliseconds of wall clock. The cap matters in both directions: sim
#: sessions compress to sub-second wall time, so an uncapped window
#: would swallow a whole session, and live deferred teleports otherwise
#: bleed into the preceding map_open's window.
WINDOW_MS: Final = 3000

_SHOOT: Final = 0x53
_MOVEMENT: Final = 0x47
_POSITION: Final = 0x3D
_TANK_INFO: Final = 0x21
_SUPERVISOR: Final = 0x52

#: Message types that become a token naming no field. Ordered as the
#: original alphabet: radar pair, viewport, inventory/fuel/equipment,
#: map and chat.
_PLAIN_TOKENS: Final[dict[int | str, str]] = {
    0x4F: "4F",
    0x46: "46",
    0x5A: "5A",
    0x49: "49",
    0x44: "44",
    0x64: "64",
    0x67: "67",
    0x4C: "4C",
    0x4D: "4D",
    "teleport_landed": "landed",
    "container_pickup": "pickup",
}


def shape_token(message: BinaryMessage, self_id: int | None) -> str | None:
    """Reduce one received message to a shape token.

    The alphabet is the SELF-CAUSED set: echoes and results a client's
    own command can draw. Broadcasts about other tanks and the periodic
    status syncs are background, not response, and returning None drops
    them from every shape.

    A token is a message type EXCEPT where fields carry semantics the
    type does not: the three self-scoped tokens name their tank, and
    the 0x52 carries its error code AND its two client directives.

    Args:
        message: One decoded received message.
        self_id: The capturing client's tank id, or None before the
            session's first 0x21 has been seen. While None, the three
            self-scoped tokens cannot be attributed and are dropped
            rather than guessed.

    Returns:
        The token, or None when the message is background.
    """
    # Each test reads ``message["msg_type"]`` directly: narrowing the
    # BinaryMessage union through an intermediate variable does not
    # propagate to ``message``, and the self-scoped tokens read fields
    # only one member owns.
    if message["msg_type"] == _SHOOT:
        return "53self" if message["shooter_id"] == self_id else None
    if message["msg_type"] == _MOVEMENT:
        return "47self" if message["tank_id"] == self_id else None
    if message["msg_type"] == _POSITION:
        return "3Dself" if message["tank_id"] == self_id else None
    if message["msg_type"] == _SUPERVISOR:
        # The 0x52 token carries its FIELDS, not just its code. Both
        # are directives the real client acts on — reset_action
        # "reset to idle", close_map "close map view"
        # ([[decode-coverage]]) — so a sim that sends the right code
        # with the wrong fields drives a real client wrongly while
        # looking identical to a code-only differ. It looked identical
        # for months: the out-of-window move refusal sent (1, 0) where
        # every one of the archive's 10 move code-0 windows sends
        # (0, 1), and nothing could see it because this function threw
        # the fields away (found 2026-09-02 by a hand sweep, which is
        # not a mechanism).
        return f"52c{message['error_code']}r{message['reset_action']}m{message['close_map']}"
    return _PLAIN_TOKENS.get(message["msg_type"])


def _decoded_frame(frame: DecodedFrameDict) -> BinaryMessage | None:
    """Decode one received frame, or None when it is not binary.

    Plaintext acks and text-route frames are discriminated BEFORE the
    cipher, exactly as production does: 0x3D is dual-use (lobby text
    vs. binary MovementResponse), so decoding a text frame as binary
    would manufacture a ``3Dself`` token out of lobby chatter.

    Args:
        frame: One received frame from the scan.

    Returns:
        The decoded message, or None when the frame is plaintext,
        text-route, of an unknown type, or malformed.
    """
    if try_decode_plaintext_ack(frame["raw"]) is not None:
        return None
    if _is_text_route(frame["msg_type"], frame["raw"]):
        return None
    try:
        return try_decode_binary_message(frame["msg_type"], frame["body"])
    except DecodeError:
        # A short frame of a known type is archive noise, counted by
        # the recipient-policy sweep; here it simply contributes no
        # token ([[recipient-policy]]).
        return None


def _frame_time(frame: DecodedFrameDict) -> int:
    """Sort key: one frame's capture time.

    A named function rather than a lambda because a lambda's parameter
    is untyped, and an untyped subscript leaks ``Any`` through the
    strict rules this codebase runs under.

    Args:
        frame: The frame to order.

    Returns:
        The frame's capture time in milliseconds.
    """
    return frame["timestamp_ms"]


def _opened_window(frame: DecodedFrameDict) -> CommandWindowDict | None:
    """Open a window for one sent frame, if it carries a command.

    The lobby shares the socket, so only ``!``-prefixed frames are
    commands; a frame whose body will not decode opens nothing rather
    than opening a window attributed to the wrong kind.

    Args:
        frame: One sent frame from the scan.

    Returns:
        A fresh empty window, or None when the frame is not a
        decodable command.
    """
    if frame["msg_type"] != COMMAND_PREFIX:
        return None
    try:
        command = decode_client_command(frame["body"])
    except DecodeError:
        return None
    return CommandWindowDict(
        command_kind=command["kind"], shape=[], timestamp_ms=frame["timestamp_ms"]
    )


def mine_session(scanned: ScannedSessionDict) -> list[CommandWindowDict]:
    """Pair every sent command in one session with the shape it drew.

    A window opens on a SENT command and closes at the next one, or
    when :data:`WINDOW_MS` of wall clock has passed — whichever comes
    first. Only one window is open at a time, so a token belongs to
    exactly one command.

    Args:
        scanned: One session's decoded frames, from
            :func:`analysis.scan.scan_session`.

    Returns:
        One entry per sent command, in capture order.
    """
    windows: list[CommandWindowDict] = []
    open_window: CommandWindowDict | None = None
    self_id: int | None = None

    for frame in sorted(scanned["frames"], key=_frame_time):
        if frame["direction"] == "sent":
            fresh = _opened_window(frame)
            if fresh is None:
                continue
            if open_window is not None:
                windows.append(open_window)
            open_window = fresh
            continue

        message = _decoded_frame(frame)
        if message is None:
            continue
        if message["msg_type"] == _TANK_INFO and self_id is None:
            self_id = message["tank_id"]
        if (
            open_window is not None
            and frame["timestamp_ms"] - open_window["timestamp_ms"] > WINDOW_MS
        ):
            windows.append(open_window)
            open_window = None
        token = shape_token(message, self_id)
        if token is not None and open_window is not None:
            open_window["shape"].append(token)

    if open_window is not None:
        windows.append(open_window)
    return windows


def tally(windows: list[CommandWindowDict]) -> list[CommandShapesDict]:
    """Reduce paired windows to a per-command shape distribution.

    Args:
        windows: Every paired window from one archive.

    Returns:
        One distribution per command kind, kinds sorted by name and
        shapes descending by count.
    """
    counts: dict[str, Counter[tuple[str, ...]]] = {}
    for window in windows:
        counts.setdefault(window["command_kind"], Counter())[tuple(window["shape"])] += 1
    return [
        CommandShapesDict(
            command_kind=kind,
            windows=sum(shapes.values()),
            shapes=[
                ShapeCountDict(shape=list(shape), count=count)
                for shape, count in shapes.most_common()
            ],
        )
        for kind, shapes in sorted(counts.items())
    ]


def diff_shapes(
    live: list[CommandShapesDict],
    sim: list[CommandShapesDict],
) -> list[ShapeDivergenceDict]:
    """Find every shape present on exactly one side.

    Args:
        live: The real archive's distribution.
        sim: The sim archive's distribution.

    Returns:
        One row per one-sided shape: missing laws first (the real
        server does it and the sim does not), then invented ones, each
        group descending by the count that makes it notable.
    """
    live_index = _index(live)
    sim_index = _index(sim)
    vocabulary = _sim_vocabulary(sim)
    rows: list[ShapeDivergenceDict] = []
    for key, count in live_index.items():
        if key not in sim_index:
            rows.append(
                ShapeDivergenceDict(
                    command_kind=key[0],
                    shape=list(key[1]),
                    live_count=count,
                    sim_count=0,
                    verdict=VERDICT_MISSING_LAW,
                    cause=_missing_cause(key[0], key[1], vocabulary),
                )
            )
    for key, count in sim_index.items():
        if key not in live_index:
            rows.append(
                ShapeDivergenceDict(
                    command_kind=key[0],
                    shape=list(key[1]),
                    live_count=0,
                    sim_count=count,
                    verdict=VERDICT_INVENTED_LAW,
                    cause=CAUSE_SIM_ONLY,
                )
            )
    rows.sort(
        key=lambda row: (
            row["verdict"] != VERDICT_MISSING_LAW,
            -(row["live_count"] + row["sim_count"]),
            row["command_kind"],
        )
    )
    return rows


def _sim_vocabulary(sim: list[CommandShapesDict]) -> dict[str, frozenset[str]]:
    """What the sim was OBSERVED to emit, per command kind.

    Args:
        sim: The sim archive's distribution.

    Returns:
        Each command kind the sim sent, mapped to every token it was
        seen answering that command with. A kind absent from this map
        was never sent by the sim corpus at all.
    """
    tokens: dict[str, set[str]] = {}
    for entry in sim:
        seen = tokens.setdefault(entry["command_kind"], set())
        for shape in entry["shapes"]:
            seen.update(shape["shape"])
    return {kind: frozenset(seen) for kind, seen in tokens.items()}


def _missing_cause(
    command_kind: str,
    shape: tuple[str, ...],
    vocabulary: dict[str, frozenset[str]],
) -> str:
    """Classify WHY a live-only shape has no sim counterpart.

    A flat missing list mixes phenomena that need different work, and
    reads as one number: a timing artifact the sim cannot reproduce by
    construction, a command the corpus never sent, and an actual
    candidate gap all look identical in it. The order of the tests
    below is the order of confidence — the cheapest, most certain
    explanations are ruled out first, so a row reaches
    :data:`CAUSE_SHAPE_NEVER_ASSEMBLED` only when nothing weaker
    accounts for it.

    Args:
        command_kind: The command kind whose window produced it.
        shape: The live shape's ordered tokens.
        vocabulary: Per-kind sim token vocabulary from
            :func:`_sim_vocabulary`.

    Returns:
        One of the ``CAUSE_`` constants.
    """
    if not shape:
        return CAUSE_LIVE_SILENT_WINDOW
    emitted = vocabulary.get(command_kind)
    if emitted is None:
        return CAUSE_COMMAND_NEVER_SENT
    if set(shape) - emitted:
        return CAUSE_TOKEN_NEVER_EMITTED
    return CAUSE_SHAPE_NEVER_ASSEMBLED


def _index(distribution: list[CommandShapesDict]) -> dict[tuple[str, tuple[str, ...]], int]:
    """Flatten a distribution to a (kind, shape) -> count map.

    Args:
        distribution: One archive's per-command distribution.

    Returns:
        Every observed (command kind, shape) pair and its count.
    """
    return {
        (entry["command_kind"], tuple(shape["shape"])): shape["count"]
        for entry in distribution
        for shape in entry["shapes"]
    }


def _mine_directories(directories: list[Path]) -> tuple[int, list[CommandWindowDict]]:
    """Mine every decodable session under the given directories.

    Args:
        directories: Directories holding ``*.capture_session.json``.

    Returns:
        The count of sessions that decoded, and every paired window.
    """
    sessions = 0
    windows: list[CommandWindowDict] = []
    for directory in directories:
        for result in scan_archive(directory):
            if result["kind"] != "scanned":
                continue
            sessions += 1
            windows.extend(mine_session(result))
    return sessions, windows


def analyze_response_shapes(
    live_directories: list[Path],
    sim_directories: list[Path],
) -> ResponseShapeDiffDict:
    """Compare the real archive's response shapes against the sim's.

    Args:
        live_directories: Directories of real capture sessions.
        sim_directories: Directories of sim capture sessions.

    Returns:
        The comparison, with every one-sided shape.

    Raises:
        OSError: If a session file cannot be read.
        InvalidJsonError: If a session file is not valid JSON.
        JSONTypeError: If a session file is not a capture session.
    """
    live_sessions, live_windows = _mine_directories(live_directories)
    sim_sessions, sim_windows = _mine_directories(sim_directories)
    return ResponseShapeDiffDict(
        live_sessions=live_sessions,
        sim_sessions=sim_sessions,
        live_windows=len(live_windows),
        sim_windows=len(sim_windows),
        divergences=diff_shapes(tally(live_windows), tally(sim_windows)),
    )


#: Missing-side buckets, ordered least to most actionable. The
#: headings say what work each bucket implies, because the whole point
#: of the split is that they imply DIFFERENT work — and a flat list
#: reads as though they imply the same work, which is how 208 rows
#: looked like 208 sim gaps ([[capture-differ]]).
_MISSING_CAUSE_ORDER: tuple[tuple[str, str], ...] = (
    (
        CAUSE_LIVE_SILENT_WINDOW,
        "[timing, not a gap] the live window drew nothing; a tick-synchronous "
        "sim cannot produce this",
    ),
    (
        CAUSE_COMMAND_NEVER_SENT,
        "[corpus gap] the sim corpus never sent this command at all; "
        "fix the scenarios, not the server",
    ),
    (
        CAUSE_TOKEN_NEVER_EMITTED,
        "[check] the live shape holds a token the sim never emits here; "
        "genuine gap or window overlap",
    ),
    (
        CAUSE_SHAPE_NEVER_ASSEMBLED,
        "[READ THESE FIRST] the sim emits every token here, just never in this combination",
    ),
)


def _render_rows(rows: list[ShapeDivergenceDict], limit: int) -> list[str]:
    """Render divergence rows under a heading, capped and honest.

    Args:
        rows: The rows to render, already ordered.
        limit: Maximum rows to render. The cap is reported rather than
            silently truncating, so a reader knows what was dropped.

    Returns:
        The rendered lines.
    """
    lines = []
    for row in rows[:limit]:
        count = row["live_count"] + row["sim_count"]
        shape = " ".join(row["shape"]) if row["shape"] else "(silent)"
        lines.append(f"    n={count:<6d} {row['command_kind']:<18s} {shape}")
    if len(rows) > limit:
        lines.append(f"    ... {len(rows) - limit} more rows not shown (limit={limit})")
    return lines


def format_response_shape_diff(diff: ResponseShapeDiffDict, limit: int) -> str:
    """Format the comparison as a readable report.

    The INVENTED side leads, because it is the half that means
    something at any sample size: a shape the sim produced is one it
    can produce, however small the corpus. The MISSING side is split
    by cause — see :data:`_MISSING_CAUSE_ORDER` — because an
    undifferentiated missing list mixes a timing artifact the sim
    cannot reproduce by construction, commands the corpus never sent,
    and genuine candidates, and reports all three as one number.

    Args:
        diff: The comparison to format.
        limit: Maximum divergence rows to render per group. A cap is
            reported explicitly rather than silently truncating, so a
            reader knows what was dropped.

    Returns:
        Multi-line human-readable summary.
    """
    lines = [
        f"live_sessions={diff['live_sessions']} live_windows={diff['live_windows']}",
        f"sim_sessions={diff['sim_sessions']} sim_windows={diff['sim_windows']}",
    ]
    invented = [row for row in diff["divergences"] if row["verdict"] == VERDICT_INVENTED_LAW]
    missing = [row for row in diff["divergences"] if row["verdict"] == VERDICT_MISSING_LAW]
    lines.append("")
    lines.append(f"INVENTED LAWS (sim does it, archive never shows it): {len(invented)}")
    lines.extend(_render_rows(invented, limit))
    lines.append("")
    lines.append(f"MISSING LAWS (real server does it, sim never does): {len(missing)}")
    for cause, title in _MISSING_CAUSE_ORDER:
        rows = [row for row in missing if row["cause"] == cause]
        if not rows:
            continue
        behind = sum(row["live_count"] for row in rows)
        lines.append("")
        lines.append(f"  {title}")
        lines.append(f"    {len(rows)} rows, {behind} live windows behind them")
        lines.extend(_render_rows(rows, limit))
    return "\n".join(lines)


__all__ = [
    "WINDOW_MS",
    "analyze_response_shapes",
    "diff_shapes",
    "format_response_shape_diff",
    "mine_session",
    "shape_token",
    "tally",
]
