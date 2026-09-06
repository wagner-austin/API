"""Posting to the agent board as a SERVICE rather than as a session.

LIFTED OUT OF ``tools/hpc-wake`` ON 2026-09-06, when a second bridge needed the
same three things. The trigger was ``tools/fleet-wake``, which announces fleet
dispatch results the way hpc-wake announces Slurm ones; the two differ entirely
in what they read and not at all in how they identify themselves or how they
post.

WHAT IS HERE IS THE SUBTLE PART, NOT THE SHARED PART. Both bridges also group
their endings into posts, and that is deliberately NOT here: hpc-wake groups
Slurm closures by (submitter, project) and fleet-wake groups dispatch rows by
(agent, project), and a grouper parameterised over both would be harder to read
than either. What IS here is what a second copy would have got WRONG:

1. THE IDENTITY, and it is the one that bites permanently. The board binds a
   session id to an agent label on FIRST WRITE and never releases it (mig 415,
   ``assertSessionLabel``). A service has no harness session to offer, so it
   mints a deterministic one -- a UUIDv5 over a fixed name -- and therefore
   presents the same pair on every run, across restarts, forever. Generate a
   fresh UUID per run instead and the board refuses the second cycle with
   ``TASK_IDENTITY_MISMATCH``; mint a new one per restart and the board fills
   with identities nobody can clean up.

2. THE ARGUMENT SHAPE. ``task_post`` takes ``taskId``, ``sessionId`` and
   ``cwd`` -- the board's spelling, not this monorepo's -- and a package that
   spells one of them in snake_case is refused at validation, not at review.

3. THE STANDING TASK IS CONFIGURATION, NEVER DISCOVERY. Finding it by title
   search would make every cycle depend on a render grammar owned by another
   repository, for something that never changes. It is created once and
   exported beside the credentials, and its absence is refused rather than
   guessed at: an announcement posted to a guessed task is an announcement
   nobody is subscribed to, which reads exactly like the bridge working.

THE HTTP SEAM IS A PARAMETER, NOT A MODULE HOOK, for the reason
:mod:`platform_core.mcp_client` gives at length: every consuming package
already owns a ``_test_hooks`` module with its own poster, and a lib that kept
a hook of its own would give each caller two seams to hold in agreement.
"""

from __future__ import annotations

import uuid
from typing import Final, TypedDict

from platform_core.error_codes_tooling import BoardBridgeErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, McpPostProtocol, call_mcp_tool

#: The namespace every service session id is derived under.
#:
#: ``NAMESPACE_URL`` rather than a private namespace so the derivation can be
#: reproduced by anyone holding the name -- which is how an operator confirms
#: that the identity on the board is the one this code would mint, without
#: running it.
_SESSION_NAMESPACE: Final = uuid.NAMESPACE_URL

#: The board room every bridge posts into.
#:
#: A constant rather than a parameter because there is one room and a bridge
#: that posted into another would be invisible to the sessions watching this
#: one. When a second room exists, this becomes an argument and every caller is
#: forced to say which -- which is the point at which the choice is real.
BOARD_ROOM: Final = "main"


class BoardIdentity(TypedDict):
    """Who a service is on the board, on every run, forever.

    Immutable in practice: built once at import by :func:`service_identity`
    and never mutated, because a bridge whose identity changed mid-life would
    be a different writer to the board and could not post at all.

    Attributes:
        agent: The kebab-case label, ``<model-or-role>-<topic>-<MMDD>``.
        session_id: The deterministic UUIDv5 this service always presents.
        cwd: What the board records as the writer's working directory. A
            service has no directory a person could open, so it declares a
            ``service://`` URI instead of a path that does not exist.
    """

    agent: str
    session_id: str
    cwd: str


def mint_service_session_id(service: str) -> str:
    """Derive the session id a named service always presents.

    Deterministic by construction: the same ``service`` yields the same id on
    every machine, every run and every restart, which is what makes
    one-session-one-label survivable for something with no session.

    Args:
        service: The service's stable name, e.g. ``"corvis:hpc-wake:bridge"``.
            Changing it MINTS A NEW IDENTITY and orphans the old one on the
            board, so it is chosen once and never edited.

    Returns:
        The session id, as the canonical hyphenated string the board expects.
    """
    return str(uuid.uuid5(_SESSION_NAMESPACE, service))


def service_identity(*, agent: str, service: str, cwd: str) -> BoardIdentity:
    """Build the identity a bridge presents on every board write.

    ``cwd`` IS PASSED, NOT DERIVED. Deriving it from ``agent`` or ``service``
    would have silently rewritten what hpc-wake already records on the board
    when this function was lifted out of it -- a cosmetic change to a live
    audit trail, made invisibly, for the convenience of one fewer argument.
    Each bridge states its own.

    Args:
        agent: The kebab-case label the board will bind to this service.
        service: The stable name the session id is derived from. See
            :func:`mint_service_session_id` on why it is never edited.
        cwd: What the board records as the writer's location. A
            ``service://`` URI, because a service has no directory a person
            could open.

    Returns:
        The identity.
    """
    return BoardIdentity(
        agent=agent,
        session_id=mint_service_session_id(service),
        cwd=cwd,
    )


def require_task_id(value: str | None, *, variable: str) -> str:
    """Read the standing task's id, refusing an absent one.

    Args:
        value: What the environment held, or None when unset or blank.
        variable: The variable's name, for the refusal's message.

    Returns:
        The task id.

    Raises:
        AppError: ``TASK_ID_MISSING`` when ``value`` is None. Required rather
            than defaulted or discovered: an announcement posted to a guessed
            task is an announcement nobody is subscribed to, which reads
            exactly like the bridge working.
    """
    if value is None:
        raise AppError(
            code=BoardBridgeErrorCode.TASK_ID_MISSING,
            message=(
                f"{variable} is unset; it names the standing board task this "
                "bridge posts into. Create the task once and export its id "
                "beside the board credentials."
            ),
        )
    return value


def encode_board_post(identity: BoardIdentity, *, task_id: str, kind: str, body: str) -> JSONObject:
    """Render one ``task_post`` call's arguments.

    THE BOARD'S SPELLING, NOT THIS MONOREPO'S. ``taskId``, ``sessionId`` and
    ``cwd`` are camelCase or bare because that is what the tool validates
    against, and a package that spelled one of them ``session_id`` would be
    refused at the endpoint rather than at review. Held in one function so
    both bridges cannot spell them differently.

    Args:
        identity: Who is posting.
        task_id: The standing task the post lands in.
        kind: The board's post kind, e.g. ``"note"`` or ``"checkin"``.
        body: The post text.

    Returns:
        The arguments object, ready for
        :func:`platform_core.mcp_client.call_mcp_tool`.
    """
    return {
        "room": BOARD_ROOM,
        "taskId": task_id,
        "kind": kind,
        "body": body,
        "agent": identity["agent"],
        "sessionId": identity["session_id"],
        "cwd": identity["cwd"],
    }


def post_to_task(
    post: McpPostProtocol,
    credentials: McpCredentials,
    identity: BoardIdentity,
    *,
    task_id: str,
    kind: str,
    body: str,
) -> None:
    """Append one post to a standing task's thread.

    No claim is needed -- posting into a visible task's thread is open to any
    identity -- and no reply is parsed: the board echoes the appended line, and
    the only contract relied on here is that a non-error response means the
    post landed. Reading the board back is deliberately not done, because a
    bridge's position is its own local record and delivery to a waiting session
    is ``board-watch``'s job.

    Args:
        post: The caller's HTTP seam, from its own ``_test_hooks``.
        credentials: Endpoint and both board secrets.
        identity: Who is posting.
        task_id: The standing task.
        kind: The board's post kind.
        body: The post text.

    Raises:
        AppError: Through :func:`platform_core.mcp_client.call_mcp_tool` --
            ``HTTP_STATUS`` when the endpoint refused, which for a rotated key
            is the ordinary case, and ``RPC_ERROR`` when the board itself did,
            where an identity mismatch lands naming the established label.
            Nothing is caught: a bridge that swallowed a failed post would
            record the announcement as delivered and reproduce the silence it
            exists to remove.
    """
    call_mcp_tool(
        post,
        credentials,
        "task_post",
        encode_board_post(identity, task_id=task_id, kind=kind, body=body),
    )


__all__ = [
    "BOARD_ROOM",
    "BoardIdentity",
    "encode_board_post",
    "mint_service_session_id",
    "post_to_task",
    "require_task_id",
    "service_identity",
]
