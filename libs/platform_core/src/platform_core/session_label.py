"""The acting session's board label, asked of the ledger rather than the shell.

WHY THE SHELL IS NOT BELIEVED. Two enrolling tools in this monorepo record
who asked for work so a bridge can @mention them later: ``hpc3`` writes
``BOARD_AGENT_LABEL`` into the Slurm ledger at submit, and ``ci-wake``'s
``pre-push`` enrolment writes it into the push record. Both took the
variable on trust, and nothing checked it against the label the board had
bound to the session. Measured 2026-09-12 (MCPs board task 3843d29f): one
session had posted to the board as ``opus-rebuild-deadlock-0910`` for
hours, then exported a label reconstructed from the work it was doing,
``opus-dashboard-0911``, and the append-only fleet journal recorded the
wrong name permanently. The session's own account is that only a refusal
would have caught it; a note in a document had not.

WHAT IS ASKED, AND OF WHOM. The harness exports the acting session's id
into every shell it spawns as ``CLAUDE_CODE_SESSION_ID`` (measured
2026-09-16, MCPs board task d207df8a). The board binds one label to one
session id on that session's first write and never releases it
(``assertSessionLabel``, mig 415), and ``task_whereis(session=<id>)`` on
the taskboard's loopback surface answers with that binding. It is called
through :func:`platform_core.mcp_client.call_mcp_tool` with the stack's
``MCP_INTERNAL_KEY`` and ``OPERATOR_TENANT_ID`` read from the MCPs
repository's ``.env``, the same file and the same two keys the
session-ledger hook reads for its checkin, so the two session-side readers
of the board agree about where the credentials are. A caller passes its
own file reader and HTTP poster, the way every consumer of
:mod:`platform_core.mcp_client` passes its poster, so the seams stay in
the package that already owns them.

THE RULE IS THE BOARD'S, RESTATED. A session the board has bound is that
label, whatever the shell says: an unset ``BOARD_AGENT_LABEL`` is filled
from the binding, and a different one is refused naming both, the shape of
``TASK_IDENTITY_MISMATCH``. A session the board has not bound keeps the
label it exported, which is what the board would bind on its first write.
A shell with no session id, a human's terminal, has nothing to check
against and keeps its export, or its absence: an unaddressed enrolment is
a first-class outcome for it, as it always was.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import Final, TypedDict

from platform_core.error_codes_tooling import SessionLabelErrorCode
from platform_core.errors import AppError
from platform_core.mcp_client import McpCredentials, McpPostProtocol, call_mcp_tool

#: Environment variable the harness exports carrying the acting session's id.
SESSION_ID_VARIABLE: Final = "CLAUDE_CODE_SESSION_ID"

#: Environment variable a shell exports naming the session's board label.
LABEL_VARIABLE: Final = "BOARD_AGENT_LABEL"

#: The compose stack's environment file, where the taskboard's key and the
#: operator tenant live. The same path the session-ledger hook reads.
STACK_ENV_PATH: Final = pathlib.Path.home() / "PROJECTS" / "MCPs" / ".env"

#: The two keys read from it.
API_KEY_NAME: Final = "MCP_INTERNAL_KEY"
TENANT_ID_NAME: Final = "OPERATOR_TENANT_ID"

#: Where taskboard-mcp is published on the host.
TASKBOARD_URL: Final = "http://127.0.0.1:8033/mcp"

#: The tool that answers "which label is bound to this session id".
WHEREIS_TOOL: Final = "task_whereis"

#: A session id as the harness writes it: lowercase hex, hyphenated.
_SESSION_ID: Final = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

#: The line ``task_whereis`` prints for a session the board has written
#: under a label (``taskboard-mcp/src/session-encode.ts``). A session the
#: board has never seen prints ``board     never written to this board``
#: instead, and a session nothing knows prints a ``WHEREIS: nothing ...``
#: miss; neither matches, and both mean "no binding".
_BOARD_LINE: Final = re.compile(r"^board\s+([a-z0-9][a-z0-9-]{2,63}) — ", re.MULTILINE)


class SessionIdentity(TypedDict):
    """What the shell said and what the board says, side by side.

    Attributes:
        session_id: The harness's session id; empty outside a session.
        exported: ``BOARD_AGENT_LABEL`` as exported; empty when unset.
        bound: The board's binding for the session; empty when there is
            no session or the board has never seen it.
    """

    session_id: str
    exported: str
    bound: str


def require_session_id(value: str) -> str:
    """Validate a session id the harness exported.

    Args:
        value: ``CLAUDE_CODE_SESSION_ID`` as read, non-empty.

    Returns:
        The value unchanged.

    Raises:
        AppError: ``SESSION_ID_MALFORMED`` when it is not a session id as
            the harness writes them. The harness writes these, so a
            malformed one is a changed harness, and asking the board about
            it would answer "unbound" for a session that may well be bound.
    """
    if _SESSION_ID.match(value) is None:
        raise AppError(
            code=SessionLabelErrorCode.SESSION_ID_MALFORMED,
            message=(
                f"{SESSION_ID_VARIABLE}={value!r} is not a session id as the harness "
                "writes them, so the board cannot be asked who this session is"
            ),
        )
    return value


def stack_credentials(env_text: str) -> McpCredentials:
    """Read the taskboard's credentials out of the stack's ``.env`` text.

    Args:
        env_text: The whole file. ``NAME=value`` lines, values optionally
            quoted; everything else is ignored.

    Returns:
        The credentials, bound to :data:`TASKBOARD_URL`.

    Raises:
        AppError: ``CREDENTIALS_MISSING`` when either key is absent or
            empty, naming the key and the file.
    """
    values: dict[str, str] = {}
    for line in env_text.splitlines():
        name, separator, value = line.partition("=")
        if separator == "":
            continue
        values[name.strip()] = value.strip().strip('"').strip("'")
    api_key = values.get(API_KEY_NAME, "")
    tenant_id = values.get(TENANT_ID_NAME, "")
    for name, value in ((API_KEY_NAME, api_key), (TENANT_ID_NAME, tenant_id)):
        if value == "":
            raise AppError(
                code=SessionLabelErrorCode.CREDENTIALS_MISSING,
                message=(
                    f"{STACK_ENV_PATH} carries no {name}, so the board cannot be asked "
                    "who this session is"
                ),
            )
    return McpCredentials(url=TASKBOARD_URL, api_key=api_key, tenant_id=tenant_id)


def bound_label(whereis_text: str) -> str:
    """The label ``task_whereis`` reports as bound to the session.

    Args:
        whereis_text: The tool's rendered text.

    Returns:
        The label, or the empty string when the board has never written
        under this session: the ``board`` line says so, or the whole
        answer is a miss.
    """
    match = _BOARD_LINE.search(whereis_text)
    return "" if match is None else match.group(1)


def read_identity(
    *,
    session_id: str | None,
    exported: str | None,
    read_text: Callable[[pathlib.Path], str],
    post: McpPostProtocol,
) -> SessionIdentity:
    """Read the shell's account and, inside a session, the board's.

    Args:
        session_id: ``CLAUDE_CODE_SESSION_ID`` as the sanctioned reader
            returned it: None when unset or blank.
        exported: ``BOARD_AGENT_LABEL`` the same way.
        read_text: The caller's file seam, for the stack's ``.env``.
        post: The caller's HTTP seam.

    Returns:
        Both accounts. The board is asked only when there is a session id
        to ask about, so a human's terminal never reads the credentials.

    Raises:
        AppError: ``SESSION_ID_MALFORMED`` or ``CREDENTIALS_MISSING`` from
            the readers, and the :class:`McpClientErrorCode` failures of
            the call: the board that cannot be asked is raised, never read
            as unbound.
    """
    declared = "" if exported is None else exported
    if session_id is None:
        return SessionIdentity(session_id="", exported=declared, bound="")
    session = require_session_id(session_id)
    text = call_mcp_tool(
        post, stack_credentials(read_text(STACK_ENV_PATH)), WHEREIS_TOOL, {"session": session}
    )
    return SessionIdentity(session_id=session, exported=declared, bound=bound_label(text))


def disagrees(identity: SessionIdentity) -> bool:
    """Whether the shell exports a label other than the board's binding.

    Args:
        identity: The two accounts.

    Returns:
        True only when both name a label and they differ. An unset export
        beside a binding is filled, not a disagreement; an export beside
        no binding is what the board would bind.
    """
    bound = identity["bound"]
    exported = identity["exported"]
    return bound != "" and exported != "" and exported != bound


def mismatch(identity: SessionIdentity) -> AppError[SessionLabelErrorCode]:
    """The refusal for a disagreement, naming both labels.

    Both are named, the shape of ``TASK_IDENTITY_MISMATCH``, because the
    refusal is the one moment the wrong name is visible before a bridge
    posts under it.

    Args:
        identity: The two accounts, which :func:`disagrees`.

    Returns:
        The error to raise.
    """
    return AppError(
        code=SessionLabelErrorCode.LABEL_MISMATCH,
        message=(
            f"session {identity['session_id']} writes to the board as "
            f"'{identity['bound']}' and cannot act as '{identity['exported']}': "
            f"{LABEL_VARIABLE} names a label other than the one the board bound to "
            "this session. Unset it; the enrolment is attributed from the board's "
            "binding (MCPs board task 3843d29f)."
        ),
    )


def resolved_label(identity: SessionIdentity) -> str:
    """The label an enrolment records for accounts that do not disagree.

    Args:
        identity: The two accounts.

    Returns:
        The board's binding when there is one, else the shell's export,
        else the empty string: the enrolment's positive "declared none".
    """
    return identity["bound"] if identity["bound"] != "" else identity["exported"]


def resolve_label(
    *,
    session_id: str | None,
    exported: str | None,
    read_text: Callable[[pathlib.Path], str],
    post: McpPostProtocol,
) -> str:
    """The acting session's label, resolved against the board, or a refusal.

    The one call an enrolling site makes; the pieces above are exposed so
    the rule is testable without a board.

    Args:
        session_id: ``CLAUDE_CODE_SESSION_ID`` as read, None when unset.
        exported: ``BOARD_AGENT_LABEL`` as read, None when unset.
        read_text: The caller's file seam.
        post: The caller's HTTP seam.

    Returns:
        See :func:`resolved_label`.

    Raises:
        AppError: ``LABEL_MISMATCH`` from :func:`mismatch`, and everything
            :func:`read_identity` raises.
    """
    identity = read_identity(
        session_id=session_id, exported=exported, read_text=read_text, post=post
    )
    if disagrees(identity):
        raise mismatch(identity)
    return resolved_label(identity)


__all__ = [
    "API_KEY_NAME",
    "LABEL_VARIABLE",
    "SESSION_ID_VARIABLE",
    "STACK_ENV_PATH",
    "TASKBOARD_URL",
    "TENANT_ID_NAME",
    "WHEREIS_TOOL",
    "SessionIdentity",
    "bound_label",
    "disagrees",
    "mismatch",
    "read_identity",
    "require_session_id",
    "resolve_label",
    "resolved_label",
    "stack_credentials",
]
