"""Reading the workspace document every command starts from.

Shared by all seven entry points, which is the point: they cannot disagree
about where the cluster is or which ledger records it, because they all read
the same file through the same function.

The document is located relative to the process, and the paths inside it
relative to the document. That is what lets a workspace live next to the runs
it describes and be used from any working directory. The one exception is the
ledger, which resolves against the document's directory as it sits in the
repository's main checkout, so every worktree reads and appends the same
record (:mod:`hpc3.core.checkout`, board task 747c391e).
"""

from __future__ import annotations

import pathlib

from platform_core import cli_args
from platform_core.config import _optional_env_str
from platform_core.json_utils import JSONValue, load_json_str
from platform_core.mcp_client import McpCredentials
from platform_core.session_label import (
    LABEL_VARIABLE,
    SESSION_ID_VARIABLE,
    STACK_ENV_PATH,
    resolve_label,
    stack_credentials,
)
from platform_core.stack_endpoints import TASKBOARD_SERVICE, declared_url

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.contracts.workspace import (
    Workspace,
    WorkspaceConnection,
    decode_workspace,
    decode_workspace_connection,
)
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.checkout import shared_directory

CONFIG_FLAG = "--config"
"""The one flag every command shares."""


def _read_document(parsed: dict[str, str]) -> tuple[JSONValue, pathlib.Path]:
    """Read the workspace document named on the command line.

    Args:
        parsed: Flags already read from the command line.

    Returns:
        The loaded JSON value and the directory it was read from, which the
        paths inside it resolve against.

    Raises:
        ValueError: If ``--config`` was not given.
        JSONTypeError: If the file is not valid JSON.
    """
    path = pathlib.Path(cli_args.require_flag(parsed, CONFIG_FLAG)).resolve()
    return load_json_str(core_hooks.read_bytes(path).decode("utf-8")), path.parent


def load_workspace_connection(parsed: dict[str, str]) -> WorkspaceConnection:
    """Read only where the cluster is, leaving the project registry unread.

    For the onboarding path, which needs the host while the project it is
    onboarding cannot yet be registered -- registration requires an image
    digest, and producing that digest is what the onboarding path does. See
    :class:`~hpc3.contracts.workspace.WorkspaceConnection`.

    Args:
        parsed: Flags already read from the command line.

    Returns:
        The validated connection.

    Raises:
        ValueError: If ``--config`` was not given, or the document declares a
            root that is not an absolute POSIX path.
        JSONTypeError: If the document's connection fields are invalid.
        AppError: With ``CLUSTER_UNKNOWN`` if the named cluster has not been
            measured, or ``LEDGER_CHECKOUT_UNREADABLE`` if the worktree's
            ``.git`` pointer cannot be followed to a main checkout.
    """
    value, config_dir = _read_document(parsed)
    return decode_workspace_connection(value, ledger_dir=shared_directory(config_dir))


def load_workspace(parsed: dict[str, str]) -> Workspace:
    """Read and validate the workspace named on the command line.

    Args:
        parsed: Flags already read from the command line.

    Returns:
        The validated workspace, with its ledger path resolved against the
        document's directory as it sits in the repository's main checkout,
        so every worktree reads and appends one ledger, and each project's
        ``repo`` against the document's own directory.

    Raises:
        ValueError: If ``--config`` was not given, or the document declares a
            root that is not an absolute POSIX path.
        JSONTypeError: If the document is not a valid workspace.
        AppError: If a project declares a GPU this cluster does not carry, or
            with ``LEDGER_CHECKOUT_UNREADABLE`` if the worktree's ``.git``
            pointer cannot be followed to a main checkout.
    """
    value, config_dir = _read_document(parsed)
    return decode_workspace(value, config_dir=config_dir, ledger_dir=shared_directory(config_dir))


def _stack_file_text(path: pathlib.Path) -> str:
    """Read one of the stack's files (its ``.env`` or its endpoint
    declaration) through the core's byte seam.

    Args:
        path: The file :mod:`platform_core.session_label` asks for.

    Returns:
        Its text.
    """
    return core_hooks.read_bytes(path).decode("utf-8")


def taskboard_credentials() -> McpCredentials:
    """Read the taskboard's address and secrets from the stack's own files.

    For a command that posts its own report under a service identity; the
    same two files :func:`submitter_label` reads, through the same seam.

    Returns:
        The credentials, bound to the declared taskboard url.

    Raises:
        AppError: ``STACK_ENDPOINT_UNDECLARED`` or ``CREDENTIALS_MISSING``
            when either file does not carry its part.
    """
    url = declared_url(_stack_file_text, TASKBOARD_SERVICE)
    return stack_credentials(_stack_file_text(STACK_ENV_PATH), url)


def submitter_label() -> str:
    """Resolve the submitting session's agent-board label against the board.

    Shared by every submitting entry point for the same reason
    :func:`load_workspace` is: the ledger's ``submitter`` field must mean
    one thing, not five slightly different readings of the environment.

    THE ENVIRONMENT IS NOT BELIEVED ON ITS OWN (MCPs board task 3843d29f).
    Inside a Claude Code session the harness exports the session's id, and
    :func:`platform_core.session_label.resolve_label` asks ``task_whereis``
    which label the board bound to it: an unset ``BOARD_AGENT_LABEL`` is
    filled from the binding, and one naming a different label is refused
    before anything is submitted, so a bridge never tags the wrong session
    for a job's terminal state. Outside a session the export is taken as
    given.

    Returns:
        The label the board binds the session to, else the label the shell
        exported, else ``""`` -- the ledger's positive "declared none". The
        spellings of not-declaring collapse deliberately: an empty export
        names nobody to tag, exactly like no export. Both variables are read
        through :func:`platform_core.config._optional_env_str` because that
        is the workspace's one sanctioned environment reader; the guard
        bans a second one.

    Raises:
        AppError: ``SESSION_LABEL_MISMATCH`` when the shell's label is not
            the one the board bound to the session; ``SESSION_ID_MALFORMED``,
            ``STACK_ENDPOINT_UNDECLARED`` or
            ``SESSION_LABEL_CREDENTIALS_MISSING`` when the board cannot be
            asked, and the :class:`McpClientErrorCode` failures when it does
            not answer.
    """
    return resolve_label(
        session_id=_optional_env_str(SESSION_ID_VARIABLE),
        exported=_optional_env_str(LABEL_VARIABLE),
        read_text=_stack_file_text,
        post=cli_hooks.http_post,
    )


__all__ = [
    "CONFIG_FLAG",
    "load_workspace",
    "load_workspace_connection",
    "submitter_label",
    "taskboard_credentials",
]
