"""Where a project's commits come from, and what an export of one needs
before its recipe can run.

THE WORKING TREE WAS THE DEFECT. Until MCPs board task fd5cabfa the only
tree a dispatch could carry was the hub's checkout of this monorepo, tarred
as it stood; a run of it was the verdict of whatever was on disk, uncommitted
edits included, and no closure could cite it as the check of a commit. A
project that declares a ``source`` can be exported instead: the runner
fetches one named sha from ``remote`` into a bare mirror on the hub and
stages ``git archive`` of it, so the tree on the node equals the commit by
construction and the queue row's sha is the citation.

WHY THE SOURCE IS DECLARED AND NOT DISCOVERED. ``git remote get-url`` on the
hub's checkout answers for this monorepo and for nothing else, and ``slime``
and the MCPs packages are projects of other repositories the hub may not
even have cloned. The registry names the remote, the directory inside the
repository the recipe runs in, and the install steps a clean export needs
before ``make check`` (an npm workspace installs at its root; a poetry
project installs inside its own recipe and declares none).

WHY THE INSTALL STEPS ARE ARGV LISTS UNDER A GRAMMAR. A closed vocabulary of
recipe names (``npm-ci``, ``npm-ci-playwright``, ...) would put every
repository's install convention into this package, and the next repository
would need a release here before it could be checked. An argv list is the
project's own business, the way its Makefile is, and it lives in a tracked,
reviewed file rather than on the queue, which still carries no command. Each
element is held to :data:`INSTALL_TOKEN` because the steps are rendered into
a script the node runs in its own shell (:mod:`fleet.core.dialect`): no
space, no quote, no metacharacter, so a token cannot compose an argument the
registry did not spell.
"""

from __future__ import annotations

import re
from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_str
from typing_extensions import TypedDict

#: A remote a mirror can fetch from: https, ssh (``git@host:path``) or ssh
#: URL. The grammar admits the characters those forms use and nothing a
#: shell reads, because the value becomes one argv element of ``git fetch``.
REMOTE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A(?:https://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._-]+)+|"
    r"git@[A-Za-z0-9.-]+:[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+|"
    r"ssh://[A-Za-z0-9.@-]+(?:/[A-Za-z0-9._-]+)+)\Z"
)

#: A directory inside the repository: empty for the root, else slash-joined
#: segments that are neither ``.`` nor ``..`` and carry nothing a shell
#: reads. The same alphabet as the queue's project grammar.
PATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A(?:[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*)?\Z"
)

#: One argv element of an install step: a word or a ``-flag``. The tokens are
#: joined with spaces, unquoted, into both dialects' scripts, so the leading
#: character may not be one either shell reads (``@`` splats in PowerShell,
#: ``$`` expands in both) and no position may carry whitespace or a quote.
INSTALL_TOKEN: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9-][A-Za-z0-9._/@:=-]*\Z")


class ProjectSource(TypedDict):
    """Where a project's commits are fetched from and how an export is readied.

    Attributes:
        remote: The git remote the runner fetches a sha from.
        path: The directory inside the repository the recipe runs in;
            ``""`` for a project that is the repository.
        install: The commands run at the EXPORT ROOT, in order, before
            ``make check`` runs in ``path``; each one an argv. Empty for a
            project whose recipe installs its own dependencies.
    """

    remote: str
    path: str
    install: tuple[tuple[str, ...], ...]


def decode_remote(value: str, *, field: str) -> str:
    """Validate a remote against :data:`REMOTE_PATTERN`.

    Args:
        value: The declared remote.
        field: The key it came from, for the message.

    Returns:
        The remote.

    Raises:
        JSONTypeError: If it is not an https, ``git@`` or ``ssh://`` remote
            in the grammar.
    """
    if REMOTE_PATTERN.fullmatch(value) is None:
        raise JSONTypeError(
            f"{field} must be an https://, git@host:path or ssh:// git remote with no "
            f"whitespace or shell metacharacters, got {value!r}"
        )
    return value


def decode_path(value: str, *, field: str) -> str:
    """Validate a repository-relative directory against :data:`PATH_PATTERN`.

    Args:
        value: The declared path.
        field: The key it came from, for the message.

    Returns:
        The path, ``""`` for the repository root.

    Raises:
        JSONTypeError: If a segment is ``.``, ``..``, empty, or carries a
            character outside the grammar.
    """
    if PATH_PATTERN.fullmatch(value) is None:
        raise JSONTypeError(
            f"{field} must be '' for the repository root or slash-joined segments of "
            f"[A-Za-z0-9._-] each starting alphanumeric, got {value!r}"
        )
    return value


def decode_install(value: JSONValue, *, field: str) -> tuple[tuple[str, ...], ...]:
    """Decode the install steps.

    Args:
        value: The value under ``field``.
        field: The key it came from, for the message.

    Returns:
        The steps, each a non-empty argv of tokens in :data:`INSTALL_TOKEN`.

    Raises:
        JSONTypeError: If the key is absent, a step is not a list, a step is
            empty, or a token is not a string in the grammar.
    """
    if value is None:
        raise JSONTypeError(
            f"{field} is required: [] for a project whose recipe installs its own "
            "dependencies, else the argv lists run at the export root before make check"
        )
    if not isinstance(value, list):
        raise JSONTypeError(f"{field} must be a list of argv lists, got {type(value).__name__}")
    steps: list[tuple[str, ...]] = []
    for index, step in enumerate(value):
        if not isinstance(step, list):
            raise JSONTypeError(
                f"{field}[{index}] must be a list of argv tokens, got {type(step).__name__}"
            )
        if not step:
            raise JSONTypeError(f"{field}[{index}] is empty; a step names its executable")
        tokens: list[str] = []
        for position, token in enumerate(step):
            if not isinstance(token, str) or INSTALL_TOKEN.fullmatch(token) is None:
                raise JSONTypeError(
                    f"{field}[{index}][{position}] must be a string of [A-Za-z0-9._/@:=-] "
                    f"starting alphanumeric or with a dash, got {token!r}; the steps are "
                    "rendered into the "
                    "node's shell, so no token may carry whitespace, a quote or a metacharacter"
                )
            tokens.append(token)
        steps.append(tuple(tokens))
    return tuple(steps)


def decode_project_source(value: JSONValue, *, field: str) -> ProjectSource | None:
    """Decode a project's ``source``, which is required and may be null.

    Args:
        value: The value under ``field``: an object, or null for a project
            with no remote (dispatched only from a working tree by
            ``fleet-run``, refused by the export runner by name). The
            caller has already established the key is PRESENT, so ``None``
            here is the declared null and never an absent key.
        field: The key it came from, for the message.

    Returns:
        The source, or None when declared null.

    Raises:
        JSONTypeError: If the value is neither an object nor null, or a
            field is missing or outside its grammar.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JSONTypeError(f"{field} must be an object or null, got {type(value).__name__}")
    return ProjectSource(
        remote=decode_remote(require_str(value, "remote"), field=f"{field}.remote"),
        path=decode_path(require_str(value, "path"), field=f"{field}.path"),
        install=decode_install(value.get("install"), field=f"{field}.install"),
    )


def encode_project_source(source: ProjectSource | None) -> JSONValue:
    """Encode a source for the workspace file.

    Args:
        source: The source, or None.

    Returns:
        The JSON object, or null.
    """
    if source is None:
        return None
    steps: list[JSONValue] = []
    for step in source["install"]:
        tokens: list[JSONValue] = list(step)
        steps.append(tokens)
    encoded: JSONObject = {
        "remote": source["remote"],
        "path": source["path"],
        "install": steps,
    }
    return encoded


__all__ = [
    "INSTALL_TOKEN",
    "PATH_PATTERN",
    "REMOTE_PATTERN",
    "ProjectSource",
    "decode_install",
    "decode_path",
    "decode_project_source",
    "decode_remote",
    "encode_project_source",
]
