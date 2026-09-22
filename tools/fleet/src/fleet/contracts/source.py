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

WHY A PROJECT MAY DECLARE COMPANIONS, AND WHY THE NODE CANNOT FETCH THEM
ITSELF. A repository whose check reads a second repository -- ``slime``
lints its lifted code against the committed ``MCPs`` workspace beside it --
has nothing to read on a node, which is handed that project's commit and
nothing else. The node cannot go and get it: both repositories are private
and a node holds no git credential, by the design this module's neighbour
states. Measured on sedona 2026-09-22, ``git ls-remote`` of the workspace
remote there exits 128 with "could not read Username for
'https://github.com'". So a companion travels from the hub with the work,
through the same mirror, the same archive and the same digest-verified
transport as the project's own commit, and lands BESIDE the export at
``<stage_root>/<directory>`` -- which is ``../<directory>`` from the export
root, the layout a workstation already has, so the recipe needs no
knowledge of where it is running.

WHY A COMPANION NAMES A REF AND THE PROJECT NAMES A SHA. The queue row is
the citation for the project's commit, and a citation must resolve to the
same tree tomorrow. A companion is not being cited: the check it serves
asks whether the lifted code still matches the workspace AS IT STANDS, so
the tip of a declared ref is the question, and pinning a sha here would
make a drifted workspace invisible instead of failing. The resolved sha is
recorded on the feed, so a verdict can still be read against the workspace
it was measured with.

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

#: The ref a companion's tip is exported from: a branch name, or its full
#: ``refs/heads/`` spelling. Slash-joined segments that each begin
#: alphanumeric, which admits neither ``..`` nor a leading dash nor anything
#: a shell reads, because the value becomes one argv element of ``git fetch``.
COMPANION_REF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*\Z"
)

#: The directory a companion takes beside the export: ONE segment, never a
#: path. It is joined to the node's stage root, and a segment that could
#: spell ``..`` or an absolute path would let a registry line write outside
#: the staging area the runner owns.
COMPANION_DIRECTORY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ProjectCompanion(TypedDict):
    """Another repository exported beside a project, for a check that reads it.

    Attributes:
        remote: The git remote the runner mirrors and fetches the ref from.
        ref: The ref whose TIP is exported, ``main`` or ``refs/heads/main``;
            never a sha, for the reason the module docstring gives.
        directory: The single directory name it lands in beside the export,
            at ``<stage_root>/<directory>``, which the recipe reaches as
            ``../<directory>`` from its own root.
    """

    remote: str
    ref: str
    directory: str


class ProjectSource(TypedDict):
    """Where a project's commits are fetched from and how an export is readied.

    Attributes:
        remote: The git remote the runner fetches a sha from.
        path: The directory inside the repository the recipe runs in;
            ``""`` for a project that is the repository.
        install: The commands run at the EXPORT ROOT, in order, before
            ``make check`` runs in ``path``; each one an argv. Empty for a
            project whose recipe installs its own dependencies.
        companions: The other repositories exported beside this one before
            the install steps run. Empty for a project whose check reads
            only its own tree, which is every project but ``slime``.
    """

    remote: str
    path: str
    install: tuple[tuple[str, ...], ...]
    companions: tuple[ProjectCompanion, ...]


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


def decode_companion_ref(value: str, *, field: str) -> str:
    """Validate a companion's ref against :data:`COMPANION_REF_PATTERN`.

    Args:
        value: The declared ref.
        field: The key it came from, for the message.

    Returns:
        The ref.

    Raises:
        JSONTypeError: If it is empty, a segment does not begin
            alphanumeric, or it carries a character outside the grammar --
            which a forty-hex sha does not, and the message says why that is
            still the wrong thing to write here.
    """
    if COMPANION_REF_PATTERN.fullmatch(value) is None:
        raise JSONTypeError(
            f"{field} must be a ref whose slash-joined segments each start alphanumeric, "
            f"such as 'main' or 'refs/heads/main', got {value!r}"
        )
    return value


def decode_companion_directory(value: str, *, field: str) -> str:
    """Validate a companion's directory against
    :data:`COMPANION_DIRECTORY_PATTERN`.

    Args:
        value: The declared directory name.
        field: The key it came from, for the message.

    Returns:
        The directory name.

    Raises:
        JSONTypeError: If it is empty, carries a slash, or begins with a
            character other than an alphanumeric.
    """
    if COMPANION_DIRECTORY_PATTERN.fullmatch(value) is None:
        raise JSONTypeError(
            f"{field} must be ONE directory name of [A-Za-z0-9._-] starting alphanumeric, "
            f"joined to the node's stage root and never a path, got {value!r}"
        )
    return value


def decode_companion(value: JSONValue, *, field: str) -> ProjectCompanion:
    """Decode one companion declaration.

    Args:
        value: The array element.
        field: The key it came from, for the message.

    Returns:
        The companion.

    Raises:
        JSONTypeError: If the element is not an object, a field is missing,
            or a field is outside its grammar.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{field} must be an object, got {type(value).__name__}")
    return ProjectCompanion(
        remote=decode_remote(require_str(value, "remote"), field=f"{field}.remote"),
        ref=decode_companion_ref(require_str(value, "ref"), field=f"{field}.ref"),
        directory=decode_companion_directory(
            require_str(value, "directory"), field=f"{field}.directory"
        ),
    )


def decode_companions(value: JSONValue, *, field: str) -> tuple[ProjectCompanion, ...]:
    """Decode the companions a project's export carries.

    Args:
        value: The value under ``field``.
        field: The key it came from, for the message.

    Returns:
        The companions in declared order.

    Raises:
        JSONTypeError: If the key is absent, the value is not a list, an
            element is outside the grammar, or two companions name one
            directory -- which would have the second stage over the first
            and leave the recipe reading whichever finished last.
    """
    if value is None:
        raise JSONTypeError(
            f"{field} is required: [] for a project whose check reads only its own tree, "
            "else the repositories exported beside it"
        )
    if not isinstance(value, list):
        raise JSONTypeError(f"{field} must be a list of companions, got {type(value).__name__}")
    companions: list[ProjectCompanion] = []
    for index, element in enumerate(value):
        companion = decode_companion(element, field=f"{field}[{index}]")
        for taken in companions:
            if taken["directory"] == companion["directory"]:
                raise JSONTypeError(
                    f"{field}[{index}].directory repeats {companion['directory']!r}; two "
                    "companions in one directory would stage over each other"
                )
        companions.append(companion)
    return tuple(companions)


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
        companions=decode_companions(value.get("companions"), field=f"{field}.companions"),
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
    companions: list[JSONValue] = []
    for companion in source["companions"]:
        declared: JSONObject = {
            "remote": companion["remote"],
            "ref": companion["ref"],
            "directory": companion["directory"],
        }
        companions.append(declared)
    encoded: JSONObject = {
        "remote": source["remote"],
        "path": source["path"],
        "install": steps,
        "companions": companions,
    }
    return encoded


__all__ = [
    "COMPANION_DIRECTORY_PATTERN",
    "COMPANION_REF_PATTERN",
    "INSTALL_TOKEN",
    "PATH_PATTERN",
    "REMOTE_PATTERN",
    "ProjectCompanion",
    "ProjectSource",
    "decode_companion",
    "decode_companion_directory",
    "decode_companion_ref",
    "decode_companions",
    "decode_install",
    "decode_path",
    "decode_project_source",
    "decode_remote",
    "encode_project_source",
]
