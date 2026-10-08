"""What a test names when it takes an effect through a failure, by kind.

The third piece of ``effect-seam-twin``: a seam's real test must take it
through a failure, and the failure must be one of ITS kind. Measured in MCPs
on 2026-10-08 (board task c96e8791): a maketools case handed the whole real
bundle over and named ``PermissionError`` for an overridden ``remove_tree``,
which credited a failure to nine process seams it never failed. So each
marker counts for the kinds of effect it is a failure of, a test's failure
is the set of kinds its markers count for, and a seam passes when one of its
real tests carries its kind (an ssh seam is satisfied by a process or a
network failure). The table is MCPs', and a cross-language test there holds
the Python and TypeScript sets identical.

The markers are read in the test and its same-file helpers:

* a NAME (a bare name, an attribute's last part, or a string key) from
  :data:`FAILURE_NAME_KINDS`, e.g. ``subprocess.TimeoutExpired``,
  ``pytest.raises(OperationalError)``, ``errno.ENOENT``;
* a read of ``timed_out`` or ``killed`` (attribute or key), and a call of
  ``kill``, ``terminate`` or ``killpg``: process;
* an exit code (``exit_code``, ``returncode``, ``exitcode``, ``code``)
  compared equal to a nonzero literal, unequal to 0 or above 0: process;
* a ``status`` or ``status_code`` compared to a literal of 400 or more:
  network;
* a string literal that exits nonzero (``sys.exit(3)``, ``process.exit(1)``,
  ``exit 2``), the child script a real process test runs: process.

A bare ``pytest.raises`` is not one: ``raises(ValueError)`` on bad input is
not a timeout, an exit, a refusal or a kill.
"""

from __future__ import annotations

import ast
import re

PROCESS = "process"
NETWORK = "network"
SSH = "ssh"
FILE_SWAP = "file swap"
SERVICE = "service"

#: Each failure name and the kinds of effect it is a failure of.
FAILURE_NAME_KINDS: dict[str, frozenset[str]] = {
    "TimeoutExpired": frozenset({PROCESS}),
    "CalledProcessError": frozenset({PROCESS}),
    "SIGKILL": frozenset({PROCESS}),
    "SIGTERM": frozenset({PROCESS}),
    "TimeoutError": frozenset({PROCESS, NETWORK, SERVICE}),
    "URLError": frozenset({NETWORK}),
    "HTTPError": frozenset({NETWORK}),
    "ConnectionResetError": frozenset({NETWORK}),
    "ConnectError": frozenset({NETWORK}),
    "ConnectTimeout": frozenset({NETWORK}),
    "ReadTimeout": frozenset({NETWORK}),
    "TimeoutException": frozenset({NETWORK}),
    "ClientConnectorError": frozenset({NETWORK}),
    "ServerDisconnectedError": frozenset({NETWORK}),
    "ECONNRESET": frozenset({NETWORK}),
    "AbortError": frozenset({NETWORK}),
    "ConnectionRefusedError": frozenset({NETWORK, SERVICE}),
    "ConnectionError": frozenset({NETWORK, SERVICE}),
    "ETIMEDOUT": frozenset({NETWORK, SERVICE}),
    "ECONNREFUSED": frozenset({NETWORK, SERVICE}),
    "OperationalError": frozenset({SERVICE}),
    "PermissionError": frozenset({FILE_SWAP}),
    "FileNotFoundError": frozenset({FILE_SWAP}),
    "FileExistsError": frozenset({FILE_SWAP}),
    "IsADirectoryError": frozenset({FILE_SWAP}),
    "NotADirectoryError": frozenset({FILE_SWAP}),
    "ENOENT": frozenset({FILE_SWAP}),
    "EEXIST": frozenset({FILE_SWAP}),
    "EISDIR": frozenset({FILE_SWAP}),
    "ENOTDIR": frozenset({FILE_SWAP}),
    "EPERM": frozenset({FILE_SWAP}),
    "EACCES": frozenset({FILE_SWAP}),
    "EBUSY": frozenset({FILE_SWAP}),
}

#: Which failure kinds satisfy a seam of each effect kind.
SATISFIED_BY: dict[str, frozenset[str]] = {
    PROCESS: frozenset({PROCESS}),
    NETWORK: frozenset({NETWORK}),
    SSH: frozenset({PROCESS, NETWORK}),
    FILE_SWAP: frozenset({FILE_SWAP}),
    SERVICE: frozenset({SERVICE}),
}

KILL_CALLS = frozenset({"kill", "terminate", "killpg"})
FAILURE_ATTRIBUTES = frozenset({"timed_out", "killed"})
EXIT_NAMES = frozenset({"exit_code", "returncode", "exitcode", "code"})
STATUS_NAMES = frozenset({"status", "status_code"})
NONZERO_EXIT_TEXT = re.compile(r"(?:sys\.exit|process\.exit)\(\s*[1-9]\d*\s*\)|\bexit\s+[1-9]\d*\b")


def _int_literal(node: ast.expr) -> int | None:
    """Read an integer literal, refusing a bool.

    Args:
        node: Expression.

    Returns:
        The integer, or None when the node is not one.
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def terminal_name(node: ast.expr) -> str | None:
    """Name a ``Name``, the last part of an ``Attribute``, or a string key.

    A string-keyed subscript reads the same field an attribute does: the
    command results here are TypedDicts, so a test states a timeout as
    ``result["timed_out"]`` and an exit as ``result["returncode"] == 3``
    (tools/fleet's ``test_core_io.py``), never ``result.timed_out``.

    Args:
        node: Expression.

    Returns:
        The identifier or key, or None for anything else.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def _compare_kind(node: ast.Compare) -> str | None:
    """Name the kind of failure a comparison asserts, if it asserts one.

    Args:
        node: One comparison (only its first operator is read; chained
            comparisons are not how a test states an exit code).

    Returns:
        Process for a nonzero exit code, network for a status of 400 or
        more, None otherwise.
    """
    left, op, right = node.left, node.ops[0], node.comparators[0]
    names = (terminal_name(left), terminal_name(right))
    values = (_int_literal(right), _int_literal(left))
    if any(name in STATUS_NAMES for name in names):
        failed = any(value is not None and value >= 400 for value in values)
        return NETWORK if failed else None
    if names[0] in EXIT_NAMES:
        value, flipped = values[0], False
    elif names[1] in EXIT_NAMES:
        value, flipped = values[1], True
    else:
        return None
    if value is None:
        return None
    if isinstance(op, ast.Eq):
        failed = value != 0
    elif isinstance(op, ast.NotEq):
        failed = value == 0
    else:
        above = ast.Lt if flipped else ast.Gt
        failed = isinstance(op, above) and value == 0
    return PROCESS if failed else None


def _node_kinds(child: ast.AST) -> frozenset[str]:
    """Name the failure kinds one node is a marker of.

    Args:
        child: Any node of a test or helper.

    Returns:
        The kinds, empty for a node that marks no failure.
    """
    if isinstance(child, (ast.Name, ast.Attribute, ast.Subscript)):
        name = terminal_name(child) or ""
        if not isinstance(child, ast.Name) and name in FAILURE_ATTRIBUTES:
            return frozenset({PROCESS})
        return FAILURE_NAME_KINDS.get(name, frozenset())
    if isinstance(child, ast.Call):
        killed = terminal_name(child.func) in KILL_CALLS
        return frozenset({PROCESS}) if killed else frozenset()
    if isinstance(child, ast.Compare):
        kind = _compare_kind(child)
        return frozenset() if kind is None else frozenset({kind})
    if isinstance(child, ast.Constant) and isinstance(child.value, str):
        exits = NONZERO_EXIT_TEXT.search(child.value) is not None
        return frozenset({PROCESS}) if exits else frozenset()
    return frozenset()


def failure_kinds(node: ast.AST) -> frozenset[str]:
    """Collect the kinds of failure a function names.

    Args:
        node: A test function or helper.

    Returns:
        Every kind a marker in it counts for; empty when it names none.
    """
    kinds: set[str] = set()
    for child in ast.walk(node):
        kinds |= _node_kinds(child)
    return frozenset(kinds)


__all__ = [
    "EXIT_NAMES",
    "FAILURE_ATTRIBUTES",
    "FAILURE_NAME_KINDS",
    "FILE_SWAP",
    "KILL_CALLS",
    "NETWORK",
    "NONZERO_EXIT_TEXT",
    "PROCESS",
    "SATISFIED_BY",
    "SERVICE",
    "SSH",
    "STATUS_NAMES",
    "failure_kinds",
    "terminal_name",
]
