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
  ``pytest.raises(OperationalError)``, ``errno.ENOENT``, or any WORD of a
  string literal that is one (``"connect ECONNREFUSED 127.0.0.1:1"``), and
  the phrases ``fetch failed`` and ``connection refused`` (any case), and
  a listener's bind failures ``EADDRINUSE`` and ``EADDRNOTAVAIL``: network;
* a read of ``timed_out`` (attribute or key): process; a read of
  ``killed``, ``SIGKILL`` or ``SIGTERM``, and a call whose final name
  carries ``kill`` or ``terminate`` as a snake or camel word
  (``terminate_process(pid)``) or is ``killpg``: process and file swap,
  since a swap killed mid-way is MCPs board task 5895c980 itself;
* an exit code (``exit_code``, ``returncode``, ``exitcode``, ``code``)
  compared equal to a nonzero literal, unequal to 0 or above 0, or stated
  as a nonzero keyword argument or string-keyed dict entry
  (``returncode=3``, ``{"exit_code": 3}``): process;
* a ``status`` or ``status_code`` compared to, or stated as a keyword or
  dict entry of, a literal of 400 or more: network;
* a string literal that exits nonzero (``sys.exit(3)``, ``process.exit(1)``,
  ``exit 2``), the child script a real process test runs, or that reports
  one (``pg_dump exited 2``, ``exit status 2``) or a spawn that found no
  executable (``spawn claude.exe ENOENT``): process.

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
    # A kill is also a file-swap failure: a swap killed mid-way is MCPs
    # board task 5895c980 itself, the launcher left missing.
    "SIGKILL": frozenset({PROCESS, FILE_SWAP}),
    "SIGTERM": frozenset({PROCESS, FILE_SWAP}),
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
    # A listener fails at bind, not at connect: aiohttp's TCPSite on an
    # address the machine does not hold (board task cc7222ca).
    "EADDRINUSE": frozenset({NETWORK}),
    "EADDRNOTAVAIL": frozenset({NETWORK}),
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

KILL_WORDS = frozenset({"kill", "terminate"})
#: What a kill is a failure of: the process, and a swap it interrupts.
KILL_KINDS = frozenset({PROCESS, FILE_SWAP})
FAILURE_ATTRIBUTES = frozenset({"timed_out", "killed"})
EXIT_NAMES = frozenset({"exit_code", "returncode", "exitcode", "code"})
STATUS_NAMES = frozenset({"status", "status_code", "statusCode"})
#: A status read the exit way below this is a child's exit, not HTTP: Node's
#: spawnSync reports a child's exit as ``.status`` (MCPs wiki-check's curl
#: seam fails with 37), and no HTTP status is below 100, so the two ranges
#: never meet. From MCPs' table (board task c96e8791).
EXIT_STATUS_CEILING = 100
NETWORK_PHRASES = ("fetch failed", "connection refused")
#: Text that says a process failed: a nonzero exit, stated as code or as a
#: message (``pg_dump exited 2``, ``exit status 2``), or a spawn that found
#: no executable (``spawn claude.exe ENOENT``, MCPs board task 5895c980).
#: Verbatim from MCPs' table (board task c96e8791).
NONZERO_EXIT_TEXT = re.compile(
    r"\b(?:sys\.exit|process\.exit|SystemExit|exit(?:ed)?)(?:\s+with)?(?:\s+(?:code|status))?"
    r"\s*\(?\s*[1-9]|\bspawn\b[^\n]*\bENOENT\b"
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _names_a_kill(name: str | None) -> bool:
    """Report whether a called name kills or terminates something.

    Args:
        name: The call's final name.

    Returns:
        True for ``killpg`` and for a name with ``kill`` or ``terminate``
        as a snake or camel word (``terminate_process``,
        ``nodeTerminateProcess``).
    """
    if name is None:
        return False
    words = set(_CAMEL_BOUNDARY.sub("_", name).lower().split("_"))
    return name == "killpg" or bool(words & KILL_WORDS)


def _stated_kind(name: str | None, value: int | None) -> str | None:
    """Name the failure a member stated equal to a literal is, if one.

    Args:
        name: The member, e.g. ``returncode`` or ``status``.
        value: The integer it is stated or compared equal to.

    Returns:
        Process for a nonzero exit code or a status between 0 and
        :data:`EXIT_STATUS_CEILING`, network for a status of 400 or more,
        None otherwise.
    """
    if value is None:
        return None
    if name in EXIT_NAMES:
        return PROCESS if value != 0 else None
    if name in STATUS_NAMES:
        if value >= 400:
            return NETWORK
        return PROCESS if 0 < value < EXIT_STATUS_CEILING else None
    return None


def _text_kinds(text: str) -> frozenset[str]:
    """Name the failures a string literal states.

    Args:
        text: The literal.

    Returns:
        The kinds of every failure name among its words, network for the
        two phrases a refused fetch or connect prints, process for an inline
        script that exits nonzero.
    """
    kinds: set[str] = set()
    spaced = "".join(char if char.isalnum() or char == "_" else " " for char in text)
    for word in spaced.split():
        kinds |= FAILURE_NAME_KINDS.get(word, frozenset())
    lowered = text.lower()
    if any(phrase in lowered for phrase in NETWORK_PHRASES):
        kinds.add(NETWORK)
    if NONZERO_EXIT_TEXT.search(text) is not None:
        kinds.add(PROCESS)
    return frozenset(kinds)


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
        Process for a nonzero exit code, or a status read the exit way below
        :data:`EXIT_STATUS_CEILING`; network for a status equal to or at
        least 400; None otherwise. Exit members are read first.
    """
    left, op, right = node.left, node.ops[0], node.comparators[0]
    names = (terminal_name(left), terminal_name(right))
    values = (_int_literal(right), _int_literal(left))
    sides = ((names[0], values[0], False), (names[1], values[1], True))
    for name, value, flipped in sides:
        if name in EXIT_NAMES:
            failed = value is not None and _asserts_nonzero(op, value, flipped, ceiling=None)
            return PROCESS if failed else None
    for name, value, flipped in sides:
        if name in STATUS_NAMES:
            return _status_kind(op, value, flipped)
    return None


def _status_kind(op: ast.cmpop, value: int | None, flipped: bool) -> str | None:
    """Name the failure a comparison of a status against a literal asserts.

    Args:
        op: The comparison's operator.
        value: The literal compared against, or None when it is not one.
        flipped: True when the status is the right-hand side.

    Returns:
        Network for ``== N`` or ``>= N`` with N of 400 or more, process for
        a nonzero exit read below :data:`EXIT_STATUS_CEILING`, else None.
    """
    if value is None:
        return None
    at_least = ast.LtE if flipped else ast.GtE
    if value >= 400 and isinstance(op, (ast.Eq, at_least)):
        return NETWORK
    if _asserts_nonzero(op, value, flipped, ceiling=EXIT_STATUS_CEILING):
        return PROCESS
    return None


def _asserts_nonzero(op: ast.cmpop, value: int, flipped: bool, *, ceiling: int | None) -> bool:
    """Report whether comparing an exit against a literal asserts it is nonzero.

    Args:
        op: The comparison's operator.
        value: The literal.
        flipped: True when the exit is the right-hand side.
        ceiling: For a status, the bound an equal value must stay under to
            be an exit; None for an exit member, where any nonzero counts.

    Returns:
        True for ``== N`` with N nonzero (and between 0 and ``ceiling`` when
        one is given), ``!= 0``, and ``> 0`` (``0 <`` flipped).
    """
    if isinstance(op, ast.Eq):
        return value != 0 if ceiling is None else 0 < value < ceiling
    if isinstance(op, ast.NotEq):
        return value == 0
    above = ast.Lt if flipped else ast.Gt
    return isinstance(op, above) and value == 0


def _only(kind: str | None) -> frozenset[str]:
    """Wrap one optional kind as a set.

    Args:
        kind: A kind, or None.

    Returns:
        The kind alone, or nothing.
    """
    return frozenset() if kind is None else frozenset({kind})


def _dict_kinds(node: ast.Dict) -> frozenset[str]:
    """Name the failures a dict literal states by its string keys.

    Args:
        node: The dict, e.g. ``{"status": 502}``.

    Returns:
        The kinds its exit and status entries state.
    """
    kinds: set[str] = set()
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            kinds |= _only(_stated_kind(key.value, _int_literal(value)))
    return frozenset(kinds)


def _node_kinds(child: ast.AST) -> frozenset[str]:
    """Name the failure kinds one node is a marker of.

    Args:
        child: Any node of a test or helper.

    Returns:
        The kinds, empty for a node that marks no failure.
    """
    if isinstance(child, (ast.Name, ast.Attribute, ast.Subscript)):
        name = terminal_name(child) or ""
        if not isinstance(child, ast.Name) and name == "killed":
            return KILL_KINDS
        if not isinstance(child, ast.Name) and name in FAILURE_ATTRIBUTES:
            return frozenset({PROCESS})
        return FAILURE_NAME_KINDS.get(name, frozenset())
    if isinstance(child, ast.Call):
        return KILL_KINDS if _names_a_kill(terminal_name(child.func)) else frozenset()
    if isinstance(child, ast.Compare):
        return _only(_compare_kind(child))
    if isinstance(child, ast.keyword):
        return _only(_stated_kind(child.arg, _int_literal(child.value)))
    if isinstance(child, ast.Dict):
        return _dict_kinds(child)
    if isinstance(child, ast.Constant) and isinstance(child.value, str):
        return _text_kinds(child.value)
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
    "KILL_KINDS",
    "KILL_WORDS",
    "NETWORK",
    "NETWORK_PHRASES",
    "NONZERO_EXIT_TEXT",
    "PROCESS",
    "SATISFIED_BY",
    "SERVICE",
    "SSH",
    "STATUS_NAMES",
    "failure_kinds",
    "terminal_name",
]
