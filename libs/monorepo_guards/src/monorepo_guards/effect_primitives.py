"""What counts as an external effect, and how a call is resolved to one.

Shared by the two rules MCPs board task c96e8791 defined and API board task
cc7222ca applies here: ``effect-seam-twin`` (every fake has a real twin) and
``state-change-verified`` (an install, swap, deploy or restart verifies what
it changed and can put it back). Both ask the same first question of a call,
"does this reach a process, ssh, the network, a file swap or a service?", so
the answer is written once.

The definitions are the ones opus-coordination-w36-1008 fixed for MCPs'
mcp-shared and mcp-shared-py on 2026-10-08, matched here so the two
repositories mean the same thing by an effect. A primitive is named by its
fully qualified dotted path, resolved through the module's own imports, so
``import subprocess as sp; sp.run(...)`` and ``from subprocess import run;
run(...)`` are both ``subprocess.run``. A call through an attribute of an
object (``client.get(...)``) is never resolved: only the module that made
the client knows what it is, and that module is where the primitive shows.

EVERY PRIMITIVE IS AN EXACT NAME, never a whole module. Measured on API on
2026-10-08: its src constructs ``web.Response`` 28 times,
``subprocess.CalledProcessError`` three times and ``httpx.Timeout`` once,
and a whole-module match read TankpitBot's ``restart_bot`` route as an
aiohttp effect because it built ``web.Response(status=409)``; and
``socket.gethostname`` reads a local name no test can make fail. So each
module contributes only the calls that start a process, open a connection,
move a file or reach a service.
"""

from __future__ import annotations

import ast
from typing import NamedTuple

from monorepo_guards.util import module_nodes

#: Each kind of effect and the exact qualified calls that are it. The kind is
#: printed in the lint line.
PRIMITIVE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "process",
        (
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.getoutput",
            "subprocess.getstatusoutput",
            "asyncio.create_subprocess_exec",
            "asyncio.create_subprocess_shell",
            "os.system",
            "os.kill",
            "os.killpg",
            "os.execv",
            "os.execve",
            "os.execvp",
            "os.execvpe",
            "os.execl",
            "os.execle",
            "os.execlp",
            "os.execlpe",
            "os.spawnv",
            "os.spawnve",
            "os.spawnvp",
            "os.spawnvpe",
            "os.spawnl",
            "os.spawnle",
            "os.spawnlp",
            "os.spawnlpe",
            "os.startfile",
        ),
    ),
    (
        "network",
        (
            "socket.socket",
            "socket.create_connection",
            "socket.create_server",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyname_ex",
            "socket.gethostbyaddr",
            "socket.getfqdn",
            "urllib.request.urlopen",
            "http.client.HTTPConnection",
            "http.client.HTTPSConnection",
            "httpx.Client",
            "httpx.AsyncClient",
            "httpx.get",
            "httpx.post",
            "httpx.put",
            "httpx.patch",
            "httpx.delete",
            "httpx.head",
            "httpx.options",
            "httpx.request",
            "httpx.stream",
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.patch",
            "requests.delete",
            "requests.head",
            "requests.options",
            "requests.request",
            "requests.Session",
            "aiohttp.ClientSession",
            "aiohttp.request",
            "aiohttp.web.run_app",
            "aiohttp.web.TCPSite",
            "aiohttp.web.UnixSite",
            "aiohttp.web.SockSite",
            "aiohttp.web.NamedPipeSite",
            "websockets.connect",
            "websockets.serve",
            "smtplib.SMTP",
            "smtplib.SMTP_SSL",
            "smtplib.LMTP",
        ),
    ),
    ("ssh", ("paramiko.SSHClient", "paramiko.Transport")),
    (
        "file swap",
        ("os.replace", "os.rename", "os.renames", "shutil.move", "shutil.rmtree"),
    ),
    (
        "service",
        (
            "psycopg.connect",
            "redis.Redis",
            "redis.StrictRedis",
            "redis.from_url",
            "redis.Sentinel",
            "redis.asyncio.Redis",
            "redis.asyncio.from_url",
        ),
    ),
)

#: Qualified call to its kind, the lookup :func:`primitive_of` makes.
PRIMITIVE_KINDS: dict[str, str] = {
    called: kind for kind, calls in PRIMITIVE_GROUPS for called in calls
}


def _relative_base(module_name: str, is_package: bool, level: int) -> str:
    """Name the package a relative import of ``level`` dots starts from.

    Args:
        module_name: Dotted name of the importing module.
        is_package: True when the importing file is an ``__init__.py``,
            whose own name is the package one dot resolves to.
        level: The number of leading dots.

    Returns:
        The dotted base, empty when the dots climb past the top.
    """
    parts = module_name.split(".") if module_name else []
    if not is_package:
        parts = parts[:-1]
    keep = len(parts) - (level - 1)
    return ".".join(parts[: max(keep, 0)])


def _dynamic_import(value: ast.expr) -> str | None:
    """Name the top-level module an ``__import__("...")`` call returns.

    Args:
        value: An assignment's value.

    Returns:
        The first component of the imported name, or None when the value
        is not ``__import__`` called with one string literal.
    """
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    if value.func.id != "__import__" or len(value.args) != 1:
        return None
    argument = value.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None
    return argument.value.split(".", maxsplit=1)[0]


def import_bindings(tree: ast.Module, module_name: str, is_package: bool) -> dict[str, str]:
    """Map every name a module binds through an import to what it names.

    Args:
        tree: Parsed module.
        module_name: The module's dotted name, used for relative imports.
        is_package: True for an ``__init__.py``.

    Returns:
        Local name to fully qualified dotted name. ``import a.b`` binds
        ``a`` to ``a``; ``import a.b as c`` binds ``c`` to ``a.b``;
        ``from a import b`` binds ``b`` to ``a.b``; and
        ``x = __import__("a.b")``, the coding standard's own dynamic import,
        binds ``x`` to ``a``, which is what ``__import__`` returns. Without
        that, RustedWarfareBot's and TankpitBot's database connects
        (``psycopg = __import__("psycopg")``) were invisible.
    """
    bindings: dict[str, str] = {}
    for node in module_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            imported = _dynamic_import(node.value)
            if isinstance(target, ast.Name) and imported is not None:
                bindings[target.id] = imported
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    bindings[alias.asname] = alias.name
                else:
                    root = alias.name.split(".", maxsplit=1)[0]
                    bindings[root] = root
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level > 0:
                start = _relative_base(module_name, is_package, node.level)
                base = f"{start}.{base}" if start and base else start or base
            for alias in node.names:
                local = alias.asname or alias.name
                bindings[local] = f"{base}.{alias.name}" if base else alias.name
    return bindings


def dotted_name(node: ast.expr) -> str | None:
    """Render a ``Name``/``Attribute`` chain as dotted text.

    Args:
        node: Expression, usually a call's ``func``.

    Returns:
        ``a.b.c`` for ``a.b.c``, or None when the chain is rooted in
        anything but a plain name (a call, a subscript, a literal).
    """
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def qualified_name(node: ast.expr, bindings: dict[str, str]) -> str | None:
    """Resolve a dotted expression's root through the module's imports.

    Args:
        node: Expression, usually a call's ``func``.
        bindings: The module's :func:`import_bindings`.

    Returns:
        The dotted name with its root replaced by what the import bound,
        the dotted name unchanged when its root is not imported, or None
        when the expression is not a dotted chain at all.
    """
    dotted = dotted_name(node)
    if dotted is None:
        return None
    root, _, rest = dotted.partition(".")
    bound = bindings.get(root)
    if bound is None:
        return dotted
    return f"{bound}.{rest}" if rest else bound


class PrimitiveCall(NamedTuple):
    """A call that is an effect primitive.

    Attributes:
        kind: The primitive's kind (process, network, ssh, file swap,
            service).
        called: The qualified name actually called, e.g.
            ``subprocess.Popen``, which is what a lint line prints.
    """

    kind: str
    called: str


def primitive_reference(node: ast.expr, bindings: dict[str, str]) -> PrimitiveCall | None:
    """Name the effect primitive an expression refers to, if it is one.

    Only an imported root counts: a local variable spelled ``socket`` is
    not the socket module.

    Args:
        node: A ``Name``/``Attribute`` chain, called or not.
        bindings: The module's :func:`import_bindings`.

    Returns:
        The primitive, or None when the expression is not one.
    """
    dotted = dotted_name(node)
    if dotted is None:
        return None
    root, _, rest = dotted.partition(".")
    bound = bindings.get(root)
    if bound is None:
        return None
    qualified = f"{bound}.{rest}" if rest else bound
    kind = PRIMITIVE_KINDS.get(qualified)
    if kind is None:
        return None
    return PrimitiveCall(kind=kind, called=qualified)


def bound_primitive(value: ast.expr, bindings: dict[str, str]) -> PrimitiveCall | None:
    """Name the primitive a hook or a local is bound to directly, if any.

    Args:
        value: The bound value: ``subprocess.run``, ``psycopg.connect`` or
            ``fetch.bind(...)``'s Python counterpart, ``x.y.bind(...)``.

    Returns:
        The primitive, or None when the value is not one.
    """
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "bind"
    ):
        return primitive_reference(value.func.value, bindings)
    return primitive_reference(value, bindings)


def local_aliases(function: ast.AST, bindings: dict[str, str]) -> dict[str, PrimitiveCall]:
    """Map every local name a function binds directly to a primitive.

    The coding standard's dynamic-import pattern binds a Protocol-typed
    local and calls that (``connector: P = psycopg.connect;
    connector(dsn)``), so a call of such a local is a call of the primitive.
    A bare reference is not enough: ``reset_hooks`` reassigns
    ``run_process = subprocess.run`` and runs nothing.

    Args:
        function: The function or lambda.
        bindings: Its module's imports.

    Returns:
        Local name to the primitive it holds.
    """
    aliases: dict[str, PrimitiveCall] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets: list[ast.expr] = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        primitive = bound_primitive(value, bindings)
        if primitive is not None:
            aliases.update((t.id, primitive) for t in targets if isinstance(t, ast.Name))
    return aliases


def primitive_of(
    call: ast.Call, bindings: dict[str, str], aliases: dict[str, PrimitiveCall]
) -> PrimitiveCall | None:
    """Name the effect primitive a call invokes, if it invokes one.

    Args:
        call: The call.
        bindings: The calling module's :func:`import_bindings`.
        aliases: The calling function's :func:`local_aliases`.

    Returns:
        The primitive call, or None when the call is not one.
    """
    if isinstance(call.func, ast.Name) and call.func.id in aliases:
        return aliases[call.func.id]
    return primitive_reference(call.func, bindings)


def calls_in_order(node: ast.AST) -> list[ast.Call]:
    """List every call under a node in source order.

    Args:
        node: A function, a lambda or any other subtree.

    Returns:
        The calls ordered by line and column, so "the first primitive
        found" is the first one a reader meets. A lambda nested inside is
        skipped: building ``Hooks(kill=lambda pid: os.kill(pid, 9))`` kills
        nothing, and the lambda is a seam of its own.
    """
    calls: list[ast.Call] = []
    pending = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        if isinstance(child, ast.Lambda):
            continue
        if isinstance(child, ast.Call):
            calls.append(child)
        pending.extend(ast.iter_child_nodes(child))
    return sorted(calls, key=_position)


def _position(call: ast.Call) -> tuple[int, int]:
    """Key a call by where it starts.

    Args:
        call: The call.

    Returns:
        Its line and column.
    """
    return (call.lineno, call.col_offset)


__all__ = [
    "PRIMITIVE_GROUPS",
    "PRIMITIVE_KINDS",
    "PrimitiveCall",
    "bound_primitive",
    "calls_in_order",
    "dotted_name",
    "import_bindings",
    "local_aliases",
    "primitive_of",
    "qualified_name",
]
