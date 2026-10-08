"""Guard rule ``state-change-verified``: no change goes unverified or unreversible.

The harness gate killed a hung ``claude.exe install`` mid-swap on 2026-10-07
and left ``C:\\Users\\Test\\.local\\bin`` with no ``claude.exe``; its
``promote()`` never checked that the launcher existed afterwards, and had
nothing to put back, so every SMS text the operator sent failed until a
session noticed (MCPs board task 5895c980). The operator: "this cannot
stand".

So a function that installs, swaps, deploys or restarts something must read
back the state it produced and must be able to put the previous state back
when that reading is wrong. The definition is MCPs' (board task c96e8791),
applied here under API board task cc7222ca:

* A STATE CHANGE is a function in ``src`` or ``scripts`` whose name LEADS
  with one of :data:`STATE_CHANGE_VERBS` as its first snake or camel word
  (``install_local``, ``promote``, ``restartHost``; further in, the same
  words are nouns: ``record_deploy``) AND whose body performs an effect: a
  primitive (:mod:`monorepo_guards.effect_primitives`) or a call through a
  hooks object (a receiver whose dotted name contains ``hooks``, or the
  result of a ``get*Hooks()``/``default_*hooks()`` call) naming an
  attribute some effect seam is exposed by
  (:func:`monorepo_guards.effect_seams.exposed_names`): ``hooks.run`` acts,
  ``hooks.now`` does not. The effect may be reached through the package's
  own functions, called by bare name or as ``module.fn`` on an import of a
  package module (never ``obj.method``). The effect clause keeps an
  in-process ``install_signal_handler`` out.
* VERIFY is a call whose callee name, leading underscores dropped, starts
  with ``verify``, ``check`` or ``probe`` or contains ``exists``.
* RESTORE is a call whose callee name starts with ``restore``, ``rollback``,
  ``roll_back`` or ``revert``, sitting under an ``if``/``else``, an
  ``except`` handler or a ``finally``: a restore on the straight-line path
  undoes every success too.
* Both may be reached through same-module helper functions called by bare
  name. There is no allow-list.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.effect_primitives import (
    dotted_name,
    local_aliases,
    primitive_of,
    qualified_name,
)
from monorepo_guards.effect_seams import (
    PackageModule,
    effect_seams,
    exposed_names,
    index_package,
    resolve_function,
    top_level_functions,
)

#: ``rollout`` is not here: ``rollout_path`` reads a Codex rollout file (MCPs,
#: measured by c96e8791's first full run on 2026-10-08).
STATE_CHANGE_VERBS = frozenset(
    {
        "install",
        "reinstall",
        "uninstall",
        "swap",
        "deploy",
        "redeploy",
        "restart",
        "promote",
        "upgrade",
    }
)
VERIFY_PREFIXES = ("verify", "check", "probe")
RESTORE_PREFIXES = ("restore", "rollback", "roll_back", "revert")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def leading_word(name: str) -> str:
    """Name the first snake or camel word of a function name.

    Args:
        name: A function name, e.g. ``promoteBuild`` or ``_swap_launcher``.

    Returns:
        Its first word, lower-cased, leading underscores dropped.
    """
    return _CAMEL_BOUNDARY.sub("_", name.lstrip("_")).lower().split("_")[0]


def _callee(call: ast.Call) -> str | None:
    """Name what a call calls, by its last component.

    Args:
        call: The call.

    Returns:
        The function or method name, leading underscores dropped, or None
        when the callee is not a name or attribute.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id.lstrip("_")
    if isinstance(func, ast.Attribute):
        return func.attr.lstrip("_")
    return None


def _is_hooks_receiver(receiver: ast.expr) -> bool:
    """Report whether a method call's receiver is a hooks object.

    Args:
        receiver: What the method is called on.

    Returns:
        True for a dotted name containing ``hooks`` or a call to a
        ``get*Hooks()``/``default_*hooks()`` factory.
    """
    if isinstance(receiver, ast.Call):
        made_by = (_callee(receiver) or "").lower()
        return made_by.endswith("hooks") and made_by.startswith(("get", "default_"))
    dotted = dotted_name(receiver)
    return dotted is not None and "hooks" in dotted.lower()


def _first_party_callee(
    index: dict[str, PackageModule], module: PackageModule, call: ast.Call
) -> tuple[PackageModule, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Resolve a call to a function of the same package, when it is one.

    Args:
        index: The package's modules.
        module: The calling module.
        call: The call.

    Returns:
        The defining module and function for a bare name, or for
        ``module.fn`` where ``module`` is an import of a package module;
        None for anything else, an ``obj.method`` call included.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return resolve_function(index, module, func.id, set())
    qualified = qualified_name(func, module.bindings)
    if qualified is None:
        return None
    source_name, _, attr = qualified.rpartition(".")
    source = index.get(source_name)
    if source is None:
        return None
    return resolve_function(index, source, attr, set())


def performs_effect(
    index: dict[str, PackageModule],
    module: PackageModule,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    exposed: frozenset[str],
    visited: set[tuple[str, str]],
) -> bool:
    """Report whether a function acts on the world, itself or by delegation.

    ``install_missing`` in tools/fleet installs tools on a fleet node through
    ``remote.run_script``, three calls from the ssh, which is why the walk
    follows first-party calls rather than reading the body alone.

    Args:
        index: The package's modules.
        module: The function's module.
        function: The function.
        exposed: The attributes the package's effect seams are reached by.
        visited: ``(module, function)`` pairs already walked.

    Returns:
        True when it, or a package function it calls by bare name or as
        ``module.fn``, calls a primitive or calls one of ``exposed`` through
        a hooks object.
    """
    aliases = local_aliases(function, module.bindings)
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if primitive_of(node, module.bindings, aliases) is not None:
            return True
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in exposed
            and _is_hooks_receiver(func.value)
        ):
            return True
        target = _first_party_callee(index, module, node)
        if target is None:
            continue
        key = (target[0].name, target[1].name)
        if key in visited:
            continue
        visited.add(key)
        if performs_effect(index, target[0], target[1], exposed, visited):
            return True
    return False


def _guarded_call_ids(function: ast.AST) -> set[int]:
    """Collect the calls that sit under a branch, a handler or a finally.

    Args:
        function: The function.

    Returns:
        ``id`` of every call inside an ``if`` body or ``else``, an
        ``except`` handler or a ``finally`` block.
    """
    blocks: list[ast.stmt] = []
    for node in ast.walk(function):
        if isinstance(node, ast.If):
            blocks.extend(node.body)
            blocks.extend(node.orelse)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            for handler in node.handlers:
                blocks.extend(handler.body)
            blocks.extend(node.finalbody)
    return {
        id(child) for block in blocks for child in ast.walk(block) if isinstance(child, ast.Call)
    }


def _reached(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    helpers: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """List a function and the same-module helpers it calls, transitively.

    Args:
        function: The state change.
        helpers: The module's top-level functions.

    Returns:
        The function first, then each helper once.
    """
    reached = [function]
    names = {function.name}
    position = 0
    while position < len(reached):
        for node in ast.walk(reached[position]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                helper = helpers.get(node.func.id)
                if helper is not None and helper.name not in names:
                    names.add(helper.name)
                    reached.append(helper)
        position += 1
    return reached


def missing_safeguards(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    helpers: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[str]:
    """Say which of verify and restore a state change lacks.

    Args:
        function: The state change.
        helpers: The module's top-level functions.

    Returns:
        ``"verify"`` and/or ``"restore"``, in that order; empty when both
        are present.
    """
    verifies = False
    restores = False
    for reached in _reached(function, helpers):
        guarded = _guarded_call_ids(reached)
        for node in ast.walk(reached):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee(node) or ""
            if callee.startswith(VERIFY_PREFIXES) or "exists" in callee:
                verifies = True
            if callee.startswith(RESTORE_PREFIXES) and id(node) in guarded:
                restores = True
    missing: list[str] = []
    if not verifies:
        missing.append("verify")
    if not restores:
        missing.append("restore")
    return missing


def _line_of(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Key a function by where it is defined.

    Args:
        function: The function.

    Returns:
        Its line, so a method is reported in place rather than after every
        top-level function the walk reaches first.
    """
    return function.lineno


class StateChangeVerifiedRule:
    """Fail every install, swap, deploy or restart without verify and restore."""

    name = "state-change-verified"

    def __init__(self, config: GuardConfig) -> None:
        """Bind the rule to the package it checks.

        Args:
            config: The guard run's configuration; its ``root`` decides
                which files are ``src`` and ``scripts``.
        """
        self._root = config.root

    def run(self, files: list[Path]) -> list[Violation]:
        """Report each state change missing a verify or a restore.

        Args:
            files: The package's files; tests are not read.

        Returns:
            One violation per state change, naming what it lacks.
        """
        violations: list[Violation] = []
        index = index_package(files, self._root)
        exposed = exposed_names(effect_seams(index))
        for name in sorted(index):
            module = index[name]
            helpers = top_level_functions(module.tree)
            where = module.path.relative_to(self._root).as_posix()
            functions = [
                node
                for node in ast.walk(module.tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for node in sorted(functions, key=_line_of):
                if leading_word(node.name) not in STATE_CHANGE_VERBS:
                    continue
                if not performs_effect(index, module, node, exposed, {(name, node.name)}):
                    continue
                missing = missing_safeguards(node, helpers)
                if not missing:
                    continue
                violations.append(
                    Violation(
                        file=module.path,
                        line_no=node.lineno,
                        kind="state-change-unverified",
                        line=f"{where}:{node.lineno} {node.name} has no {' and no '.join(missing)}",
                    )
                )
        return violations


__all__ = [
    "RESTORE_PREFIXES",
    "STATE_CHANGE_VERBS",
    "VERIFY_PREFIXES",
    "StateChangeVerifiedRule",
    "leading_word",
    "missing_safeguards",
    "performs_effect",
]
