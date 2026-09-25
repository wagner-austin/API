"""Guard rule: every subprocess call is bounded, and a bound is not a promise.

THE FIRST INCIDENT, which is why a deadline is required at all. The fleet
agent's queue drained nothing from 2026-09-17 to 2026-09-20 because a single
tick never exited: one ``ssh`` to a sleeping laptop with no ``timeout``, with
the scheduled task's ``MultipleInstances`` set to ``IgnoreNew`` dropping every
tick behind it. Three days of the fleet doing nothing, from one unbounded
child (board tasks 41ac6ed2, 35940277, 0d891468).

THE SECOND INCIDENT, which is why passing ``timeout=`` is not enough, and
which is the half a rule reading only for the keyword would mark green.
``subprocess.run(input=...)`` hands the payload over by WRITING IT FROM THE
CALLING THREAD, and on Windows CPython that write is synchronous in the
caller while only the stdout and stderr readers get threads of their own.
The deadline governs what happens after the write, so it does not govern the
write. Measured on austinpc 2026-09-24, Python 3.11.9, against a child that
never drains its pipe: 60 MB with ``timeout=5`` was still blocked at 100
seconds, and 8 MB returned after 60.2 seconds, ended by the CHILD exiting
rather than by the clock. It cost a fleet staging send 40 minutes against a
120-second bound, reporting nothing, four days after every remote command in
that package had been given a deadline for exactly the first reason above
(board task 1e57ebe5).

So the rule has two clauses, and the second is the one with teeth:

* A call with no ``timeout=`` is refused. It has no bound.
* A call with ``timeout=`` AND ``input=`` is refused. It reads as bounded
  and is not, which is worse than the first: the first is visible to anyone
  grepping for the keyword and the second survives that grep.

``stdin=`` IS NOT ``input=``, AND THE DIFFERENCE IS THE WHOLE REMEDY. Handing
the child a FILE it reads itself costs the calling thread nothing, so the
deadline governs the entire call. ``tools/fleet/src/fleet/core/_command.py``
is written that way on purpose and is the one call site in this monorepo that
gets this right; a clause keyed on ``stdin=`` would refuse the fix and pass
the defect.

SIZE IS DELIBERATELY NOT A CRITERION. Whether an ``input=`` write blocks
depends on whether the child drains its pipe, not on how many bytes it is, so
there is no threshold that separates a safe payload from an unsafe one. A
small payload that happens not to fill a pipe buffer is not bounded; it is
lucky, and it stops being lucky when the child changes.

``Popen.communicate(input=..., timeout=...)`` CARRIES THE IDENTICAL HOLE, and
is checked here rather than set aside as "a Popen shape". ``subprocess.run``
is implemented on top of it, so the measurement above is a measurement of
``communicate``. Putting the bound on ``communicate`` instead of on ``wait``
is necessary and not sufficient.

SCOPE IS ``src`` AND ``scripts``, NOT TESTS, and that is a reasoned boundary
rather than an exemption. A test's unbounded child is bounded by the CI job
it runs inside, and that job's own bound is enforced by
:mod:`monorepo_guards.workflow_timeout_rules` for the same incident class. A
shipped module has no such enclosing clock.

A SPLATTED KEYWORD IS AN UNPROVEN DEADLINE, and is refused as one. A call
written ``subprocess.run(argv, **options)`` may or may not carry a timeout,
and no reading of the source can say which; a rule that passed it would be
reporting a clean result over a call it did not examine, and would hand
anyone a one-line bypass. There are no such call sites today (measured
2026-09-25, zero), so this costs nothing now and closes the hole before it
is used.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar, Final

from monorepo_guards import Violation
from monorepo_guards.util import parse_source

#: The subprocess helpers that run a child to completion in one call. Each
#: accepts ``timeout``, so each can be bounded and each must be.
AWAITING_CALLS: Final[frozenset[str]] = frozenset({"run", "check_output", "check_call", "call"})

#: ``Popen.communicate``, checked for the paired-payload clause only: it is
#: the call the others are built on, and the one that carries ``input``.
COMMUNICATE: Final[str] = "communicate"

#: A call that cannot outlive anything, because nothing bounds it.
MISSING_DEADLINE: Final[str] = "subprocess-no-timeout"

#: A call that reads as bounded and is not: the payload is written before the
#: deadline governs.
UNBOUNDED_PAYLOAD: Final[str] = "subprocess-timeout-with-input"

#: A call whose keywords are splatted, so no reading proves a deadline.
UNPROVEN_DEADLINE: Final[str] = "subprocess-unprovable-timeout"


class SubprocessTimeoutRule:
    """Refuse an unbounded subprocess call, and a bound that is a false promise.

    Attributes:
        name: The rule's name in the guard summary.
    """

    #: The rule's name in the guard summary, as board task 0d891468 named it.
    #: Spelled REQUIRED rather than "subprocess-timeout" because the summary
    #: line is read as a claim: "subprocess-timeout: 0 violations" reads like a
    #: topic that was looked at, and "subprocess-timeout-required: 0
    #: violations" reads like a rule that held.
    name = "subprocess-timeout-required"

    #: Directories whose Python ships; a file outside both is not judged.
    _SCOPES: ClassVar[frozenset[str]] = frozenset({"src", "scripts"})

    def _in_scope(self, path: Path) -> bool:
        """Whether this file's subprocess calls are judged.

        Args:
            path: The file.

        Returns:
            True for a file under ``src/`` or ``scripts/`` and not under a
            ``tests/`` directory, per the module docstring's scope note.
        """
        parts = path.as_posix().split("/")
        if "tests" in parts:
            return False
        return any(scope in parts for scope in self._SCOPES)

    def _imported_names(self, tree: ast.Module) -> frozenset[str]:
        """The awaiting helpers this module bound by a ``from`` import.

        ``from subprocess import run`` makes a bare ``run(...)`` the same call
        as ``subprocess.run(...)``, and resolving it by import rather than by
        name is what keeps an unrelated local function called ``run`` out of
        the report.

        Args:
            tree: The module's AST.

        Returns:
            The local names, ``as`` aliases included, that refer to one of
            :data:`AWAITING_CALLS`.
        """
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "subprocess":
                continue
            for alias in node.names:
                if alias.name in AWAITING_CALLS:
                    names.add(alias.asname if alias.asname is not None else alias.name)
        return frozenset(names)

    def _awaiting_callee(self, node: ast.Call, imported: frozenset[str]) -> str | None:
        """The awaiting subprocess helper a call names, if it names one.

        Args:
            node: The call.
            imported: Local names bound by :meth:`_imported_names`.

        Returns:
            The helper's name, or None when the call is not one of them.
        """
        func = node.func
        if isinstance(func, ast.Attribute):
            value = func.value
            if (
                isinstance(value, ast.Name)
                and value.id == "subprocess"
                and func.attr in AWAITING_CALLS
            ):
                return func.attr
            return None
        if isinstance(func, ast.Name) and func.id in imported:
            return func.id
        return None

    def _is_communicate(self, node: ast.Call) -> bool:
        """Whether a call is a ``.communicate(...)`` on some object.

        Args:
            node: The call.

        Returns:
            True for any attribute call named ``communicate``. The receiver is
            not resolved: a ``communicate`` that takes ``input`` and
            ``timeout`` together is the shape this refuses whatever the object
            is called, and no other API in this monorepo spells that pair.
        """
        func = node.func
        return isinstance(func, ast.Attribute) and func.attr == COMMUNICATE

    def _judge(self, path: Path, node: ast.Call, source: str) -> Violation | None:
        """Judge one already-recognised call.

        Args:
            path: The file it is in.
            node: The call.
            source: What the report should quote for it.

        Returns:
            The violation, or None when the call is bounded.
        """
        if any(keyword.arg is None for keyword in node.keywords):
            return Violation(file=path, line_no=node.lineno, kind=UNPROVEN_DEADLINE, line=source)
        named = {keyword.arg for keyword in node.keywords}
        if "timeout" not in named:
            return Violation(file=path, line_no=node.lineno, kind=MISSING_DEADLINE, line=source)
        if "input" in named:
            return Violation(file=path, line_no=node.lineno, kind=UNBOUNDED_PAYLOAD, line=source)
        return None

    def run(self, files: list[Path]) -> list[Violation]:
        """Judge every subprocess call in the scanned files.

        Args:
            files: The files to scan.

        Returns:
            One violation per unbounded or falsely-bounded call.
        """
        out: list[Violation] = []
        for path in files:
            if not self._in_scope(path):
                continue
            tree = parse_source(path)
            imported = self._imported_names(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = self._awaiting_callee(node, imported)
                if callee is not None:
                    found = self._judge(path, node, f"subprocess {callee}(...)")
                    if found is not None:
                        out.append(found)
                    continue
                if not self._is_communicate(node):
                    continue
                # Judged by the same predicate as the rest, then narrowed to
                # the ONE clause communicate is judged on: a deadline paired
                # with a payload. Its other answers are deliberately dropped
                # rather than recomputed here -- a communicate carrying no
                # timeout is the Popen shape whose bound belongs on the wait,
                # which this rule does not reach into, and restating the
                # keyword test for it would be a second copy of _judge that
                # drifts from the first.
                verdict = self._judge(path, node, "communicate(input=..., timeout=...)")
                if verdict is not None and verdict.kind == UNBOUNDED_PAYLOAD:
                    out.append(verdict)
        return out


__all__ = [
    "AWAITING_CALLS",
    "MISSING_DEADLINE",
    "UNBOUNDED_PAYLOAD",
    "UNPROVEN_DEADLINE",
    "SubprocessTimeoutRule",
]
