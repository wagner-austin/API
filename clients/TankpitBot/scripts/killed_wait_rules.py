"""Guard rule: a test waits for a process it killed with no timeout.

Once ``kill()`` has returned, the process WILL end; how long the end takes
is the host's business, not the test's. On Windows ``kill`` is
``TerminateProcess``, which returns before the process has run down, and a
venv's ``python.exe`` is a launcher whose real interpreter is a second
process torn down with it. Measured on austinpc on 2026-09-29 with sixteen
workers each reaping a half-second-old ``sys.executable`` sleeper beside
sixteen busy processes: 48 of 1,601 waits after ``kill`` took over a second
and the slowest 4.83 s through the venv launcher, against 16 of 1,706 and
2.48 s through the base interpreter. Under a full 16-worker ``make check``
with other sessions on the host, ``process.wait(10.0)`` in
``tests/stream/test_capture.py``'s reap fixture exceeded its bound and failed
a passing test at teardown (board task 06fc3195).

A timeout there is a bet on the host's load, and losing it reports a defect
that does not exist. An unbounded ``wait()`` is deterministic: it returns
when the kill has taken effect. Inside a test function pytest-timeout's
bound still catches a real hang; ``timeout_func_only`` leaves fixture
teardown unbounded on purpose, for the reason ``pyproject.toml`` records.

The rule reads each function in ``tests/`` and flags a ``R.wait(...)`` that
passes a timeout, positionally or by keyword, on a line after ``R.kill()``
in the same function, where ``R`` is the same receiver expression. A wait on
a process nothing killed is not this rule's business, and production code is
out of scope: ``stream/capture.py`` bounds its post-kill wait on purpose, so
a helper stuck in uninterruptible sleep fails the stop loudly.

Runs as part of ``make check`` through
``tests/scripts/test_killed_wait_rules.py``, which holds the real tests
tree to it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _method_receiver(call: ast.Call, method: str) -> str | None:
    """Return the receiver's source when ``call`` is ``<receiver>.<method>(...)``.

    Args:
        call: Any call.
        method: The method name to match.

    Returns:
        The receiver expression, unparsed, or None when ``call`` calls
        anything else.
    """
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != method:
        return None
    return ast.unparse(func.value)


def _passes_timeout(call: ast.Call) -> bool:
    """Say whether a ``wait`` call bounds itself.

    Args:
        call: The ``wait`` call.

    Returns:
        True when it passes any positional argument or ``timeout=``.
    """
    return bool(call.args) or any(keyword.arg == "timeout" for keyword in call.keywords)


def _function_violations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[int]:
    """Return the lines of bounded waits on receivers killed earlier.

    Args:
        function: One function definition.

    Returns:
        Line numbers of the offending ``wait`` calls, ascending.
    """
    first_kill: dict[str, int] = {}
    bounded_waits: list[tuple[str, int]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        killed = _method_receiver(node, "kill")
        if killed is not None:
            first_kill[killed] = min(node.lineno, first_kill.get(killed, node.lineno))
        waited = _method_receiver(node, "wait")
        if waited is not None and _passes_timeout(node):
            bounded_waits.append((waited, node.lineno))
    return sorted(
        line
        for receiver, line in bounded_waits
        if receiver in first_kill and line > first_kill[receiver]
    )


def _module_violations(path: Path) -> list[str]:
    """Return every violation in one test module.

    Args:
        path: The module to read.

    Returns:
        ``<path>:<line>`` for each bounded wait after a kill, once each
        even when nested functions put it inside several definitions.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = {
        line
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for line in _function_violations(node)
    }
    return [f"{path}:{line}" for line in sorted(lines)]


def run_killed_wait_rules(project_root: Path) -> int:
    """Run the killed-wait guard rule over a project's tests.

    Args:
        project_root: Project root containing ``tests``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    tests_root = project_root / "tests"
    if not tests_root.is_dir():
        return 0
    violations: list[str] = []
    for module_path in sorted(tests_root.rglob("*.py")):
        violations.extend(_module_violations(module_path))
    for violation in violations:
        sys.stdout.write(f"killed_wait_violation {violation}\n")
    return len(violations)


__all__ = ["run_killed_wait_rules"]
