"""Finding each effect seam's real twins: tests that run the real thing failing.

The second half of ``effect-seam-twin``. A fake stands in for a seam in most
tests, which the coding standard sanctions; what it may not do is stand
ALONE. Some test must call the seam's real implementation, and at least one
of those calls must take it through a failure, because the success path is
the one a fake describes correctly by construction.

THE DEFINITION IS SHARED WITH MCPs (board task c96e8791):

* A REAL TEST of a seam is a test function under ``tests/`` that calls or
  references the implementation itself, not the rebindable hook: by its
  name imported from the hooks module, through the module object
  (``_test_hooks.fn``), or as ``<factory>().<field>`` (also
  ``h = factory()`` then ``h.field``) for a function bound into a hooks
  bundle. A reference counts because the common pattern binds the real
  implementation into a fake bundle and runs the code under test. Reaching
  it through a same-file helper function, or a pytest fixture the test
  requests (from its own file or a ``conftest.py``), counts.
* What it EXERCISES is the set of failure kinds the test, or a same-file
  helper it calls, names (:mod:`monorepo_guards.effect_failures`); the
  rule asks for one of the seam's own kind.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

from monorepo_guards.effect_failures import failure_kinds, terminal_name
from monorepo_guards.effect_primitives import import_bindings, qualified_name
from monorepo_guards.effect_seams import EffectSeam, PackageModule, resolve_function
from monorepo_guards.util import parse_source


class RealTest(NamedTuple):
    """One test that runs a seam's real implementation.

    Attributes:
        label: ``<file>::<Class>.<test>`` relative to the package root.
        kinds: The failure kinds it exercises, empty when it exercises none.
    """

    label: str
    kinds: frozenset[str]


class ParsedTestFile(NamedTuple):
    """A parsed test file and what it binds.

    Attributes:
        path: The file.
        tree: Its AST.
        bindings: Its imports.
        functions: Its top-level functions by name.
    """

    path: Path
    tree: ast.Module
    bindings: dict[str, str]
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]


def _is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a function is a pytest fixture.

    Args:
        node: Function.

    Returns:
        True when a decorator is ``fixture`` or ``pytest.fixture``, called
        or bare.
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if terminal_name(target) == "fixture":
            return True
    return False


def _load_tests(files: list[Path], root: Path) -> list[ParsedTestFile]:
    """Parse every file under the package's ``tests``.

    Args:
        files: The files the guard run collected.
        root: The package root.

    Returns:
        The test modules.
    """
    tests_root = root / "tests"
    loaded: list[ParsedTestFile] = []
    for path in files:
        if not path.is_relative_to(tests_root):
            continue
        tree = parse_source(path)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        loaded.append(ParsedTestFile(path, tree, import_bindings(tree, "tests", False), functions))
    return loaded


def _seam_key(index: dict[str, PackageModule], qualified: str | None) -> tuple[str, str] | None:
    """Resolve a called dotted name to the package function it is.

    Args:
        index: The package's modules.
        qualified: The call's qualified name.

    Returns:
        ``(defining module, function)``, or None when the name is not a
        function of the package.
    """
    if qualified is None:
        return None
    module_name, _, attr = qualified.rpartition(".")
    module = index.get(module_name)
    if module is None:
        return None
    found = resolve_function(index, module, attr, set())
    if found is None:
        return None
    return (found[0].name, found[1].name)


def _direct_calls(
    index: dict[str, PackageModule],
    module: ParsedTestFile,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    references: bool,
) -> set[tuple[str, str, str]]:
    """Collect the package functions a test-side function calls or references.

    A reference counts as well as a call in the test and its same-file
    helpers: the common pattern is
    ``set_hooks(replace(fake, remove_tree=real.remove_tree))`` and then
    running the code under test, which runs the real implementation without
    the test ever calling it by name. In a fixture only a call counts:
    tools/maketools' ``world`` fixture restores every real hook at teardown
    (``_test_hooks.kill = _test_hooks._default_kill``), so counting its
    references made every test that requests it a real test of five seams
    while it ran only fakes.

    Args:
        index: The package's modules.
        module: The test module holding it.
        function: The function.
        references: Whether a bare reference counts, or only a call.

    Returns:
        ``(module, function, "")`` for a function named directly or through
        a module object, ``(module, factory, field)`` for a bundle field
        read off a factory's result, and ``(module, factory, "*")`` for a
        whole bundle handed over as an argument
        (``set_hooks(default_hooks())``), which is a reference and so
        counts only where references do.
    """
    factories = _bound_factories(index, module, function)
    called = {id(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
    found = _handed_bundles(index, module, function) if references else set()
    for node in ast.walk(function):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        if not references and id(node) not in called:
            continue
        key = _seam_key(index, qualified_name(node, module.bindings))
        if key is not None:
            found.add((key[0], key[1], ""))
        if isinstance(node, ast.Attribute):
            made = _factory_of(index, module, node.value, factories)
            if made is not None:
                found.add((made[0], made[1], node.attr))
    return found


def _bound_factories(
    index: dict[str, PackageModule],
    module: ParsedTestFile,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, tuple[str, str]]:
    """Map each local a function binds to a package function's result.

    Args:
        index: The package's modules.
        module: The test module.
        function: The function.

    Returns:
        Local name to the ``(module, factory)`` whose result it holds, as in
        ``real = default_hooks()``.
    """
    factories: dict[str, tuple[str, str]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            key = _seam_key(index, qualified_name(node.value.func, module.bindings))
            if key is not None:
                factories.update((t.id, key) for t in node.targets if isinstance(t, ast.Name))
    return factories


def _handed_bundles(
    index: dict[str, PackageModule],
    module: ParsedTestFile,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[tuple[str, str, str]]:
    """Collect the bundles a function hands over whole as an argument.

    Args:
        index: The package's modules.
        module: The test module.
        function: The function.

    Returns:
        ``(module, factory, "*")`` for each ``f(factory())`` or
        ``f(x=factory())`` whose factory is a package function.
    """
    handed: set[tuple[str, str, str]] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            if isinstance(argument, ast.Call):
                made = _seam_key(index, qualified_name(argument.func, module.bindings))
                if made is not None:
                    handed.add((made[0], made[1], "*"))
    return handed


def _factory_of(
    index: dict[str, PackageModule],
    module: ParsedTestFile,
    receiver: ast.expr,
    factories: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Name the factory whose result an attribute is read from.

    Args:
        index: The package's modules.
        module: The test module.
        receiver: What the attribute is read off.
        factories: The function's :func:`_bound_factories`.

    Returns:
        ``(module, factory)`` for ``factory().field`` or ``real.field``
        with ``real = factory()``, else None.
    """
    if isinstance(receiver, ast.Call):
        return _seam_key(index, qualified_name(receiver.func, module.bindings))
    if isinstance(receiver, ast.Name):
        return factories.get(receiver.id)
    return None


def _helpers_reached(
    module: ParsedTestFile, function: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """List a function and every same-file helper it calls, transitively.

    Args:
        module: The test module.
        function: The starting function.

    Returns:
        The function first, then each helper once.
    """
    reached = [function]
    names = {function.name}
    position = 0
    while position < len(reached):
        for node in ast.walk(reached[position]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                helper = module.functions.get(node.func.id)
                if helper is not None and helper.name not in names:
                    names.add(helper.name)
                    reached.append(helper)
        position += 1
    return reached


def _depth(module: ParsedTestFile) -> int:
    """Key a test module by how deep it sits.

    Args:
        module: The test module.

    Returns:
        Its path's component count, so an outer ``conftest.py`` applies
        before an inner one overrides it.
    """
    return len(module.path.parts)


def _fixtures_for(
    test_path: Path, modules: list[ParsedTestFile]
) -> dict[str, tuple[ParsedTestFile, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Map every fixture visible to a test file by name.

    Args:
        test_path: The test file.
        modules: Every test module of the package.

    Returns:
        Fixture name to its module and definition: those of each
        ``conftest.py`` in the file's directory or above, then its own,
        which shadow them as pytest does.
    """
    visible: dict[str, tuple[ParsedTestFile, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    conftests = [
        m
        for m in modules
        if m.path.name == "conftest.py" and test_path.is_relative_to(m.path.parent)
    ]
    for module in sorted(conftests, key=_depth) + [m for m in modules if m.path == test_path]:
        visible.update(
            (name, (module, node)) for name, node in module.functions.items() if _is_fixture(node)
        )
    return visible


def _test_functions(
    module: ParsedTestFile,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """List a module's test functions with their pytest-style labels.

    Args:
        module: The test module.

    Returns:
        ``(label, node)`` for each ``test*`` function and each ``test*``
        method of a ``Test*`` class.
    """
    found = [(name, node) for name, node in module.functions.items() if name.startswith("test")]
    for node in module.tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            found.extend(
                (f"{node.name}.{child.name}", child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test")
            )
    return found


def _calls_reached(
    index: dict[str, PackageModule],
    module: ParsedTestFile,
    test: ast.FunctionDef | ast.AsyncFunctionDef,
    fixtures: dict[str, tuple[ParsedTestFile, ast.FunctionDef | ast.AsyncFunctionDef]],
) -> set[tuple[str, str, str]]:
    """Collect every package call a test reaches through helpers and fixtures.

    Args:
        index: The package's modules.
        module: The test's module.
        test: The test.
        fixtures: The fixtures visible to it.

    Returns:
        The calls, as :func:`_direct_calls` spells them: references too
        from the test and its own helpers, calls only from fixtures.
    """
    calls: set[tuple[str, str, str]] = set()
    pending = [(module, test)]
    seen: set[tuple[Path, str]] = set()
    while pending:
        holder, function = pending.pop()
        if (holder.path, function.name) in seen:
            continue
        seen.add((holder.path, function.name))
        for reached in _helpers_reached(holder, function):
            calls |= _direct_calls(index, holder, reached, references=function is test)
            pending.extend(fixtures[arg.arg] for arg in reached.args.args if arg.arg in fixtures)
    return calls


def real_tests(
    files: list[Path],
    root: Path,
    index: dict[str, PackageModule],
    seams: list[EffectSeam],
) -> dict[tuple[str, str], list[RealTest]]:
    """Find every real test of every effect seam.

    Args:
        files: The files the guard run collected.
        root: The package root.
        index: The package's modules.
        seams: Its effect seams.

    Returns:
        ``(hooks module, seam label)`` to its real tests, empty for a seam
        no test runs.
    """
    wanted: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    found: dict[tuple[str, str], list[RealTest]] = {}
    for effect in seams:
        seam = effect.seam
        key = (seam.module.name, seam.label)
        found[key] = []
        impl = seam.impl
        if isinstance(impl, (ast.FunctionDef, ast.AsyncFunctionDef)):
            wanted.setdefault((seam.owner.name, impl.name, ""), []).append(key)
        for factory, field in seam.fields:
            wanted.setdefault((seam.module.name, factory, field), []).append(key)
            wanted.setdefault((seam.module.name, factory, "*"), []).append(key)
    modules = _load_tests(files, root)
    for module in modules:
        fixtures = _fixtures_for(module.path, modules)
        relative = module.path.relative_to(root).as_posix()
        for label, test in _test_functions(module):
            hits = _calls_reached(index, module, test, fixtures) & wanted.keys()
            if not hits:
                continue
            kinds: frozenset[str] = frozenset().union(
                *(failure_kinds(node) for node in _helpers_reached(module, test))
            )
            for key in sorted({key for hit in hits for key in wanted[hit]}):
                found[key].append(RealTest(label=f"{relative}::{label}", kinds=kinds))
    return found


__all__ = [
    "ParsedTestFile",
    "RealTest",
    "real_tests",
]
