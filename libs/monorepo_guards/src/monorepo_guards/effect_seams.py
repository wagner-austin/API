"""Finding a package's effect seams: the hooks whose real code acts on the world.

A seam is what a test may rebind to a fake. The coding standard makes that
the sanctioned way to test without mocks ("tests set them to fakes"), and on
2026-10-07 it was also how a defect reached the operator: the harness gate's
whole suite reached the installer through a fake, the real hook was run once
on a command that succeeded, and nothing ever ran the real installer failing.
So the first half of ``effect-seam-twin`` is knowing which seams have a real
implementation that does something a fake can misdescribe.

THE DEFINITION IS SHARED WITH MCPs (board task c96e8791), word for word:

* A seam is a module-level function of a hooks module (``def`` or
  ``name: Proto = fn``), or a function bound into a hooks bundle there
  (keyword ``field=fn`` or ``field=lambda``, dict entry ``"field": fn``).
  A hooks module is any file whose path has a component starting
  ``_test_hooks``: ``_test_hooks.py``, ``_test_hooks/cdp.py``,
  ``_test_hooks_repositories.py``.
* It is an EFFECT seam when its body reaches a primitive
  (:mod:`monorepo_guards.effect_primitives`), following bare-name calls to
  functions of the same module and to functions imported with
  ``from <module> import fn`` from the same package's ``src`` or
  ``scripts``, transitively. Calls through an attribute, and calls into
  another package, are not followed.

The delegation clause exists because of a measured case: the fleet's
``run`` hook, every ssh the hub's runners make, reaches ``subprocess``
only through ``fleet.core._command._awaited``, and a same-module-only walk
reported it as pure.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

from monorepo_guards.effect_primitives import (
    PrimitiveCall,
    bound_primitive,
    calls_in_order,
    import_bindings,
    local_aliases,
    primitive_of,
    qualified_name,
)
from monorepo_guards.util import module_nodes, parse_source

HOOKS_PREFIX = "_test_hooks"


class PackageModule(NamedTuple):
    """One importable module of the package under check.

    Attributes:
        name: Its dotted import name (``fleet.core._test_hooks``, or
            ``scripts.amex._test_hooks`` for a script).
        path: The file.
        tree: Its parsed AST.
        bindings: Its :func:`import_bindings`.
    """

    name: str
    path: Path
    tree: ast.Module
    bindings: dict[str, str]


class Seam(NamedTuple):
    """One seam of a hooks module.

    Attributes:
        module: The hooks module declaring it.
        label: How the lint line names it: the function's name, the
            binding's name, or ``<factory>.<field>`` for a lambda.
        line_no: Where it is declared.
        names: Names a test may call the implementation by, imported from
            the hooks module or through it (``hooks.fn(...)``).
        fields: ``(factory, field)`` pairs a test may call it through, as
            ``factory().field(...)``.
        owner: The module whose context the implementation is read in,
            which differs from ``module`` for an imported function.
        impl: The implementation's node: a function, a lambda, or the
            primitive itself for a hook bound straight to one
            (``run_process: P = subprocess.run``).
    """

    module: PackageModule
    label: str
    line_no: int
    names: frozenset[str]
    fields: frozenset[tuple[str, str]]
    owner: PackageModule
    impl: ast.FunctionDef | ast.AsyncFunctionDef | ast.expr


class Reach(NamedTuple):
    """How a seam reaches its first primitive.

    Attributes:
        chain: Each helper followed, then the primitive's qualified name,
            in call order (the seam itself is not repeated).
        primitive: The primitive call reached.
    """

    chain: tuple[str, ...]
    primitive: PrimitiveCall


def module_name_for(path: Path, root: Path) -> str | None:
    """Name the module a file is imported as.

    Args:
        path: A Python file under ``root``.
        root: The package root (the directory holding ``src``).

    Returns:
        The dotted name, relative to ``src`` for library code and to the
        root for ``scripts`` (which tests import as ``scripts.x``), or None
        for a file under neither, such as a test.
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    elif parts[0] != "scripts":
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_hooks_module(module: PackageModule) -> bool:
    """Report whether a module is a hooks module.

    Args:
        module: The module.

    Returns:
        True when any component of its dotted name starts ``_test_hooks``.
    """
    return any(part.startswith(HOOKS_PREFIX) for part in module.name.split("."))


def index_package(files: list[Path], root: Path) -> dict[str, PackageModule]:
    """Index every ``src`` and ``scripts`` module of a package by name.

    Args:
        files: The files the guard run collected.
        root: The package root.

    Returns:
        Dotted name to module.
    """
    index: dict[str, PackageModule] = {}
    for path in files:
        name = module_name_for(path, root)
        if name is None:
            continue
        tree = parse_source(path)
        bindings = import_bindings(tree, name, path.name == "__init__.py")
        index[name] = PackageModule(name=name, path=path, tree=tree, bindings=bindings)
    return index


def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map a module's own top-level functions by name.

    Args:
        tree: Parsed module.

    Returns:
        Name to definition.
    """
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def resolve_function(
    index: dict[str, PackageModule],
    module: PackageModule,
    name: str,
    seen: set[tuple[str, str]],
) -> tuple[PackageModule, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Find the first-party function a bare name in a module refers to.

    Args:
        index: The package's modules.
        module: The module the name appears in.
        name: The bare name.
        seen: ``(module, name)`` pairs already tried, so a re-export cycle
            ends.

    Returns:
        The defining module and the definition, or None when the name is
        not a function of this package.
    """
    key = (module.name, name)
    if key in seen:
        return None
    seen.add(key)
    local = top_level_functions(module.tree).get(name)
    if local is not None:
        return (module, local)
    qualified = module.bindings.get(name)
    if qualified is None:
        return None
    source_name, _, attr = qualified.rpartition(".")
    source = index.get(source_name)
    if source is None:
        return None
    return resolve_function(index, source, attr, seen)


def package_function(
    index: dict[str, PackageModule], module: PackageModule, node: ast.expr
) -> tuple[PackageModule, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Find the package function an expression names, when it names one.

    Args:
        index: The package's modules.
        module: The module the expression appears in.
        node: A bare name, or ``module.fn`` on an import of a package
            module.

    Returns:
        The defining module and function, or None for anything else, an
        ``obj.method`` included.
    """
    if isinstance(node, ast.Name):
        return resolve_function(index, module, node.id, set())
    qualified = qualified_name(node, module.bindings)
    if qualified is None:
        return None
    source_name, _, attr = qualified.rpartition(".")
    source = index.get(source_name)
    if source is None:
        return None
    return resolve_function(index, source, attr, set())


def _factory_built(index: dict[str, PackageModule], module: PackageModule, value: ast.expr) -> bool:
    """Report whether a bound value is a package function's result.

    Args:
        index: The package's modules.
        module: The hooks module.
        value: The bound value, e.g. ``make_run_git(spawn, root)``.

    Returns:
        True for a call of a package function.
    """
    return isinstance(value, ast.Call) and package_function(index, module, value.func) is not None


def _reach_bound(
    index: dict[str, PackageModule],
    owner: PackageModule,
    bound: ast.expr,
    visited: set[tuple[str, str]],
) -> Reach | None:
    """Follow a hook bound to a value, not a function, to its effect.

    A value bound straight to a primitive (``subprocess.run``,
    ``x.bind(...)``) is that primitive. A factory-built one
    (``make_run_git(spawn_and_collect, root)``) reaches through the
    factory's own walk first and, failing that, the first argument it is
    handed that is a primitive or a package function reaching one, since a
    factory acts through what it is given (corvis-stick, measured for MCPs
    board task c96e8791).

    Args:
        index: The package's modules.
        owner: The hooks module.
        bound: The bound value.
        visited: ``(module, function)`` pairs already walked.

    Returns:
        The chain from the value, the factory or the argument, or None.
    """
    handed: list[ast.expr] = [bound]
    if isinstance(bound, ast.Call):
        handed.extend([bound.func, *bound.args, *(keyword.value for keyword in bound.keywords)])
    for argument in handed:
        primitive = bound_primitive(argument, owner.bindings)
        if primitive is not None:
            return Reach(chain=(primitive.called,), primitive=primitive)
        target = package_function(index, owner, argument)
        if target is None:
            continue
        key = (target[0].name, target[1].name)
        if key in visited:
            continue
        visited.add(key)
        deeper = reach_primitive(index, target[0], target[1], visited)
        if deeper is not None:
            return Reach(chain=(target[1].name, *deeper.chain), primitive=deeper.primitive)
    return None


def reach_primitive(
    index: dict[str, PackageModule],
    owner: PackageModule,
    impl: ast.FunctionDef | ast.AsyncFunctionDef | ast.expr,
    visited: set[tuple[str, str]],
) -> Reach | None:
    """Follow an implementation to the first primitive it reaches.

    Args:
        index: The package's modules.
        owner: The module the implementation is read in.
        impl: The function, the lambda, or a bound value.
        visited: ``(module, function)`` pairs already walked.

    Returns:
        The chain to the first primitive in source order, or None when the
        implementation reaches none.
    """
    if not isinstance(impl, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return _reach_bound(index, owner, impl, visited)
    aliases = local_aliases(impl, owner.bindings)
    for call in calls_in_order(impl):
        primitive = primitive_of(call, owner.bindings, aliases)
        if primitive is not None:
            return Reach(chain=(primitive.called,), primitive=primitive)
        if not isinstance(call.func, ast.Name):
            continue
        target = resolve_function(index, owner, call.func.id, set())
        if target is None:
            continue
        target_module, target_node = target
        key = (target_module.name, target_node.name)
        if key in visited:
            continue
        visited.add(key)
        deeper = reach_primitive(index, target_module, target_node, visited)
        if deeper is not None:
            return Reach(chain=(target_node.name, *deeper.chain), primitive=deeper.primitive)
    return None


def _enclosing_factories(tree: ast.Module) -> dict[int, str]:
    """Map every node inside a top-level function to that function's name.

    Args:
        tree: Parsed module.

    Returns:
        ``id(node)`` to the name of the top-level function holding it.
    """
    owners: dict[int, str] = {}
    for function in top_level_functions(tree).values():
        for child in ast.walk(function):
            owners[id(child)] = function.name
    return owners


def _bundle_entries(tree: ast.Module) -> list[tuple[int, str, ast.expr]]:
    """List every ``field=value`` keyword and ``"field": value`` entry.

    Args:
        tree: Parsed module.

    Returns:
        ``(id of the holding call or dict, field, value)`` in walk order.
    """
    entries: list[tuple[int, str, ast.expr]] = []
    for node in module_nodes(tree):
        if isinstance(node, ast.Call):
            entries.extend(
                (id(node), keyword.arg, keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            )
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    entries.append((id(node), key.value, value))
    return entries


def module_seams(index: dict[str, PackageModule], module: PackageModule) -> list[Seam]:
    """List every seam a hooks module declares.

    Args:
        index: The package's modules.
        module: A hooks module.

    Returns:
        Its seams, in declaration order: functions first, then hooks bound
        to an imported function or straight to a primitive, then bundled
        lambdas and primitives. A module-level hook bound straight to a
        primitive has no name a test can reach it by except the rebindable
        hook, so no test form matches it and it is always reported: the fix
        is a named implementation.
    """
    functions = top_level_functions(module.tree)
    fields: dict[str, set[tuple[str, str]]] = {name: set() for name in functions}
    bundled: list[Seam] = []
    owners = _enclosing_factories(module.tree)
    for holder, field, value in _bundle_entries(module.tree):
        factory = owners.get(holder, "")
        if isinstance(value, ast.Name) and value.id in functions and factory:
            fields[value.id].add((factory, field))
        elif (
            isinstance(value, ast.Lambda)
            or bound_primitive(value, module.bindings) is not None
            or _factory_built(index, module, value)
        ):
            pairs = frozenset({(factory, field)}) if factory else frozenset()
            label = f"{factory}.{field}" if factory else field
            bundled.append(Seam(module, label, value.lineno, frozenset(), pairs, module, value))
    seams = [
        Seam(module, name, node.lineno, frozenset({name}), frozenset(fields[name]), module, node)
        for name, node in functions.items()
    ]
    seams.extend(_bound_seams(index, module, frozenset(functions)))
    return [*seams, *bundled]


def _bound_seams(
    index: dict[str, PackageModule], module: PackageModule, functions: frozenset[str]
) -> list[Seam]:
    """List the module-level hooks bound to an imported function or a primitive.

    Args:
        index: The package's modules.
        module: A hooks module.
        functions: Its own top-level function names, already seams.

    Returns:
        ``name: P = imported_fn`` (resolved into the package) and
        ``name = <primitive>``, ``name: P = <primitive>``, a lambda or a
        package function's product bound the same way. None of these has a
        twin form: only a bundle field gets ``factory().field``.
    """
    seams: list[Seam] = []
    for node in module.tree.body:
        target: ast.expr
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if (
            isinstance(value, ast.Lambda)
            or bound_primitive(value, module.bindings) is not None
            or _factory_built(index, module, value)
        ):
            seams.append(
                Seam(module, target.id, node.lineno, frozenset(), frozenset(), module, value)
            )
            continue
        if not isinstance(node, ast.AnnAssign) or not isinstance(value, ast.Name):
            continue
        if value.id in functions:
            continue
        resolved = resolve_function(index, module, value.id, set())
        if resolved is not None:
            seams.append(
                Seam(
                    module,
                    target.id,
                    node.lineno,
                    frozenset({value.id}),
                    frozenset(),
                    resolved[0],
                    resolved[1],
                )
            )
    return seams


class EffectSeam(NamedTuple):
    """A seam whose implementation reaches a primitive.

    Attributes:
        seam: The seam.
        reach: How it reaches its primitive.
    """

    seam: Seam
    reach: Reach


def effect_seams(index: dict[str, PackageModule]) -> list[EffectSeam]:
    """List every effect seam of a package.

    Args:
        index: The package's modules.

    Returns:
        Each seam whose implementation reaches a primitive, with its chain,
        ordered by hooks module and then declaration.
    """
    found: list[EffectSeam] = []
    for name in sorted(index):
        module = index[name]
        if not is_hooks_module(module):
            continue
        for seam in module_seams(index, module):
            start = (seam.owner.name, seam.label)
            reach = reach_primitive(index, seam.owner, seam.impl, {start})
            if reach is not None:
                found.append(EffectSeam(seam=seam, reach=reach))
    return found


def _bound_to(tree: ast.Module, name: str) -> list[str]:
    """List the module attributes a module binds to one of its names.

    Args:
        tree: The hooks module.
        name: A function's name, e.g. ``_default_run``.

    Returns:
        Every top-level ``x: Proto = name`` or ``x = name`` target, e.g.
        ``run``.
    """
    bound: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        if isinstance(value, ast.Name) and value.id == name:
            bound.extend(target.id for target in targets if isinstance(target, ast.Name))
    return bound


def exposed_names(effects: list[EffectSeam]) -> frozenset[str]:
    """Name every attribute through which some effect seam is reached.

    ``state-change-verified`` counts a call through a hooks object as an
    effect only when it names one of these: ``hooks.run(...)`` changes
    something, ``hooks.read_text(...)`` and ``hooks.now()`` do not.

    Args:
        effects: The package's effect seams.

    Returns:
        Each seam's own name, every module attribute bound to it and every
        bundle field it is bound into.
    """
    names: set[str] = set()
    for effect in effects:
        seam = effect.seam
        names.add(seam.label.rpartition(".")[2])
        names.update(_bound_to(seam.module.tree, seam.label))
        names.update(field for _, field in seam.fields)
    return frozenset(names)


__all__ = [
    "HOOKS_PREFIX",
    "EffectSeam",
    "PackageModule",
    "Reach",
    "Seam",
    "effect_seams",
    "exposed_names",
    "index_package",
    "is_hooks_module",
    "module_name_for",
    "module_seams",
    "reach_primitive",
    "resolve_function",
    "top_level_functions",
]
