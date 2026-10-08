"""Tests for resolving calls to effect primitives through a module's imports."""

from __future__ import annotations

import ast
import textwrap

from monorepo_guards.effect_primitives import (
    PRIMITIVE_GROUPS,
    PRIMITIVE_KINDS,
    PrimitiveCall,
    bound_primitive,
    calls_in_order,
    dotted_name,
    import_bindings,
    local_aliases,
    primitive_of,
    qualified_name,
)


def _module(source: str) -> ast.Module:
    """Parse dedented source.

    Args:
        source: Python source.

    Returns:
        The module.
    """
    return ast.parse(textwrap.dedent(source))


def _expr(source: str) -> ast.expr:
    """Parse one expression.

    Args:
        source: An expression.

    Returns:
        Its node.
    """
    return ast.parse(source, mode="eval").body


def _call(source: str) -> ast.Call:
    """Parse one call expression.

    Args:
        source: A call.

    Returns:
        Its node.

    Raises:
        TypeError: When the source is not a call, which is a broken test.
    """
    node = _expr(source)
    if not isinstance(node, ast.Call):
        raise TypeError(f"not a call: {source}")
    return node


class TestPrimitiveTable:
    def test_every_kind_is_named_and_every_call_is_exact(self) -> None:
        kinds = [kind for kind, _ in PRIMITIVE_GROUPS]
        assert kinds == ["process", "network", "ssh", "file swap", "service"]
        assert PRIMITIVE_KINDS["subprocess.run"] == "process"
        assert PRIMITIVE_KINDS["socket.create_connection"] == "network"
        assert PRIMITIVE_KINDS["paramiko.SSHClient"] == "ssh"
        assert PRIMITIVE_KINDS["os.replace"] == "file swap"
        assert PRIMITIVE_KINDS["psycopg.connect"] == "service"

    def test_constructors_and_local_reads_are_not_primitives(self) -> None:
        for name in (
            "subprocess.CompletedProcess",
            "subprocess.CalledProcessError",
            "aiohttp.web.Response",
            "httpx.Timeout",
            "socket.gethostname",
            "subprocess",
        ):
            assert name not in PRIMITIVE_KINDS


class TestImportBindings:
    def test_plain_dotted_and_aliased_imports(self) -> None:
        tree = _module(
            """
            import subprocess
            import urllib.request
            import os.path as osp
            """
        )
        bindings = import_bindings(tree, "pkg.mod", False)
        assert bindings == {"subprocess": "subprocess", "urllib": "urllib", "osp": "os.path"}

    def test_from_imports_absolute_and_relative(self) -> None:
        tree = _module(
            """
            from shutil import rmtree as rm
            from ._command import awaited
            from .. import sibling
            from . import local
            from ...... import too_far
            """
        )
        bindings = import_bindings(tree, "pkg.core.mod", False)
        assert bindings["rm"] == "shutil.rmtree"
        assert bindings["awaited"] == "pkg.core._command.awaited"
        assert bindings["sibling"] == "pkg.sibling"
        assert bindings["local"] == "pkg.core.local"
        assert bindings["too_far"] == "too_far"

    def test_relative_import_from_a_package_init_starts_at_the_package(self) -> None:
        tree = _module("from .runtime import resolve\n")
        assert import_bindings(tree, "pkg._test_hooks", True) == {
            "resolve": "pkg._test_hooks.runtime.resolve"
        }

    def test_relative_import_in_a_top_level_module(self) -> None:
        tree = _module("from .x import y\n")
        assert import_bindings(tree, "", False) == {"y": "x.y"}

    def test_dynamic_import_binds_the_top_level_module(self) -> None:
        tree = _module(
            """
            def connect(dsn):
                psycopg = __import__("psycopg")
                req = __import__("urllib.request")
                return psycopg, req
            """
        )
        bindings = import_bindings(tree, "pkg.mod", False)
        assert bindings == {"psycopg": "psycopg", "req": "urllib"}

    def test_assignments_that_are_not_one_literal_dynamic_import_bind_nothing(self) -> None:
        tree = _module(
            """
            a = 1
            b = load("x")
            c = obj.__import__("x")
            d = __import__("x", None)
            e = __import__(name)
            f = g = __import__("x")
            h.attr = __import__("x")
            """
        )
        assert import_bindings(tree, "pkg.mod", False) == {}


class TestNames:
    def test_dotted_name_renders_a_chain_and_refuses_other_roots(self) -> None:
        assert dotted_name(_expr("a.b.c")) == "a.b.c"
        assert dotted_name(_expr("a")) == "a"
        assert dotted_name(_expr("f().b")) is None

    def test_qualified_name_resolves_the_root_only(self) -> None:
        bindings = {"sp": "subprocess", "rm": "shutil.rmtree"}
        assert qualified_name(_expr("sp.run"), bindings) == "subprocess.run"
        assert qualified_name(_expr("rm"), bindings) == "shutil.rmtree"
        assert qualified_name(_expr("local.run"), bindings) == "local.run"
        assert qualified_name(_expr("f().run"), bindings) is None


class TestPrimitiveOf:
    def test_a_call_through_an_aliased_import_is_its_primitive(self) -> None:
        bindings = {"sp": "subprocess"}
        assert primitive_of(_call("sp.run(['x'])"), bindings, {}) == PrimitiveCall(
            kind="process", called="subprocess.run"
        )

    def test_a_from_imported_primitive_called_bare(self) -> None:
        bindings = {"rmtree": "shutil.rmtree"}
        found = primitive_of(_call("rmtree(path)"), bindings, {})
        assert found == PrimitiveCall(kind="file swap", called="shutil.rmtree")

    def test_an_unimported_root_is_not_the_module_it_is_spelled_like(self) -> None:
        assert primitive_of(_call("socket.socket()"), {}, {}) is None

    def test_an_imported_module_member_that_is_not_listed(self) -> None:
        bindings = {"subprocess": "subprocess"}
        assert primitive_of(_call("subprocess.CompletedProcess([], 0)"), bindings, {}) is None

    def test_a_call_whose_callee_is_not_a_dotted_chain(self) -> None:
        assert primitive_of(_call("make()()"), {"make": "subprocess.run"}, {}) is None

    def test_a_local_alias_called_is_its_primitive(self) -> None:
        alias = PrimitiveCall(kind="service", called="psycopg.connect")
        assert primitive_of(_call("connector(dsn)"), {}, {"connector": alias}) == alias


class TestBoundPrimitive:
    def test_a_bare_reference_and_a_bound_method(self) -> None:
        bindings = {"subprocess": "subprocess", "socket": "socket"}
        assert bound_primitive(_expr("subprocess.run"), bindings) == PrimitiveCall(
            kind="process", called="subprocess.run"
        )
        assert bound_primitive(_expr("socket.create_connection.bind(x)"), bindings) == (
            PrimitiveCall(kind="network", called="socket.create_connection")
        )

    def test_values_that_are_not_a_primitive(self) -> None:
        bindings = {"subprocess": "subprocess"}
        assert bound_primitive(_expr("subprocess.run(['x'])"), bindings) is None
        assert bound_primitive(_expr("helper.bind(x)"), bindings) is None
        assert bound_primitive(_expr("bind(x)"), bindings) is None
        assert bound_primitive(_expr("subprocess.run.partial(x)"), bindings) is None


class TestLocalAliases:
    def test_annotated_and_plain_aliases_are_collected(self) -> None:
        tree = _module(
            """
            def connect(dsn):
                psycopg = __import__("psycopg")
                connector: Connector = psycopg.connect
                runner = subprocess.run
                count: int
                total = 3
                obj.attr = subprocess.run
                return connector(dsn)
            """
        )
        bindings = import_bindings(tree, "pkg.mod", False) | {"subprocess": "subprocess"}
        aliases = local_aliases(tree.body[0], bindings)
        assert aliases == {
            "connector": PrimitiveCall(kind="service", called="psycopg.connect"),
            "runner": PrimitiveCall(kind="process", called="subprocess.run"),
        }


class TestCallsInOrder:
    def test_calls_are_listed_by_position(self) -> None:
        tree = _module(
            """
            def f():
                second(first())
                third()
            """
        )
        names = [dotted_name(call.func) for call in calls_in_order(tree)]
        assert names == ["second", "first", "third"]
