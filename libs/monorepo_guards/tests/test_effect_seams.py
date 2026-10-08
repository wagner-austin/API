"""Tests for finding a package's effect seams."""

from __future__ import annotations

from pathlib import Path

from monorepo_guards.effect_seams import (
    effect_seams,
    exposed_names,
    index_package,
    is_hooks_module,
    module_name_for,
    module_seams,
    resolve_function,
)
from tests._effect_support import files_of, write


class TestModuleNames:
    def test_src_scripts_and_tests_are_named_as_they_are_imported(self, tmp_path: Path) -> None:
        assert module_name_for(tmp_path / "src/pkg/core/_test_hooks.py", tmp_path) == (
            "pkg.core._test_hooks"
        )
        assert module_name_for(tmp_path / "src/pkg/__init__.py", tmp_path) == "pkg"
        assert module_name_for(tmp_path / "scripts/amex/_test_hooks.py", tmp_path) == (
            "scripts.amex._test_hooks"
        )
        assert module_name_for(tmp_path / "tests/test_x.py", tmp_path) is None

    def test_any_component_starting_test_hooks_marks_a_hooks_module(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/_test_hooks/cdp.py", "x = 1\n")
        write(tmp_path, "src/pkg/stream/_test_hooks_model.py", "x = 1\n")
        write(tmp_path, "src/pkg/stream/model.py", "x = 1\n")
        write(tmp_path, "tests/test_x.py", "x = 1\n")
        index = index_package(files_of(tmp_path), tmp_path)
        assert sorted(index) == [
            "pkg._test_hooks.cdp",
            "pkg.stream._test_hooks_model",
            "pkg.stream.model",
        ]
        flagged = sorted(name for name, module in index.items() if is_hooks_module(module))
        assert flagged == ["pkg._test_hooks.cdp", "pkg.stream._test_hooks_model"]


class TestResolveFunction:
    def test_local_reexported_missing_and_cyclic_names(self, tmp_path: Path) -> None:
        write(tmp_path, "src/pkg/__init__.py", "from pkg.run import go\nfrom pkg import loop\n")
        write(tmp_path, "src/pkg/run.py", "def go():\n    pass\n")
        write(tmp_path, "src/pkg/other.py", "from os import getcwd\nfrom pkg import go, loop\n")
        index = index_package(files_of(tmp_path), tmp_path)
        other = index["pkg.other"]
        found = resolve_function(index, other, "go", set())
        named = None if found is None else (found[0].name, found[1].name)
        assert named == ("pkg.run", "go")
        assert resolve_function(index, other, "getcwd", set()) is None
        assert resolve_function(index, other, "unbound", set()) is None
        assert resolve_function(index, other, "loop", set()) is None


class TestEffectSeams:
    def test_a_def_seam_reaches_its_primitive_through_a_delegate(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "src/pkg/_command.py",
            """
            import subprocess

            def awaited(argv):
                note(argv)
                return subprocess.run(argv, timeout=5)

            def note(argv):
                return note(argv)
            """,
        )
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            from pkg._command import awaited

            def _default_run(argv):
                helper()
                obj.method()
                return awaited(argv)

            def helper():
                return 1

            def _default_now():
                return helper()

            run: RunProto = _default_run
            """,
        )
        index = index_package(files_of(tmp_path), tmp_path)
        found = effect_seams(index)
        assert [(e.seam.label, e.reach.chain, e.reach.primitive.kind) for e in found] == [
            ("_default_run", ("awaited", "subprocess.run"), "process"),
        ]

    def test_a_cycle_back_to_the_seam_is_walked_once(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            def ping():
                return pong()

            def pong():
                return ping()
            """,
        )
        assert effect_seams(index_package(files_of(tmp_path), tmp_path)) == []

    def test_a_dynamic_import_alias_called_is_a_service_seam(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            def _connect_impl(dsn):
                psycopg = __import__("psycopg")
                connector: Connector = psycopg.connect
                return connector(dsn)

            def reset_hooks():
                global run_process
                run_process = subprocess.run
            """,
        )
        found = effect_seams(index_package(files_of(tmp_path), tmp_path))
        assert [(e.seam.label, e.reach.chain) for e in found] == [
            ("_connect_impl", ("psycopg.connect",)),
        ]

    def test_hooks_bound_straight_to_primitives_and_imported_functions(
        self, tmp_path: Path
    ) -> None:
        write(
            tmp_path,
            "src/pkg/net.py",
            "import socket\n\ndef dial(h):\n    return socket.create_connection(h)\n",
        )
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            import subprocess
            from shutil import rmtree
            from pkg.net import dial
            from os.path import join

            def _local():
                return 1

            run_process: RunProto = subprocess.run
            remove_tree = rmtree
            first = second = rmtree
            dialer: DialProto = dial
            joiner: JoinProto = join
            local: LocalProto = _local
            count: int
            limit: int = 3
            settings.value = rmtree
            spawn = lambda argv: subprocess.Popen(argv)
            idle = lambda: 1
            """,
        )
        index = index_package(files_of(tmp_path), tmp_path)
        found = effect_seams(index)
        assert [(e.seam.label, e.reach.chain) for e in found] == [
            ("run_process", ("subprocess.run",)),
            ("remove_tree", ("shutil.rmtree",)),
            ("dialer", ("socket.create_connection",)),
            ("spawn", ("subprocess.Popen",)),
        ]
        assert found[3].seam.names == frozenset()
        assert found[3].seam.fields == frozenset()
        dialer = found[2].seam
        assert dialer.names == frozenset({"dial"})
        assert dialer.owner.name == "pkg.net"

    def test_bundled_functions_lambdas_and_primitives(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            import subprocess
            import os

            def _real_run(argv):
                return subprocess.run(argv)

            def default_hooks():
                return Hooks(
                    run=_real_run,
                    kill=lambda pid: os.kill(pid, 9),
                    spawn=subprocess.Popen,
                    tick=lambda: 1,
                    **extra,
                )

            def table():
                return {"run": _real_run, "rename": os.replace, 3: _real_run, **more}

            MODULE_BUNDLE = Hooks(run=_real_run, swap=lambda a, b: os.replace(a, b))
            """,
        )
        index = index_package(files_of(tmp_path), tmp_path)
        seams = {seam.label: seam for seam in module_seams(index, index["pkg._test_hooks"])}
        assert seams["_real_run"].fields == frozenset({("default_hooks", "run"), ("table", "run")})
        assert seams["default_hooks.kill"].fields == frozenset({("default_hooks", "kill")})
        assert seams["default_hooks.spawn"].fields == frozenset({("default_hooks", "spawn")})
        assert seams["table.rename"].fields == frozenset({("table", "rename")})
        assert seams["swap"].fields == frozenset()
        labels = sorted(e.seam.label for e in effect_seams(index))
        assert labels == [
            "_real_run",
            "default_hooks.kill",
            "default_hooks.spawn",
            "swap",
            "table.rename",
        ]

    def test_factory_built_hooks_reach_through_the_factory_or_what_it_is_handed(
        self, tmp_path: Path
    ) -> None:
        write(
            tmp_path,
            "src/pkg/spawn.py",
            """
            import subprocess

            def spawn_and_collect(argv):
                return subprocess.run(argv)

            def make_run_git(spawn, root):
                return lambda args: spawn(["git", "-C", root, *args])

            def make_dialer(host):
                import socket
                return socket.create_connection((host, 1))

            def make_clock():
                return 1
            """,
        )
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            import shutil
            from pkg import spawn
            from pkg.spawn import make_clock, make_run_git, spawn_and_collect

            def default_hooks():
                return Hooks(
                    run_git=make_run_git(spawn_and_collect, "."),
                    remove=make_run_git(shutil.rmtree, "."),
                    dial=spawn.make_dialer("hub"),
                    clock=make_clock(),
                    again=make_run_git(make_run_git, ".", os.sep),
                )

            run_git = make_run_git(spawn_and_collect, ".")
            clock = make_clock()
            """,
        )
        found = effect_seams(index_package(files_of(tmp_path), tmp_path))
        assert sorted((e.seam.label, e.reach.chain) for e in found) == [
            ("default_hooks.dial", ("make_dialer", "socket.create_connection")),
            ("default_hooks.remove", ("shutil.rmtree",)),
            ("default_hooks.run_git", ("spawn_and_collect", "subprocess.run")),
            ("run_git", ("spawn_and_collect", "subprocess.run")),
        ]

    def test_modules_that_are_not_hooks_modules_declare_no_seams(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "src/pkg/runner.py",
            "import subprocess\n\ndef go():\n    subprocess.run([])\n",
        )
        assert effect_seams(index_package(files_of(tmp_path), tmp_path)) == []


class TestExposedNames:
    def test_each_seam_is_exposed_by_its_name_bindings_and_fields(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "src/pkg/_test_hooks.py",
            """
            import subprocess

            def _default_run(argv):
                return subprocess.run(argv)

            def default_hooks():
                return Hooks(runner=_default_run, kill=lambda p: subprocess.call(p))

            run: RunProto = _default_run
            also = _default_run
            raw = subprocess.Popen
            now: NowProto = clock
            import_marker = 1
            """,
        )
        exposed = exposed_names(effect_seams(index_package(files_of(tmp_path), tmp_path)))
        assert exposed == frozenset({"_default_run", "run", "also", "runner", "kill", "raw"})
