"""Tests for the ``__all__`` resolution rule.

Two cases carry the rule's whole justification and are worth naming. The first
is the defect it exists for: a name in ``__all__`` that nothing defines, which
ruff and mypy both pass. The second is the false-positive class that made a
ruff setting the wrong mechanism: a SUBMODULE named in ``__all__`` resolves at
runtime via ``_handle_fromlist``, so flagging it would have meant twenty-one
spurious edits across the fleet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.all_exports_rules import AllExportsRule
from monorepo_guards.orchestrator import run_for_project


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_name_nothing_defines_is_flagged(tmp_path: Path) -> None:
    # The motivating defect, reduced: procart_api shipped exactly this.
    path = tmp_path / "mod.py"
    _write(path, '__all__ = ["create_app"]\n')

    violations = AllExportsRule().run([path])

    assert [(v.kind, v.line, v.line_no) for v in violations] == [
        ("all-exports-undefined", "create_app", 1)
    ]


def test_a_submodule_named_in_all_resolves_and_is_not_flagged(tmp_path: Path) -> None:
    # ``from pkg import *`` imports submodules named in __all__ on demand, so
    # this is valid Python. Ruff's F822 flags it; that is why it is not the
    # mechanism here.
    pkg = tmp_path / "pkg"
    _write(pkg / "__init__.py", '__all__ = ["embeds", "types"]\n')
    _write(pkg / "embeds.py", "x = 1\n")
    _write(pkg / "types.py", "y = 2\n")

    assert AllExportsRule().run([pkg / "__init__.py"]) == []


def test_a_subpackage_directory_also_resolves(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write(pkg / "__init__.py", '__all__ = ["routes"]\n')
    _write(pkg / "routes" / "__init__.py", "z = 3\n")

    assert AllExportsRule().run([pkg / "__init__.py"]) == []


def test_a_directory_without_an_init_is_not_a_submodule(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write(pkg / "__init__.py", '__all__ = ["assets"]\n')
    (pkg / "assets").mkdir(parents=True)

    violations = AllExportsRule().run([pkg / "__init__.py"])

    assert [v.line for v in violations] == ["assets"]


def test_the_submodule_escape_applies_only_to_package_inits(tmp_path: Path) -> None:
    # A plain module does not import siblings on star-import, so a sibling
    # file of the same name does NOT make the promise true.
    _write(tmp_path / "helpers.py", "x = 1\n")
    path = tmp_path / "mod.py"
    _write(path, '__all__ = ["helpers"]\n')

    violations = AllExportsRule().run([path])

    assert [v.line for v in violations] == ["helpers"]


@pytest.mark.parametrize(
    "code",
    [
        "import os\n__all__ = ['os']\n",
        "import os.path as osp\n__all__ = ['osp']\n",
        "from collections import abc\n__all__ = ['abc']\n",
        "from collections import abc as c\n__all__ = ['c']\n",
        "def f() -> None: ...\n__all__ = ['f']\n",
        "async def g() -> None: ...\n__all__ = ['g']\n",
        "class C: ...\n__all__ = ['C']\n",
        "V = 1\n__all__ = ['V']\n",
        "V: int = 1\n__all__ = ['V']\n",
        "if True:\n    A = 1\nelse:\n    A = 2\n__all__ = ['A']\n",
        "for i in []:\n    B = i\nelse:\n    B = 0\n__all__ = ['B']\n",
        "while False:\n    D = 1\nelse:\n    D = 2\n__all__ = ['D']\n",
        "import contextlib\nwith contextlib.suppress(OSError):\n    E = 1\n__all__ = ['E']\n",
        "__all__ = ('T',)\nT = 1\n",
    ],
    ids=[
        "import",
        "import-as",
        "from-import",
        "from-import-as",
        "def",
        "async-def",
        "class",
        "assign",
        "ann-assign",
        "if-else",
        "for-else",
        "while-else",
        "with",
        "tuple-all",
    ],
)
def test_module_level_bindings_satisfy_the_promise(tmp_path: Path, code: str) -> None:
    path = tmp_path / "mod.py"
    _write(path, code)

    assert AllExportsRule().run([path]) == []


@pytest.mark.parametrize(
    "code",
    [
        "def outer() -> None:\n    inner = 1\n__all__ = ['inner']\n",
        "class C:\n    member = 1\n__all__ = ['member']\n",
    ],
    ids=["function-body", "class-body"],
)
def test_names_bound_in_another_namespace_do_not_count(tmp_path: Path, code: str) -> None:
    path = tmp_path / "mod.py"
    _write(path, code)

    violations = AllExportsRule().run([path])

    assert [v.kind for v in violations] == ["all-exports-undefined"]


@pytest.mark.parametrize(
    "code",
    [
        "V = 1\n__all__ = ['V'] + []\n",
        "V = 1\n__all__ = ['V']\n__all__ += ['W']\n",
        "V = 1\n__all__ = [V]\n",
        "V = 1\n__all__ = ['V', 2]\n",
    ],
    ids=["concatenation", "aug-assign", "non-literal-element", "non-string-element"],
)
def test_an_all_that_cannot_be_read_statically_is_rejected(tmp_path: Path, code: str) -> None:
    # A computed __all__ makes a promise nothing can check, so the rule
    # refuses the construction rather than silently skipping the file.
    path = tmp_path / "mod.py"
    _write(path, code)

    violations = AllExportsRule().run([path])

    assert [v.kind for v in violations] == ["all-exports-not-literal"]


@pytest.mark.parametrize(
    "code",
    [
        "x = 1\n",
        "other = 1\nother += 1\n",
        "d = {}\nd['k'] = 1\n",
        "a, b = 1, 2\n",
        "print(1)\n",
        "obj = object()\nobj.attr: int = 1\n",
    ],
    ids=["no-all", "aug-assign-elsewhere", "subscript", "tuple-unpack", "expr", "ann-attribute"],
)
def test_files_without_a_checkable_all_are_silent(tmp_path: Path, code: str) -> None:
    path = tmp_path / "mod.py"
    _write(path, code)

    assert AllExportsRule().run([path]) == []


def test_only_the_unresolvable_names_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    _write(path, "V = 1\n__all__ = ['V', 'missing', 'gone']\n")

    violations = AllExportsRule().run([path])

    assert [v.line for v in violations] == ["missing", "gone"]


def test_an_unparseable_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    _write(path, "def (:\n")

    with pytest.raises(RuntimeError, match="failed to parse"):
        AllExportsRule().run([path])


def test_the_rule_is_named_for_its_summary_line() -> None:
    assert AllExportsRule().name == "all-exports"


def test_the_orchestrator_actually_runs_this_rule(tmp_path: Path) -> None:
    # Registration is not invocation. This drives the real entry point and
    # asserts the defect changes the exit code.
    monorepo_root = tmp_path / "repo"
    monorepo_root.mkdir()
    (monorepo_root / "monorepo-guards.toml").write_text(
        "[guards]\n"
        'directories = ["src"]\n'
        'exclude_parts = [".venv"]\n'
        "forbid_pyi = true\n"
        "allow_print_in_tests = false\n"
        "dataclass_ban_segments = []\n",
        encoding="utf-8",
    )
    project_root = monorepo_root / "services" / "svc"
    _write(project_root / "src" / "pkg" / "__init__.py", '__all__ = ["create_app"]\n')

    assert run_for_project(monorepo_root, project_root) == 2

    _write(
        project_root / "src" / "pkg" / "__init__.py",
        "from pkg.app import create_app\n\n__all__ = ['create_app']\n",
    )
    _write(project_root / "src" / "pkg" / "app.py", "def create_app() -> None: ...\n")

    assert run_for_project(monorepo_root, project_root) == 0
