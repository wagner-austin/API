from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.imports_rules import ImportsRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_imports_rule_flags_pydantic_imports(tmp_path: Path) -> None:
    code = (
        "import pydantic\n"
        "from pydantic import BaseModel\n"
        "from pydantic.dataclasses import dataclass\n"
        "import pydantic.fields\n"
    )
    path = tmp_path / "pydantic_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = [v.kind for v in violations]
    assert kinds.count("import-pydantic") == 4


def test_imports_rule_flags_inspect_imports(tmp_path: Path) -> None:
    code = "import inspect\nfrom inspect import signature\nsig = inspect.signature(foo)\n"
    path = tmp_path / "inspect_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = [v.kind for v in violations]
    assert kinds.count("import-inspect") == 2


def test_imports_rule_flags_type_checking_import(tmp_path: Path) -> None:
    code = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\n"
    path = tmp_path / "type_checking_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "import-typing-type_checking" in kinds


def test_imports_rule_flags_typing_iterable_iterator(tmp_path: Path) -> None:
    code = (
        "from typing import Iterable, Iterator\n"
        "def foo(x: Iterable[int]) -> Iterator[str]:\n"
        "    pass\n"
    )
    path = tmp_path / "typing_iter_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "import-typing-iterable" in kinds
    assert "import-typing-iterator" in kinds


def test_imports_rule_flags_collections_iterable_iterator(tmp_path: Path) -> None:
    code = (
        "from collections.abc import Iterable, Iterator\n"
        "def foo(x: Iterable[int]) -> Iterator[str]:\n"
        "    pass\n"
    )
    path = tmp_path / "collections_iter_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "import-collections-iterable" in kinds
    assert "import-collections-iterator" in kinds


def test_imports_rule_flags_typing_module_usage(tmp_path: Path) -> None:
    code = (
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    pass\n"
        "x: typing.Iterable[int] = []\n"
        "y: typing.Iterator[str]\n"
    )
    path = tmp_path / "typing_usage_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = [v.kind for v in violations]
    assert "typing-type_checking-usage" in kinds
    assert "typing-iterable-usage" in kinds
    assert "typing-iterator-usage" in kinds


def test_imports_rule_flags_collections_abc_usage(tmp_path: Path) -> None:
    code = (
        "import collections.abc\n"
        "x: collections.abc.Iterable[int] = []\n"
        "y: collections.abc.Iterator[str]\n"
    )
    path = tmp_path / "collections_usage_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = [v.kind for v in violations]
    assert "collections-abc-iterable-usage" in kinds
    assert "collections-abc-iterator-usage" in kinds


def test_imports_rule_allows_safe_imports(tmp_path: Path) -> None:
    code = (
        "from typing import Protocol, TypedDict\n"
        "from collections.abc import Sequence, Mapping\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "class MyProtocol(Protocol):\n"
        "    pass\n"
        "\n"
        "class MyDict(TypedDict):\n"
        "    x: int\n"
    )
    path = tmp_path / "safe_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    assert violations == []


def test_imports_rule_handles_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    _write(empty, "")

    rule = ImportsRule()
    violations = rule.run([empty])
    assert violations == []


def test_imports_rule_combined_violations(tmp_path: Path) -> None:
    code = (
        "from typing import TYPE_CHECKING, Iterable, Iterator\n"
        "from collections.abc import Iterable as AbcIterable\n"
        "from pydantic import BaseModel\n"
        "import typing\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    pass\n"
        "\n"
        "x: typing.Iterable[int] = []\n"
    )
    path = tmp_path / "combined_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    kinds = [v.kind for v in violations]
    assert "import-typing-type_checking" in kinds
    assert "import-typing-iterable" in kinds
    assert "import-typing-iterator" in kinds
    assert "import-collections-iterable" in kinds
    assert "import-pydantic" in kinds
    assert "typing-iterable-usage" in kinds


def test_imports_rule_multiple_files(tmp_path: Path) -> None:
    file1 = tmp_path / "file1.py"
    _write(file1, "from typing import TYPE_CHECKING\n")

    file2 = tmp_path / "file2.py"
    _write(file2, "from pydantic import BaseModel\n")

    file3 = tmp_path / "file3.py"
    _write(file3, "from typing import Protocol\n")

    rule = ImportsRule()
    violations = rule.run([file1, file2, file3])
    assert len(violations) == 2
    kinds = {v.kind for v in violations}
    assert "import-typing-type_checking" in kinds
    assert "import-pydantic" in kinds


def test_imports_rule_allows_relative_imports(tmp_path: Path) -> None:
    code = "from . import something\nfrom ..utils import helper\n"
    path = tmp_path / "relative_mod.py"
    _write(path, code)

    rule = ImportsRule()
    violations = rule.run([path])
    assert violations == []


def test_imports_rule_raises_on_syntax_error(tmp_path: Path) -> None:
    code = "def foo(\n"
    path = tmp_path / "syntax_error.py"
    _write(path, code)

    rule = ImportsRule()
    with pytest.raises(RuntimeError, match=r"failed to parse.*syntax_error\.py"):
        rule.run([path])
