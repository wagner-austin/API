from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.typing_rules import TypingRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_typing_rule_flags_any_cast_and_type_ignore(tmp_path: Path) -> None:
    any_kw = "An" + "y"
    ti = "# " + "type" + ": " + "ignore"
    src = (
        f"from typing import {any_kw}, cast\n"
        "from typing import Optional\n"
        f"x: {any_kw} = 1  {ti}\n"
        "y = cast(int, 1)\n"
        "import typing\n"
        f"z: typing.{any_kw} = 2\n"
        "w = typing.cast(str, 3)\n"
    )
    path = tmp_path / "mod.py"
    _write(path, src)

    rule = TypingRule()
    violations = rule.run([path])

    kinds = {v.kind for v in violations}
    assert "typing-import-any" in kinds
    assert "typing-import-cast" in kinds
    assert "any-usage" in kinds
    assert "cast-call" in kinds
    assert "typing-cast-usage" in kinds
    assert "type-ignore" in kinds


def test_typing_rule_raises_on_syntax_error(tmp_path: Path) -> None:
    code = "class Foo\n"
    path = tmp_path / "syntax_err.py"
    _write(path, code)

    rule = TypingRule()
    with pytest.raises(RuntimeError, match=r"failed to parse.*syntax_err\.py"):
        rule.run([path])


def test_typing_rule_flags_object_in_annotations(tmp_path: Path) -> None:
    code = (
        "x: object = 1\ndef foo(y: object) -> object:\n    return y\nz: dict[object, object] = {}\n"
    )
    path = tmp_path / "object_annot.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "object-in-annotation" in kinds
    assert len([v for v in violations if v.kind == "object-in-annotation"]) == 4


def test_typing_rule_allows_valid_object_usage(tmp_path: Path) -> None:
    code = (
        "class Foo(object):\n"
        "    pass\n"
        "def bar() -> None:\n"
        "    x = object()\n"
        "    if isinstance(x, object):\n"
        "        pass\n"
    )
    path = tmp_path / "valid_object.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    object_violations = [v for v in violations if v.kind == "object-in-annotation"]
    assert object_violations == []


def test_typing_rule_flags_unknownjson_misuse(tmp_path: Path) -> None:
    # Using string annotation pattern (no TypeAlias)
    uj_def = "dict[str, UnknownJson] | list[UnknownJson] | str | int | float | bool | None"
    code = (
        "# Recursive JSON type as string annotation (correct pattern)\n"
        f"UnknownJson = '{uj_def}'\n"
        "# Public function with UnknownJson return\n"
        "def get_data() -> UnknownJson:\n"
        "    return {}\n"
        "# Public function with UnknownJson param\n"
        "def process(data: UnknownJson) -> None:\n"
        "    pass\n"
        "# Class with UnknownJson attribute\n"
        "class Cache:\n"
        "    data: UnknownJson\n"
        "# Module-level variable\n"
        "cache: UnknownJson = {}\n"
    )
    path = tmp_path / "bad_unknownjson.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "unknownjson-public-return" in kinds
    assert "unknownjson-public-param" in kinds
    assert "unknownjson-class-attr" in kinds
    assert "unknownjson-module-var" in kinds


def test_typing_rule_allows_valid_unknownjson_usage(tmp_path: Path) -> None:
    # Using string annotation pattern (no TypeAlias)
    uj_def = "dict[str, UnknownJson] | list[UnknownJson] | str | int | float | bool | None"
    code = (
        "# Recursive JSON type as string annotation (correct pattern)\n"
        f"UnknownJson = '{uj_def}'\n"
        "# Internal helper allowed to use UnknownJson\n"
        "def _load_json_dict(s: str) -> dict[str, UnknownJson] | None:\n"
        "    import json\n"
        "    raw: UnknownJson = json.loads(s)\n"
        "    if not isinstance(raw, dict):\n"
        "        return None\n"
        "    return raw\n"
        "def _decode_started(obj: dict[str, UnknownJson]) -> dict[str, str] | None:\n"
        "    return None\n"
        "def _attach_optional(target: dict[str, str], src: dict[str, UnknownJson]) -> None:\n"
        "    pass\n"
    )
    path = tmp_path / "valid_unknownjson.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    uj_violations = [v for v in violations if v.kind.startswith("unknownjson-")]
    assert uj_violations == []


def test_typing_rule_flags_typealias_import(tmp_path: Path) -> None:
    code = "from typing import TypeAlias\nx: int = 1\n"
    path = tmp_path / "typealias_import.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "typing-import-typealias" in kinds


def test_typing_rule_flags_all_typealias(tmp_path: Path) -> None:
    code = (
        "from typing import TypeAlias\n"
        "from platform_core.config import TurkicApiSettings\n"
        "# Non-recursive alias (forbidden)\n"
        "Settings: TypeAlias = TurkicApiSettings\n"
        "# Non-recursive dict alias (forbidden)\n"
        "JsonDict: TypeAlias = dict[str, str | int | float | bool | None]\n"
        "# Recursive alias (also forbidden - all TypeAlias banned)\n"
        "UnknownJson: TypeAlias = (\n"
        "    dict[str, 'UnknownJson'] | list['UnknownJson'] | str | int | float | bool | None\n"
        ")\n"
    )
    path = tmp_path / "bad_alias.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    alias_violations = [v for v in violations if v.kind == "typealias-forbidden"]
    # All 3 TypeAlias usages should be flagged
    assert len(alias_violations) == 3


def test_typing_rule_string_annotation_pattern_allowed(tmp_path: Path) -> None:
    # String annotation pattern is the correct way to define recursive types
    uj_def = "dict[str, UnknownJson] | list[UnknownJson] | str | int | float | bool | None"
    jv_def = "dict[str, JSONValue] | list[JSONValue] | str | int | float | bool | None"
    code = (
        "# Recursive JSON type - string annotation (allowed)\n"
        f"UnknownJson = '{uj_def}'\n"
        "# Another recursive type - string annotation (allowed)\n"
        f"JSONValue = '{jv_def}'\n"
    )
    path = tmp_path / "valid_string_annotations.py"
    _write(path, code)

    rule = TypingRule()
    violations = rule.run([path])
    # No TypeAlias violations since we're not using TypeAlias
    alias_violations = [v for v in violations if v.kind == "typealias-forbidden"]
    assert alias_violations == []
