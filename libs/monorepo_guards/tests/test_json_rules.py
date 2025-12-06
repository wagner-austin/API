from __future__ import annotations

from pathlib import Path

from monorepo_guards.json_rules import JsonRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_json_rule_flags_direct_loads(tmp_path: Path) -> None:
    bad_attr = tmp_path / "src" / "uses_json_attr.py"
    bad_name = tmp_path / "src" / "uses_json_name.py"
    ok_file = tmp_path / "src" / "platform_core" / "json_utils.py"
    ok_test = tmp_path / "tests" / "test_json_utils.py"
    _write(bad_attr, "import json\nobj = json.loads('{}')\n")
    _write(bad_name, "from json import loads\nobj = loads('{}')\n")
    _write(ok_file, "import json\nobj = json.loads('{}')\n")
    _write(ok_test, "import json\nobj = json.loads('{}')\n")

    rule = JsonRule()
    violations = rule.run([bad_attr, bad_name, ok_file, ok_test])
    kinds = {v.kind for v in violations}
    files = {v.file.name for v in violations}
    assert "json-loads-banned" in kinds
    assert "json-import-banned" in kinds
    assert "uses_json_attr.py" in files
    assert "uses_json_name.py" in files


def test_json_rule_allows_helper_suffixes(tmp_path: Path) -> None:
    ok_file = tmp_path / "src" / "platform_core" / "json_utils.py"
    ok_test = tmp_path / "tests" / "test_json_utils.py"
    _write(ok_file, "import json\nobj = json.loads('{}')\n")
    _write(ok_test, "import json\nobj = json.loads('{}')\n")

    rule = JsonRule()
    violations = rule.run([ok_file, ok_test])
    assert violations == []
