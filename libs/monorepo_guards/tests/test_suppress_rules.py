from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.suppress_rules import SuppressRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_suppress_rule_flags_qualified_and_unqualified(tmp_path: Path) -> None:
    code = (
        "import contextlib\n"
        "from contextlib import suppress\n"
        "with contextlib.suppress(Exception):\n"
        "    pass\n"
        "with suppress(RuntimeError):\n"
        "    pass\n"
        "with contextlib.ExitStack():\n"
        "    pass\n"
    )
    path = tmp_path / "sup_mod.py"
    _write(path, code)

    rule = SuppressRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "suppress-usage" in kinds
    assert len(violations) >= 2


def test_suppress_rule_deduplicates_same_line(tmp_path: Path) -> None:
    code = (
        "from contextlib import suppress\n"
        "with suppress(ValueError), suppress(RuntimeError):\n"
        "    pass\n"
    )
    path = tmp_path / "sup_multi.py"
    _write(path, code)

    rule = SuppressRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert kinds == {"suppress-usage"}
    assert len(violations) == 1


def test_suppress_rule_non_suppress_context_has_no_violation(tmp_path: Path) -> None:
    code = "def other() -> None:\n    pass\nwith other():\n    pass\n"
    path = tmp_path / "non_sup.py"
    _write(path, code)

    rule = SuppressRule()
    violations = rule.run([path])
    assert violations == []


def test_suppress_rule_skips_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    _write(empty, "")

    rule = SuppressRule()
    violations = rule.run([empty])
    assert violations == []


def test_suppress_rule_flags_pragma_no_cover(tmp_path: Path) -> None:
    pragma_word = "pra" + "gma"
    code = (
        "def foo() -> None:\n"
        "    try:\n"
        "        pass\n"
        f"    except Exception:  # {pragma_word}: no cover\n"
        "        raise\n"
    )
    path = tmp_path / "pragma_mod.py"
    _write(path, code)

    rule = SuppressRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "pragma-comment" in kinds


def test_suppress_rule_flags_pragma_variants(tmp_path: Path) -> None:
    pragma_word = "pra" + "gma"
    code = f"# {pragma_word}: no cover\nx = 1  # {pragma_word}: no branch\n# {pragma_word}:nocov\n"
    path = tmp_path / "pragma_variants.py"
    _write(path, code)

    rule = SuppressRule()
    violations = rule.run([path])
    assert len(violations) == 3
    assert all(v.kind == "pragma-comment" for v in violations)


def test_suppress_rule_allows_normal_comments(tmp_path: Path) -> None:
    code = (
        "# This is a normal comment\nx = 1  # inline comment about x\n# NOTE: this is important\n"
    )
    path = tmp_path / "normal_comments.py"
    _write(path, code)

    rule = SuppressRule()
    violations = rule.run([path])
    pragma_violations = [v for v in violations if v.kind == "pragma-comment"]
    assert pragma_violations == []


def test_suppress_rule_raises_on_syntax_error(tmp_path: Path) -> None:
    code = "if True\n"
    path = tmp_path / "bad_syntax.py"
    _write(path, code)

    rule = SuppressRule()
    with pytest.raises(RuntimeError, match=r"failed to parse.*bad_syntax\.py"):
        rule.run([path])
