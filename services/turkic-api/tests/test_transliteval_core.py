from __future__ import annotations

from pathlib import Path

import pytest

import turkic_api.core.transliteval as te


def _write_temp_rule(rules_dir: Path, name: str, content: str) -> Path:
    path = rules_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def test_word_initial_macro_context_and_literal_context(tmp_path: Path) -> None:
    # Covers: word-initial (^), macro lookbehind ($V }), literal lookbehind (n } a), simple rule
    content = """
    $V = [a];
    ^ a > X ;
    $V } b > Y ;
    n } a > Z ;
    c > C ;
    a > q ;
    b > r ;
    :: NFC ;
    """
    temp = _write_temp_rule(te._RULES_DIR, "combo_test.rules", content.strip())
    try:
        rules = te.load_rules("combo_test.rules")
        assert te.apply_rules("abc", rules) == "XYC"
        assert te.apply_rules("nac", rules) == "ZaC"
        assert te.apply_rules("a", rules) == "X"
    finally:
        temp.unlink(missing_ok=True)


def test_macro_expansion_in_lhs_matches_and_skips(tmp_path: Path) -> None:
    content = """
    $Apo = [xE�];
    o $Apo > O ;
    o > o ;
    """
    temp = _write_temp_rule(te._RULES_DIR, "macro_lhs.rules", content.strip())
    try:
        rules = te.load_rules("macro_lhs.rules")
        # Matches when macro char follows prefix
        assert te.apply_rules("ox", rules) == "O"
        # Does not match when different following char; falls back to simple rule
        assert te.apply_rules("oy", rules) == "oy"
    finally:
        temp.unlink(missing_ok=True)


def test_empty_rhs_performs_deletion(tmp_path: Path) -> None:
    content = """
    x > ;
    y > z ;
    """
    temp = _write_temp_rule(te._RULES_DIR, "delete.rules", content.strip())
    try:
        rules = te.load_rules("delete.rules")
        assert te.apply_rules("xyx", rules) == "z"
    finally:
        temp.unlink(missing_ok=True)


def test_undefined_macro_raises_parse_error(tmp_path: Path) -> None:
    content = """
    $Missing } a > b ;
    """
    temp = _write_temp_rule(te._RULES_DIR, "bad_macro.rules", content.strip())
    try:
        with pytest.raises(te.RuleParseError) as exc:
            te.load_rules("bad_macro.rules")
        assert "not defined" in str(exc.value)
    finally:
        temp.unlink(missing_ok=True)


def test_iter_rule_specs_splits_multiple_statements() -> None:
    entries = te._iter_rule_specs("a > b ; c > d; ; e > f")
    assert entries == ["a > b", "c > d", "e > f"]
