from __future__ import annotations

from pathlib import Path

from monorepo_guards.exceptions_rules import ExceptionsRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_exceptions_rule_flags_silent_and_broad(tmp_path: Path) -> None:
    code = (
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    logger.error('x')\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    raise\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except ValueError:\n"
        "    a = 1\n"
    )
    path = tmp_path / "exc_mod.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "silent-except-body" in kinds
    assert "broad-except-requires-log-and-raise" in kinds
    assert "except-without-log-or-raise" in kinds


def test_exceptions_rule_typed_with_log_or_raise_is_ok(tmp_path: Path) -> None:
    code = (
        "try:\n"
        "    1/0\n"
        "except ValueError:\n"
        "    logger.error('x')\n"
        "\n"
        "try:\n"
        "    1/0\n"
        "except KeyError:\n"
        "    raise\n"
    )
    path = tmp_path / "typed_ok.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    assert violations == []


def test_exceptions_rule_no_body_detected(tmp_path: Path) -> None:
    # An except header at EOF with no body should be flagged as silent body
    code = "try:\n    1/0\nexcept Exception:\n"
    path = tmp_path / "no_body.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "silent-except-body" in kinds


def test_exceptions_rule_broad_with_log_and_raise_and_skip_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    _write(empty, "")

    code = (
        "try:\n    1/0\nexcept Exception:\n    logger.error('x')\n    raise RuntimeError('fail')\n"
    )
    path = tmp_path / "broad_ok.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([empty, path])
    assert violations == []


def test_exceptions_rule_body_start_skips_blank_lines(tmp_path: Path) -> None:
    code = "try:\n    1/0\nexcept Exception:\n\n    logger.error('x')\n    raise\n"
    path = tmp_path / "blank_body.py"
    _write(path, code)

    rule = ExceptionsRule()
    violations = rule.run([path])
    # Broad except with both log and raise after a blank line should be allowed
    assert violations == []
