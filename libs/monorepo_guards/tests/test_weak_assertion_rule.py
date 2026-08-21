"""Tests for WeakAssertionRule's core detections.

Edge-case branch coverage and the key-in-dict detection live in
``test_weak_assertion_edge_cases.py``; the MLTestQualityRule tests live
in ``test_ml_test_quality_rule.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.test_quality_rules import WeakAssertionRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestWeakAssertionRule:
    """Tests for WeakAssertionRule."""

    def test_detects_assert_is_not_none(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = 1\n    assert x is not None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-is-not-none"

    def test_detects_isinstance_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = 1\n    assert isinstance(x, int)\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-isinstance"

    def test_detects_hasattr_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = object()\n    assert hasattr(x, 'a')\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-hasattr"

    def test_detects_len_greater_than_zero(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = [1]\n    assert len(x) > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-len-zero"

    def test_detects_len_gte_one(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    x = [1]\n    assert len(x) >= 1\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-len-zero"

    def test_detects_string_in_output(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = (
            "def test_example(capsys):\n"
            "    print('hello')\n"
            "    captured = capsys.readouterr()\n"
            "    assert 'hello' in captured.out\n"
        )
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-in-output"

    def test_detects_string_in_stderr(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'err' in result.stderr\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-in-output"

    def test_detects_mock_called_without_args(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    mock = Mock()\n    mock()\n    assert mock.called\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "mock-without-assert-called-with"

    def test_detects_excessive_mocking(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from unittest.mock import patch

def test_example():
    with patch('a.b'):
        with patch('c.d'):
            with patch('e.f'):
                with patch('g.h'):
                    pass
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert any(v.kind == "excessive-mocking" for v in violations)

    def test_detects_patch_decorator(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
from unittest import mock

def test_example():
    with mock.patch('a.b'):
        with mock.patch('c.d'):
            with mock.patch('e.f'):
                with mock.patch('g.h'):
                    pass
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert any(v.kind == "excessive-mocking" for v in violations)

    def test_ignores_non_test_files(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src" / "foo.py"
        code = "def foo():\n    x = None\n    assert x is None\n"
        _write(src_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([src_file])

        assert len(violations) == 0

    def test_ignores_non_test_functions(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def helper():\n    x = None\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_allows_strong_assertions(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    before = get_value()
    do_something()
    after = get_value()
    assert after < before
    assert result == expected
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        weak_kinds = {
            "weak-assertion-is-not-none",
        }
        weak_violations = [v for v in violations if v.kind in weak_kinds]
        assert len(weak_violations) == 0

    def test_handles_async_test_functions(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "async def test_example():\n    x = 1\n    assert x is not None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 1
        assert violations[0].kind == "weak-assertion-is-not-none"

    def test_raises_on_syntax_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example(\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        with pytest.raises(RuntimeError, match="failed to parse"):
            rule.run([test_file])

    def test_ignores_non_test_prefix_files(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "conftest.py"
        code = "def test_example():\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_ignores_valid_len_comparisons(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) == 5\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ignores_non_constant_comparisons(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        is_violations = [v for v in violations if "is-none" in v.kind]
        assert len(is_violations) == 0

    def test_ml_project_mode_detects_training_issues(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_model():
    model.train()
    assert model is not None
"""
        _write(test_file, code)

        rule = WeakAssertionRule(is_ml_project=True)
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-comparison" for v in violations)

    def test_ml_project_mode_allows_loss_comparison(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_model():
    model.train()
    assert loss_after < loss_before
"""
        _write(test_file, code)

        rule = WeakAssertionRule(is_ml_project=True)
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-comparison"]
        assert len(train_violations) == 0

    def test_ml_project_mode_rejects_wrong_comparison(self, tmp_path: Path) -> None:
        """Comparison with wrong variable names should not satisfy the loss check."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_model():
    model.train()
    assert x < y
"""
        _write(test_file, code)

        rule = WeakAssertionRule(is_ml_project=True)
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-comparison" for v in violations)

    def test_ml_project_mode_rejects_non_name_comparison(self, tmp_path: Path) -> None:
        """Comparison with attribute access should not satisfy the loss check."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_model():
    model.train()
    assert obj.loss_after < obj.loss_before
"""
        _write(test_file, code)

        rule = WeakAssertionRule(is_ml_project=True)
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-comparison" for v in violations)
