"""Tests for test_quality_rules module."""

from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.test_quality_rules import MLTestQualityRule, WeakAssertionRule


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


class TestMLTestQualityRule:
    """Tests for MLTestQualityRule."""

    def test_detects_training_without_loss_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_epoch():
    model.train()
    optimizer.step()
    assert model is not None
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_allows_training_with_loss_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_epoch():
    loss_before = get_loss()
    model.train()
    optimizer.step()
    loss_after = get_loss()
    assert loss_after < loss_before
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        loss_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(loss_violations) == 0

    def test_detects_forward_pass_shape_only(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = model(input)
    assert output.shape == (batch, seq, vocab)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-forward-shape-only" for v in violations)

    def test_allows_forward_pass_with_value_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = model(input)
    assert output.shape == (batch, seq, vocab)
    assert output.mean().item() > 0.0
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        shape_violations = [v for v in violations if v.kind == "ml-forward-shape-only"]
        assert len(shape_violations) == 0

    def test_detects_optimizer_without_weight_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    optimizer.step()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-optimizer-no-weight-check" for v in violations)

    def test_allows_optimizer_with_weight_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    weights_before = model.linear.weight.clone()
    optimizer.step()
    weights_after = model.linear.weight
    assert not torch.equal(weights_before, weights_after)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        weight_violations = [v for v in violations if v.kind == "ml-optimizer-no-weight-check"]
        assert len(weight_violations) == 0

    def test_ignores_non_test_files(self, tmp_path: Path) -> None:
        src_file = tmp_path / "src" / "train.py"
        code = """
def train():
    model.train()
    optimizer.step()
"""
        _write(src_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([src_file])

        assert len(violations) == 0

    def test_ignores_non_test_prefix_files(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "conftest.py"
        code = """
def test_train():
    model.train()
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_raises_on_syntax_error(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example(\n"
        _write(test_file, code)

        rule = MLTestQualityRule()
        with pytest.raises(RuntimeError, match="failed to parse"):
            rule.run([test_file])

    def test_ignores_non_test_functions(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def helper():
    model.train()
    optimizer.step()
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_detects_train_call_usage(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train_loop():
    model.train()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_allows_http_client_train_call(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_http_client.py"
        code = """
async def test_http_client_train_method():
    http = HTTPModelTrainerClient(base_url="url", api_key="k")
    out = await http.train(user_id=1, model_family="gpt2")
    assert out["run_id"] == "r1"
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 0

    def test_allows_client_train_call(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_api_client.py"
        code = """
async def test_api_client_methods():
    client = ModelTrainerClient(base_url="url")
    result = await client.train(params)
    assert result["status"] == "ok"
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        train_violations = [v for v in violations if v.kind == "ml-train-no-loss-check"]
        assert len(train_violations) == 0

    def test_detects_chained_train_call(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_chained.py"
        code = """
def test_chained_train():
    # Chained access like self.model.train() should still be flagged
    self.model.train()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        # Should be flagged since it's not a simple http/client variable
        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_detects_backward_call(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_backward():
    loss.backward()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_allows_allclose_weight_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    optimizer.step()
    torch.allclose(w1, w2)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        weight_violations = [v for v in violations if v.kind == "ml-optimizer-no-weight-check"]
        assert len(weight_violations) == 0

    def test_allows_state_dict_before_check(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_optimizer():
    state_dict_before = model.state_dict()
    optimizer.step()
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        weight_violations = [v for v in violations if v.kind == "ml-optimizer-no-weight-check"]
        assert len(weight_violations) == 0


class TestBranchCoverage:
    """Additional tests for branch coverage."""

    def test_async_non_test_function_ignored(self, tmp_path: Path) -> None:
        """Async helper functions should be ignored."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "async def helper():\n    assert x is None\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert len(violations) == 0

    def test_is_not_none_with_non_none_constant(self, tmp_path: Path) -> None:
        """assert x is not 1 should not trigger is_not_none check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is not 1\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        not_none_violations = [v for v in violations if v.kind == "weak-assertion-is-not-none"]
        assert len(not_none_violations) == 0

    def test_len_with_non_call_left(self, tmp_path: Path) -> None:
        """assert x > 0 (not len(x)) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_len_with_non_len_func(self, tmp_path: Path) -> None:
        """assert size(x) > 0 should not trigger len check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert size(x) > 0\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_len_with_multiple_ops(self, tmp_path: Path) -> None:
        """assert 0 < len(x) < 10 should not trigger len check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 0 < len(x) < 10\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_string_in_stdout(self, tmp_path: Path) -> None:
        """assert 'x' in result.stdout should trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.stdout\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        assert any(v.kind == "weak-assertion-in-output" for v in violations)

    def test_string_in_non_output_attr(self, tmp_path: Path) -> None:
        """assert 'x' in result.data should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result.data\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        output_violations = [v for v in violations if v.kind == "weak-assertion-in-output"]
        assert len(output_violations) == 0

    def test_is_not_none_with_non_constant_comparator(self, tmp_path: Path) -> None:
        """assert x is not y should not trigger (non-constant)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert x is not y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        not_none_violations = [v for v in violations if v.kind == "weak-assertion-is-not-none"]
        assert len(not_none_violations) == 0

    def test_len_with_non_constant_comparator(self, tmp_path: Path) -> None:
        """assert len(x) > y should not trigger (non-constant)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) > y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_string_in_non_attribute(self, tmp_path: Path) -> None:
        """assert 'x' in result should not trigger (not attribute)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert 'x' in result\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        output_violations = [v for v in violations if v.kind == "weak-assertion-in-output"]
        assert len(output_violations) == 0

    def test_len_with_chained_comparison(self, tmp_path: Path) -> None:
        """assert len(x) > 0 < y should not trigger (chained comparison)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = "def test_example():\n    assert len(x) > 0 < y\n"
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        len_violations = [v for v in violations if v.kind == "weak-assertion-len-zero"]
        assert len(len_violations) == 0

    def test_ml_comparison_with_attribute_access(self, tmp_path: Path) -> None:
        """Attribute access in comparison should not match name patterns."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train():
    loss.backward()
    assert obj.after < obj.before
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        # Should still flag as no loss check since obj.after is not a Name node
        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_ml_comparison_with_subscript(self, tmp_path: Path) -> None:
        """Subscript in comparison should not match name patterns."""
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train():
    loss.backward()
    assert losses[0] < losses[1]
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        # Should still flag as no loss check since subscript is not a Name node
        assert any(v.kind == "ml-train-no-loss-check" for v in violations)


class TestKeyInDictRule:
    """Tests for weak-assertion-key-in-dict detection.

    The rule only flags `assert "key" in d` when `d` is also accessed via
    subscript elsewhere in the function (proving it's a dict, not a set).
    """

    def test_detects_key_in_dict_without_value_check(self, tmp_path: Path) -> None:
        """assert 'key' in d when d is used as dict should trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"key": "value", "other": "x"}
    assert "key" in d
    x = d["other"]  # proves d is a dict
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 1
        assert '"key"' in key_violations[0].line

    def test_allows_key_in_dict_with_value_check(self, tmp_path: Path) -> None:
        """assert 'key' in d followed by assert d['key'] == value should pass."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"key": "value"}
    assert "key" in d
    assert d["key"] == "value"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_detects_multiple_unverified_keys(self, tmp_path: Path) -> None:
        """Multiple unverified key checks should all be flagged."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"a": 1, "b": 2, "c": 3}
    assert "a" in d
    assert "b" in d
    x = d["c"]  # proves d is a dict
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 2

    def test_allows_partial_verification(self, tmp_path: Path) -> None:
        """Only unverified keys should be flagged, verified ones pass."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"a": 1, "b": 2}
    assert "a" in d
    assert "b" in d
    assert d["a"] == 1
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 1
        assert '"b"' in key_violations[0].line

    def test_ignores_set_membership_check(self, tmp_path: Path) -> None:
        """assert 'key' in s when s is only used as set should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    kinds = {"error-type-a", "error-type-b"}
    assert "error-type-a" in kinds
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_non_string_key(self, tmp_path: Path) -> None:
        """assert 1 in d (integer key) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {1: "value", 2: "other"}
    assert 1 in d
    x = d[2]
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_variable_key(self, tmp_path: Path) -> None:
        """assert key in d (variable key) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    key = "foo"
    d = {"foo": "value"}
    assert key in d
    x = d["foo"]
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_attribute_dict(self, tmp_path: Path) -> None:
        """assert 'key' in obj.d should not trigger (attribute access)."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    assert "key" in obj.data
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_ignores_chained_in_check(self, tmp_path: Path) -> None:
        """assert 'a' in 'b' in d (chained) should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    assert "a" in "abc" in some_list
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_nested_dict_access_verifies_outer_key(self, tmp_path: Path) -> None:
        """assert d['outer']['inner'] == x should verify 'outer' key."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"outer": {"inner": "value"}}
    assert "outer" in d
    assert d["outer"]["inner"] == "value"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_real_world_pattern_request_in_payload(self, tmp_path: Path) -> None:
        """Real-world pattern: assert 'request' in payload without value check."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_rq_enqueuer_methods():
    payload = {"run_id": "r1", "request": {"device": "cpu"}}
    assert "request" in payload
    assert "run_id" in payload
    assert payload["run_id"] == "r1"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        # "request" is not verified, "run_id" is verified
        # payload is used as dict (via payload["run_id"])
        assert len(key_violations) == 1
        assert '"request"' in key_violations[0].line

    def test_separate_functions_independent(self, tmp_path: Path) -> None:
        """Keys verified in one function don't satisfy checks in another."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_first():
    d = {"key": "value", "other": "x"}
    assert "key" in d
    x = d["other"]  # proves d is a dict

def test_second():
    d = {"key": "value"}
    assert d["key"] == "value"
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        # test_first should have violation, test_second should not
        assert len(key_violations) == 1
        assert "test_first" in key_violations[0].line

    def test_ignores_chained_comparisons_multiple_comparators(self, tmp_path: Path) -> None:
        """Chained comparisons with multiple comparators should not trigger."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"a": 1}
    x = d["a"]
    assert "key" in d in some_other  # multiple comparators
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0

    def test_variable_subscript_in_assert_does_not_verify(self, tmp_path: Path) -> None:
        """assert d[var] == x doesn't verify string key assertions."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    d = {"key": "value"}
    var = "key"
    assert d[var] == "value"  # variable subscript in assert
    assert "key" in d
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        # d is dict-like (via d[var] subscript)
        # "key" in d is not verified by d[var] (var is not a string constant)
        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 1

    def test_ignores_attribute_subscript_access(self, tmp_path: Path) -> None:
        """Attribute subscript like obj.attr["key"] should handle edge cases."""
        test_file = tmp_path / "tests" / "test_foo.py"
        code = """
def test_example():
    x = obj.data["key"]  # attr.subscript pattern
    assert "other" in d
"""
        _write(test_file, code)

        rule = WeakAssertionRule()
        violations = rule.run([test_file])

        # Should not crash, d is not used as dict
        key_violations = [v for v in violations if v.kind == "weak-assertion-key-in-dict"]
        assert len(key_violations) == 0
