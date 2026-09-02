"""Tests for MLTestQualityRule.

The WeakAssertionRule tests live in ``test_weak_assertion_rule.py`` and
``test_weak_assertion_edge_cases.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.ml_test_quality_rules import MLTestQualityRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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

    def test_allows_forward_pass_checked_with_torch_equal(self, tmp_path: Path) -> None:
        """``torch.equal`` is bitwise equality, the strictest check available.

        It was absent from the value-check list until 2026-08-30, so a test
        asserting the exact bytes a module produced was reported as "forward
        pass only checks shapes" -- the opposite of true. Left that way the
        rule pushes an author toward ``allclose``, trading a strict assertion
        for a loose one to get a green build.
        """
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = module.forward(x)
    assert torch.equal(output, expected)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations if v.kind == "ml-forward-shape-only"] == []

    def test_still_flags_a_forward_pass_with_no_value_check_at_all(self, tmp_path: Path) -> None:
        # The other direction of the pair: widening the list must not have
        # made the rule unable to fire.
        test_file = tmp_path / "tests" / "test_model.py"
        code = """
def test_forward():
    output = module.forward(x)
    assert output.shape == (2, 3)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations if v.kind == "ml-forward-shape-only"] == [
            "ml-forward-shape-only"
        ]

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


class TestMLBranchCoverage:
    """Branch coverage for the loss-comparison name matching."""

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


class TestGradientFormulaCalls:
    """A backward reached through a CLASS is a formula call, not training."""

    def test_a_cls_backward_call_is_not_training(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_autograd.py"
        code = """
def test_the_gradient_formula():
    cls = _matmul_class()
    grad_x, grad_w = cls.backward(ctx, grad_out)
    assert torch.equal(grad_x, expected)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations] == []

    def test_a_pascal_case_backward_call_is_not_training(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_autograd.py"
        code = """
def test_the_gradient_formula():
    grads = OwnedMatmul.backward(ctx, grad_out)
    assert torch.equal(grads[0], expected)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations] == []

    def test_a_lowercase_tensor_backward_still_flags(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train():
    loss.backward()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_a_chained_backward_still_flags(self, tmp_path: Path) -> None:
        # The receiver is an Attribute, not a Name -- outside the formula
        # carve-out, so the conservative side keeps flagging.
        test_file = tmp_path / "tests" / "test_train.py"
        code = """
def test_train():
    outputs.loss.backward()
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)


class TestPytestRaisesSuppression:
    """A call that must raise produces no tensor to value-check."""

    def test_a_forward_only_inside_raises_is_not_flagged(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
def test_every_unowned_path_is_refused():
    with pytest.raises(ValueError, match="refused"):
        wrapper.forward(hidden, attention_mask=mask)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations] == []

    def test_a_train_call_inside_raises_is_not_flagged(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
def test_training_is_refused():
    with pytest.raises(ValueError, match="eval-only"):
        trainer.train()
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations] == []

    def test_a_forward_outside_raises_still_flags(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
def test_mixed():
    out = module.forward(x)
    with pytest.raises(ValueError):
        module.forward(bad)
    assert out.shape == (1, 4)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-forward-shape-only" for v in violations)

    def test_a_flag_set_before_the_raises_block_survives_it(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
def test_backward_then_refusal():
    loss.backward()
    with pytest.raises(ValueError):
        helper.forward(bad)
    assert True
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-train-no-loss-check" for v in violations)

    def test_a_plain_with_block_does_not_suppress(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
def test_under_no_grad():
    with torch.no_grad():
        out = module.forward(x)
    assert out.shape == (1, 4)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-forward-shape-only" for v in violations)

    def test_non_call_and_bare_call_contexts_do_not_suppress(self, tmp_path: Path) -> None:
        # A bare name context and a plain-function context are neither of
        # them pytest.raises; the forward inside still counts.
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
def test_under_locks():
    with lock:
        with open(path) as handle:
            out = module.forward(x)
    assert out.shape == (1, 4)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert any(v.kind == "ml-forward-shape-only" for v in violations)

    def test_an_async_with_raises_block_suppresses_too(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tests" / "test_refusals.py"
        code = """
async def test_async_refusal():
    async with pytest.raises(ValueError):
        await runner.forward(bad)
"""
        _write(test_file, code)

        rule = MLTestQualityRule()
        violations = rule.run([test_file])

        assert [v.kind for v in violations] == []
