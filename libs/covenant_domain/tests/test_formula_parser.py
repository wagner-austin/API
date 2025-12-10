"""Tests for covenant_domain.formula_parser module."""

from __future__ import annotations

import pytest

from covenant_domain.formula_parser import (
    FormulaEvalError,
    FormulaParseError,
    _apply_operator,
    _parse_operand,
    _to_rpn,
    _tokenize,
    evaluate_formula,
)


class TestTokenize:
    def test_simple_metric(self) -> None:
        result = _tokenize("total_debt")
        assert result == ["total_debt"]

    def test_binary_operation(self) -> None:
        result = _tokenize("a + b")
        assert result == ["a", "+", "b"]

    def test_division(self) -> None:
        result = _tokenize("total_debt / ebitda")
        assert result == ["total_debt", "/", "ebitda"]

    def test_multiplication(self) -> None:
        result = _tokenize("a * b")
        assert result == ["a", "*", "b"]

    def test_subtraction(self) -> None:
        result = _tokenize("a - b")
        assert result == ["a", "-", "b"]

    def test_parentheses(self) -> None:
        result = _tokenize("(a + b) * c")
        assert result == ["(", "a", "+", "b", ")", "*", "c"]

    def test_no_spaces(self) -> None:
        result = _tokenize("a+b")
        assert result == ["a", "+", "b"]

    def test_numeric_literal(self) -> None:
        result = _tokenize("a + 100")
        assert result == ["a", "+", "100"]

    def test_alphanumeric_metric(self) -> None:
        result = _tokenize("metric_1 + metric_2")
        assert result == ["metric_1", "+", "metric_2"]

    def test_invalid_character(self) -> None:
        with pytest.raises(FormulaParseError) as exc_info:
            _tokenize("a @ b")
        assert "Invalid character" in str(exc_info.value)

    def test_empty_formula(self) -> None:
        result = _tokenize("")
        assert result == []

    def test_whitespace_only(self) -> None:
        result = _tokenize("   ")
        assert result == []


class TestToRpn:
    def test_simple_addition(self) -> None:
        tokens = ["a", "+", "b"]
        result = _to_rpn(tokens)
        assert result == ["a", "b", "+"]

    def test_precedence_mul_over_add(self) -> None:
        tokens = ["a", "+", "b", "*", "c"]
        result = _to_rpn(tokens)
        assert result == ["a", "b", "c", "*", "+"]

    def test_precedence_with_parentheses(self) -> None:
        tokens = ["(", "a", "+", "b", ")", "*", "c"]
        result = _to_rpn(tokens)
        assert result == ["a", "b", "+", "c", "*"]

    def test_left_associativity(self) -> None:
        tokens = ["a", "-", "b", "-", "c"]
        result = _to_rpn(tokens)
        assert result == ["a", "b", "-", "c", "-"]

    def test_nested_parentheses(self) -> None:
        tokens = ["(", "(", "a", "+", "b", ")", ")"]
        result = _to_rpn(tokens)
        assert result == ["a", "b", "+"]

    def test_mismatched_close_paren(self) -> None:
        tokens = ["a", "+", "b", ")"]
        with pytest.raises(FormulaParseError) as exc_info:
            _to_rpn(tokens)
        assert "Mismatched parentheses" in str(exc_info.value)

    def test_mismatched_open_paren(self) -> None:
        tokens = ["(", "a", "+", "b"]
        with pytest.raises(FormulaParseError) as exc_info:
            _to_rpn(tokens)
        assert "Mismatched parentheses" in str(exc_info.value)


class TestApplyOperator:
    def test_addition(self) -> None:
        result = _apply_operator("+", 1_000_000, 2_000_000)
        assert result == 3_000_000

    def test_subtraction(self) -> None:
        result = _apply_operator("-", 5_000_000, 2_000_000)
        assert result == 3_000_000

    def test_multiplication(self) -> None:
        result = _apply_operator("*", 2_000_000, 3_000_000)
        assert result == 6_000_000

    def test_division(self) -> None:
        result = _apply_operator("/", 6_000_000, 2_000_000)
        assert result == 3_000_000

    def test_division_by_zero(self) -> None:
        with pytest.raises(FormulaEvalError) as exc_info:
            _apply_operator("/", 6_000_000, 0)
        assert "Division by zero" in str(exc_info.value)


class TestParseOperand:
    def test_numeric_literal(self) -> None:
        metrics: dict[str, int] = {}
        result = _parse_operand("5", metrics)
        assert result == 5_000_000

    def test_metric_lookup(self) -> None:
        metrics: dict[str, int] = {"total_debt": 100_000_000}
        result = _parse_operand("total_debt", metrics)
        assert result == 100_000_000

    def test_metric_not_found(self) -> None:
        metrics: dict[str, int] = {}
        with pytest.raises(KeyError):
            _parse_operand("unknown_metric", metrics)


class TestEvaluateFormula:
    def test_simple_division(self) -> None:
        metrics = {"total_debt": 100_000_000, "ebitda": 50_000_000}
        result = evaluate_formula("total_debt / ebitda", metrics)
        assert result == 2_000_000

    def test_simple_addition(self) -> None:
        metrics = {"a": 1_000_000, "b": 2_000_000}
        result = evaluate_formula("a + b", metrics)
        assert result == 3_000_000

    def test_simple_subtraction(self) -> None:
        metrics = {"a": 5_000_000, "b": 2_000_000}
        result = evaluate_formula("a - b", metrics)
        assert result == 3_000_000

    def test_simple_multiplication(self) -> None:
        metrics = {"a": 2_000_000, "b": 3_000_000}
        result = evaluate_formula("a * b", metrics)
        assert result == 6_000_000

    def test_complex_formula(self) -> None:
        metrics = {"a": 10_000_000, "b": 2_000_000, "c": 3_000_000}
        result = evaluate_formula("(a + b) * c", metrics)
        assert result == 36_000_000

    def test_with_numeric_literal(self) -> None:
        metrics = {"a": 10_000_000}
        result = evaluate_formula("a + 5", metrics)
        assert result == 15_000_000

    def test_division_by_zero(self) -> None:
        metrics = {"a": 10_000_000, "b": 0}
        with pytest.raises(FormulaEvalError) as exc_info:
            evaluate_formula("a / b", metrics)
        assert "Division by zero" in str(exc_info.value)

    def test_missing_metric(self) -> None:
        metrics = {"a": 10_000_000}
        with pytest.raises(KeyError):
            evaluate_formula("a + unknown", metrics)

    def test_invalid_formula_syntax(self) -> None:
        metrics = {"a": 10_000_000}
        with pytest.raises(FormulaParseError):
            evaluate_formula("a @ b", metrics)

    def test_insufficient_operands(self) -> None:
        metrics = {"a": 10_000_000}
        with pytest.raises(FormulaEvalError) as exc_info:
            evaluate_formula("+", metrics)
        assert "insufficient operands" in str(exc_info.value)

    def test_too_many_values(self) -> None:
        metrics = {"a": 10_000_000, "b": 20_000_000}
        with pytest.raises(FormulaEvalError) as exc_info:
            evaluate_formula("a b", metrics)
        assert "wrong number of values" in str(exc_info.value)

    def test_debt_to_ebitda_ratio(self) -> None:
        metrics = {"total_debt": 350_000_000, "ebitda": 100_000_000}
        result = evaluate_formula("total_debt / ebitda", metrics)
        assert result == 3_500_000

    def test_interest_coverage_ratio(self) -> None:
        metrics = {"ebitda": 50_000_000, "interest_expense": 10_000_000}
        result = evaluate_formula("ebitda / interest_expense", metrics)
        assert result == 5_000_000

    def test_current_ratio(self) -> None:
        metrics = {"current_assets": 200_000_000, "current_liabilities": 100_000_000}
        result = evaluate_formula("current_assets / current_liabilities", metrics)
        assert result == 2_000_000
