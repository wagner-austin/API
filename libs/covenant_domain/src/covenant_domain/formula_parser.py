from __future__ import annotations

from collections.abc import Mapping, Sequence

# Token types
_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})
_PRECEDENCE: dict[str, int] = {"+": 1, "-": 1, "*": 2, "/": 2}


class FormulaParseError(Exception):
    """Raised when formula parsing fails."""


class FormulaEvalError(Exception):
    """Raised when formula evaluation fails."""


def _tokenize(formula: str) -> list[str]:
    """
    Tokenize formula into operators, parentheses, and metric names.

    Raises FormulaParseError on invalid characters.
    """
    tokens: list[str] = []
    current: list[str] = []

    for char in formula:
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        elif char in _OPERATORS or char in ("(", ")"):
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
        elif char.isalnum() or char == "_":
            current.append(char)
        else:
            raise FormulaParseError(f"Invalid character in formula: {char!r}")

    if current:
        tokens.append("".join(current))

    return tokens


def _to_rpn(tokens: Sequence[str]) -> list[str]:
    """
    Convert infix tokens to Reverse Polish Notation using shunting-yard.

    Raises FormulaParseError on mismatched parentheses.
    """
    output: list[str] = []
    operator_stack: list[str] = []

    for token in tokens:
        if token in _OPERATORS:
            while (
                operator_stack
                and operator_stack[-1] != "("
                and operator_stack[-1] in _OPERATORS
                and _PRECEDENCE[operator_stack[-1]] >= _PRECEDENCE[token]
            ):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            if not operator_stack:
                raise FormulaParseError("Mismatched parentheses")
            operator_stack.pop()  # Remove the "("
        else:
            # Metric name or number
            output.append(token)

    while operator_stack:
        op = operator_stack.pop()
        if op == "(":
            raise FormulaParseError("Mismatched parentheses")
        output.append(op)

    return output


def _apply_operator(operator: str, a: int, b: int) -> int:
    """
    Apply a binary operator to two scaled integer operands.

    Raises FormulaEvalError on division by zero.
    """
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        # For multiplication of scaled values, adjust scale
        return (a * b) // 1_000_000
    # operator == "/"
    if b == 0:
        raise FormulaEvalError("Division by zero")
    # For division of scaled values, adjust scale
    return (a * 1_000_000) // b


def _parse_operand(token: str, metrics: Mapping[str, int]) -> int:
    """
    Parse a token as either an integer literal or metric name.

    Raises KeyError if metric name not found.
    """
    if token.isdigit():
        return int(token) * 1_000_000
    # Metric name lookup - raises KeyError if not found
    return metrics[token]


def _evaluate_rpn(rpn: Sequence[str], metrics: Mapping[str, int]) -> int:
    """
    Evaluate RPN expression with scaled integer arithmetic.

    Raises:
        KeyError: Unknown metric name
        FormulaEvalError: Division by zero or invalid expression
    """
    stack: list[int] = []

    for token in rpn:
        if token in _OPERATORS:
            if len(stack) < 2:
                raise FormulaEvalError("Invalid expression: insufficient operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(_apply_operator(token, a, b))
        else:
            stack.append(_parse_operand(token, metrics))

    if len(stack) != 1:
        raise FormulaEvalError("Invalid expression: wrong number of values")

    return stack[0]


def evaluate_formula(formula: str, metrics: Mapping[str, int]) -> int:
    """
    Evaluate arithmetic formula against metric values.

    All values are scaled integers (multiply by 1_000_000 for 6 decimal places).

    Supported operators: +, -, *, /
    Supported: parentheses, metric names, integer literals

    Raises:
        FormulaParseError: Invalid formula syntax
        FormulaEvalError: Division by zero or invalid expression
        KeyError: Unknown metric name
    """
    tokens = _tokenize(formula)
    rpn = _to_rpn(tokens)
    return _evaluate_rpn(rpn, metrics)


__all__ = [
    "FormulaEvalError",
    "FormulaParseError",
    "evaluate_formula",
]
