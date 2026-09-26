"""Tests for the logistic-regression vocabularies."""

from __future__ import annotations

from covenant_ml import types
from covenant_ml.types_logreg import LogRegPenalty, LogRegSolver


def test_solver_words_are_sklearns() -> None:
    """The solver members carry sklearn's six solver words, in declaration order."""
    assert [member.value for member in LogRegSolver] == [
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga",
    ]


def test_penalty_words() -> None:
    """The penalty members carry the four penalty words, in declaration order."""
    assert [member.value for member in LogRegPenalty] == ["l1", "l2", "elasticnet", "none"]


def test_sklearn_penalty_spells_none_as_absence() -> None:
    """NONE maps to sklearn's None; every other penalty maps to its own word."""
    assert LogRegPenalty.NONE.sklearn_penalty is None
    assert LogRegPenalty.L1.sklearn_penalty == "l1"
    assert LogRegPenalty.L2.sklearn_penalty == "l2"
    assert LogRegPenalty.ELASTICNET.sklearn_penalty == "elasticnet"


def test_types_re_exports_the_same_classes() -> None:
    """covenant_ml.types re-exports these classes, not copies."""
    assert types.LogRegSolver is LogRegSolver
    assert types.LogRegPenalty is LogRegPenalty
