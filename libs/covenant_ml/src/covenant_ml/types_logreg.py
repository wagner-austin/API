"""Vocabularies of the logistic-regression backend.

The solver and penalty a LogRegConfig names, as enums whose values are
sklearn's own words, so a config, a saved model's metadata and the external
training parser all accept exactly the words sklearn does.
"""

from __future__ import annotations

from enum import StrEnum


class LogRegSolver(StrEnum):
    """An optimization algorithm sklearn's LogisticRegression accepts."""

    LBFGS = "lbfgs"
    LIBLINEAR = "liblinear"
    NEWTON_CG = "newton-cg"
    NEWTON_CHOLESKY = "newton-cholesky"
    SAG = "sag"
    SAGA = "saga"


class LogRegPenalty(StrEnum):
    """A regularization type a LogRegConfig may name."""

    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"
    NONE = "none"

    @property
    def sklearn_penalty(self) -> str | None:
        """The value sklearn's LogisticRegression takes for this penalty.

        Returns:
            None for NONE, which sklearn spells as the absence of a penalty;
            the member's word otherwise.
        """
        return None if self is LogRegPenalty.NONE else self.value


__all__ = ["LogRegPenalty", "LogRegSolver"]
