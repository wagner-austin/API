"""Hook protocols for scripts.amex."""

from __future__ import annotations

from typing import TypedDict


class FakeDatasetSpec(TypedDict, total=True):
    """Specification for fake dataset generation.

    Attributes:
        n_samples: Number of samples to generate.
        n_features: Number of features to generate.
        positive_ratio: Ratio of positive samples.
    """

    n_samples: int
    n_features: int
    positive_ratio: float
