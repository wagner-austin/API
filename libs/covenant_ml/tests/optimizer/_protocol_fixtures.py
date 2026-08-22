"""Shared fixtures and helpers for test_protocol splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)


class _ConcreteObjective:
    """Concrete implementation of ObjectiveProtocol."""

    def __init__(self, return_value: float = 0.85) -> None:
        self._return_value = return_value
        self.call_count = 0

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        _ = (
            x_features,
            y_labels,
            feature_names,
            int_params,
            float_params,
            string_params,
            train_ratio,
            val_ratio,
            test_ratio,
            random_state,
        )
        self.call_count += 1
        return self._return_value
