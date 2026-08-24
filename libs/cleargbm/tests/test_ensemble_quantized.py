"""End-to-end tests for ``quantized_gradient_bins`` at the Python surface.

The knob crosses the boundary as config field 24; these tests hold the
Python-visible contract: a quantized run trains and predicts, is
deterministic per config, and a coarse quantization actually changes the
model relative to the float path (config honesty).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm.ensemble import predict_raw, train_gradient_boosting
from tests.conftest import make_config


def _make_noisy_dataset() -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    """Return a 60-row binary dataset with label noise.

    Noise keeps split gains close together, so gradient coarsening can
    reorder them — the property the knob-honesty test needs.
    """
    rng = np.random.default_rng(7)
    x: NDArray[np.float64] = rng.random((60, 3), dtype=np.float64)
    flips: NDArray[np.float64] = rng.random(60)
    base: NDArray[np.bool_] = x[:, 0] + 0.4 * x[:, 1] > 0.7
    noisy: NDArray[np.bool_] = np.logical_xor(base, flips < 0.25)
    y: NDArray[np.int64] = noisy.astype(np.int64)
    return x, y, ("f0", "f1", "f2")


class TestQuantizedTraining:
    """The quantized knob at the ensemble surface."""

    def test_quantized_run_trains_and_predicts(self) -> None:
        """A quantized config trains and predicts one score per row."""
        x, y, names = _make_noisy_dataset()
        config = make_config(n_estimators=8, quantized_gradient_bins=4)
        model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=config, feature_names=names
        )
        scores = predict_raw(model, x)
        assert scores.shape == (60,)
        assert bool(np.isfinite(scores).all())

    def test_quantized_training_is_deterministic(self) -> None:
        """Two runs under one quantized config score identically."""
        x, y, names = _make_noisy_dataset()
        config = make_config(n_estimators=8, quantized_gradient_bins=4)
        first = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=config, feature_names=names
        )
        second = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=config, feature_names=names
        )
        assert np.array_equal(predict_raw(first, x), predict_raw(second, x))

    def test_coarse_quantization_changes_the_model(self) -> None:
        """Config honesty: 2-bin quantization must not be decorative."""
        x, y, names = _make_noisy_dataset()
        float_config = make_config(n_estimators=8, max_depth=4)
        quant_config = make_config(n_estimators=8, max_depth=4, quantized_gradient_bins=2)
        float_model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=float_config, feature_names=names
        )
        quant_model = train_gradient_boosting(
            x_train=x, y_train=y, x_val=None, y_val=None, config=quant_config, feature_names=names
        )
        assert not np.array_equal(predict_raw(float_model, x), predict_raw(quant_model, x))
