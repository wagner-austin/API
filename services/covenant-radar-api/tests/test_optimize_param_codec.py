"""Tests for the sampled-parameter codec.

The codec's per-key presence branches are what keep a sampled
hyperparameter from silently vanishing between the optimizer summary and
the saved optimal-config record — the dataflow a decorative knob hides
in. Each key is asserted both present-and-carried and absent-and-omitted.
"""

from __future__ import annotations

from covenant_ml.optimizer.types import SampledIntParams

from covenant_radar_api.worker._optimize_param_codec import encode_sampled_int_params


class TestEncodeSampledIntParams:
    """Present keys are carried verbatim; absent keys never appear."""

    def test_coarseness_divisor_is_carried_when_sampled(self) -> None:
        """min_data_in_bin_denom reaches the record — the optimal-config
        report writes best_<key> for every carried key, so a dropped key
        here would erase the tuner's coarseness choice from the record."""
        params = SampledIntParams(
            n_estimators=100,
            max_depth=5,
            min_data_in_bin_denom=16,
        )
        encoded = encode_sampled_int_params(params)
        assert encoded["min_data_in_bin_denom"] == 16
        assert encoded["n_estimators"] == 100
        assert encoded["max_depth"] == 5

    def test_absent_keys_are_omitted(self) -> None:
        """A params dict without the divisor encodes without the key."""
        params = SampledIntParams(n_estimators=100, max_depth=5)
        encoded = encode_sampled_int_params(params)
        assert "min_data_in_bin_denom" not in encoded
