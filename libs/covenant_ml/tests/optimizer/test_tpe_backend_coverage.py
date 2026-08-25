"""Tests that every backend's search space has a TPE sampler and extractor.

The TPE strategy dispatched on duck-typed key presence, and only four
backends had a branch. is_xgboost_search_space matched on "max_depth" alone,
which RandomForest and ClearGBM also carry, so both were routed into the
XGBoost sampler and failed on a learning_rate RandomForest never samples.
LogReg matched nothing and fell through to a branch that assumed LightGBM.
Optimization was therefore broken for three of the registered backends while
the API accepted all of them.

Each guard now keys off a field unique to its own space, so the dispatch is
order-independent. These tests pin that: every registered backend's default
space must route to its own sampler and round-trip through its extractor.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import math

import pytest

from covenant_ml.backends.registry import default_registry
from covenant_ml.optimizer.strategies._tpe_params import (
    _extract_best_params,
    _sample_params,
)
from covenant_ml.optimizer.type_guards import (
    is_cleargbm_search_space,
    is_lightgbm_search_space,
    is_logreg_search_space,
    is_lstm_search_space,
    is_mlp_search_space,
    is_random_forest_search_space,
    is_xgboost_search_space,
)
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    LogRegSearchSpace,
    SearchSpace,
)
from covenant_ml.types import BackendName


class _RecordingTrial:
    """Trial that records every suggestion, mirroring Optuna's interface."""

    def __init__(self) -> None:
        self._params: dict[str, float | int | str] = {}

    @property
    def number(self) -> int:
        """Trial index."""
        return 0

    def get_params(self) -> dict[str, float | int | str]:
        """Return everything suggested so far."""
        return dict(self._params)

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Return the midpoint of the range, deterministically."""
        value = (low + high) // 2
        self._params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Return the midpoint, geometric when the range is log-scaled."""
        value = math.sqrt(low * high) if log else (low + high) / 2.0
        self._params[name] = value
        return value

    def suggest_categorical(
        self,
        name: str,
        choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...],
    ) -> float | int | str:
        """Return the first choice, deterministically."""
        value = choices[0]
        self._params[name] = value
        return value


def _default_space(backend_name: BackendName) -> SearchSpace:
    """Get a backend's real default search space from the registry."""
    return default_registry().get(backend_name).get_default_search_space()


_REGISTERED: list[BackendName] = ["xgboost", "lightgbm", "cleargbm", "logreg", "random_forest"]


class TestGuardsAreMutuallyExclusive:
    """Each space matches exactly one guard.

    Order-independence is the property that matters: a dispatch whose guards
    overlap works only until someone reorders the branches.
    """

    @pytest.mark.parametrize("backend_name", _REGISTERED)
    def test_exactly_one_guard_matches(self, backend_name: BackendName) -> None:
        """No space is claimed by two guards, and none by zero."""
        space = _default_space(backend_name)

        matches = [
            name
            for name, guard in (
                ("xgboost", is_xgboost_search_space),
                ("lightgbm", is_lightgbm_search_space),
                ("cleargbm", is_cleargbm_search_space),
                ("logreg", is_logreg_search_space),
                ("random_forest", is_random_forest_search_space),
                ("mlp", is_mlp_search_space),
                ("lstm", is_lstm_search_space),
            )
            if guard(space)
        ]

        assert matches == [backend_name], f"{backend_name} matched {matches}"

    def test_random_forest_is_not_xgboost(self) -> None:
        """The specific collision that broke RandomForest optimization."""
        space = _default_space("random_forest")

        assert "max_depth" in space
        assert not is_xgboost_search_space(space)

    def test_cleargbm_is_not_xgboost(self) -> None:
        """ClearGBM carries max_depth too, and was misrouted the same way."""
        space = _default_space("cleargbm")

        assert "max_depth" in space
        assert not is_xgboost_search_space(space)


class TestSamplingRoundTrip:
    """Every registered backend samples, then extracts what it sampled."""

    @pytest.mark.parametrize("backend_name", _REGISTERED)
    def test_sample_then_extract(self, backend_name: BackendName) -> None:
        """Sampled params survive the extractor without loss.

        The extractor reads the study's best_params back into the same typed
        dicts the sampler produced, so a field sampled but not extracted would
        silently vanish from the optimization result.
        """
        space = _default_space(backend_name)
        trial = _RecordingTrial()

        int_params, float_params, string_params = _sample_params(trial, space)
        sampled_keys = set(int_params) | set(float_params) | set(string_params)
        assert sampled_keys, f"{backend_name} sampled nothing"

        extracted_int, extracted_float, extracted_string = _extract_best_params(
            space, trial.get_params()
        )
        extracted_keys = set(extracted_int) | set(extracted_float) | set(extracted_string)

        assert extracted_keys == sampled_keys

    def test_cleargbm_samples_the_coarseness_divisor(self) -> None:
        """The strategy layer samples min_data_in_bin_denom.

        When the dial landed (2026-08-25) it initially reached only the
        per-backend optimizer, not this layer — the production optimize
        path — and the first end-to-end run silently tuned without it.
        This pins the layer that actually runs.
        """
        space = _default_space("cleargbm")
        int_params, _, _ = _sample_params(_RecordingTrial(), space)
        assert int_params.get("min_data_in_bin_denom") in (1, 256, 64, 16, 4)

    def test_random_forest_samples_no_learning_rate(self) -> None:
        """RandomForest is not boosted; demanding a learning rate is the bug."""
        space = _default_space("random_forest")

        _, float_params, _ = _sample_params(_RecordingTrial(), space)

        assert "learning_rate" not in float_params

    def test_logreg_samples_its_own_params(self) -> None:
        """LogReg reaches its own sampler rather than a bare assert."""
        space = _default_space("logreg")

        int_params, float_params, _ = _sample_params(_RecordingTrial(), space)

        assert "C" in float_params
        assert "tol" in float_params
        assert "max_iter" in int_params


class TestLogRegOptionalParams:
    """LogReg's penalty, solver and l1_ratio are optional in its space.

    The default space always supplies them, so the absent path is only
    reachable through a space built without them -- which the type allows,
    since those three fields are declared total=False.
    """

    def _minimal_space(self) -> LogRegSearchSpace:
        """Build a LogReg space carrying only its required fields."""
        c_spec: FloatRangeSpec = {
            "param_type": "float",
            "low": 1e-3,
            "high": 1e3,
            "log_scale": True,
        }
        max_iter_spec: IntRangeSpec = {
            "param_type": "int",
            "low": 100,
            "high": 500,
            "log_scale": False,
        }
        tol_spec: FloatRangeSpec = {
            "param_type": "float",
            "low": 1e-6,
            "high": 1e-3,
            "log_scale": True,
        }
        return {"C": c_spec, "max_iter": max_iter_spec, "tol": tol_spec}

    def test_sampler_omits_absent_optional_params(self) -> None:
        """Nothing optional is invented when the space does not offer it."""
        int_params, float_params, string_params = _sample_params(
            _RecordingTrial(), self._minimal_space()
        )

        assert set(int_params) == {"max_iter"}
        assert set(float_params) == {"C", "tol"}
        assert string_params == {}

    def test_extractor_omits_absent_optional_params(self) -> None:
        """The extractor mirrors the sampler when the study lacks them."""
        best_params: dict[str, float | int | str] = {"C": 1.0, "max_iter": 200, "tol": 1e-4}

        int_params, float_params, string_params = _extract_best_params(
            self._minimal_space(), best_params
        )

        assert set(int_params) == {"max_iter"}
        assert set(float_params) == {"C", "tol"}
        assert string_params == {}
