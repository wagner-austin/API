"""Tests that every neural backend exposes a real hyperparameter search space.

Hyperparameter optimization calls get_default_search_space() on the selected
backend. Before these were wired, the classifier backends inherited the
Protocol's empty stub by subclassing ClassifierBackend and returned None, while
the regressor backends had no such attribute at all. mypy reported only the
regressors, because subclassing a Protocol makes its `...` body count as an
implementation.

Each backend is asserted to delegate to the canonical space factory in
covenant_ml. The ranges themselves are that package's contract and are tested
there; what is verified here is the wiring that was missing.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.search_spaces import (
    make_lstm_default_space,
    make_lstm_focused_space,
    make_mlp_default_space,
    make_mlp_focused_space,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams

from covenant_nn.backends.lstm.backend import LSTMBackend
from covenant_nn.backends.lstm.regressor import LSTMRegressorBackend
from covenant_nn.backends.mlp.backend import MLPBackend
from covenant_nn.backends.mlp.regressor import MLPRegressorBackend

_MLP_BEST_INT = SampledIntParams(n_layers=2, hidden_size=128)
_LSTM_BEST_INT = SampledIntParams(hidden_size=128, num_layers=2)
_BEST_FLOAT = SampledFloatParams(learning_rate=0.001)


class TestDefaultSearchSpaces:
    """Every neural backend returns the canonical default space."""

    def test_mlp_classifier(self) -> None:
        """MLPBackend returns the MLP space, not the Protocol stub's None."""
        assert MLPBackend().get_default_search_space() == make_mlp_default_space()

    def test_lstm_classifier(self) -> None:
        """LSTMBackend returns the LSTM space, not the Protocol stub's None."""
        assert LSTMBackend().get_default_search_space() == make_lstm_default_space()

    def test_mlp_regressor(self) -> None:
        """MLPRegressorBackend has the method at all now, and delegates."""
        assert MLPRegressorBackend().get_default_search_space() == make_mlp_default_space()

    def test_lstm_regressor(self) -> None:
        """LSTMRegressorBackend has the method at all now, and delegates."""
        assert LSTMRegressorBackend().get_default_search_space() == make_lstm_default_space()


class TestFocusedSearchSpaces:
    """Every neural backend narrows around prior best params."""

    def test_mlp_classifier(self) -> None:
        """MLP forwards n_layers, hidden_size and learning_rate."""
        space = MLPBackend().get_focused_search_space(
            best_int_params=_MLP_BEST_INT,
            best_float_params=_BEST_FLOAT,
        )

        assert space == make_mlp_focused_space(
            best_n_layers=2,
            best_hidden_size=128,
            best_learning_rate=0.001,
        )

    def test_lstm_classifier(self) -> None:
        """LSTM forwards hidden_size, num_layers and learning_rate."""
        space = LSTMBackend().get_focused_search_space(
            best_int_params=_LSTM_BEST_INT,
            best_float_params=_BEST_FLOAT,
        )

        assert space == make_lstm_focused_space(
            best_hidden_size=128,
            best_num_layers=2,
            best_learning_rate=0.001,
        )

    def test_mlp_regressor(self) -> None:
        """The regressor mirrors the classifier's narrowing."""
        space = MLPRegressorBackend().get_focused_search_space(
            best_int_params=_MLP_BEST_INT,
            best_float_params=_BEST_FLOAT,
        )

        assert space == make_mlp_focused_space(
            best_n_layers=2,
            best_hidden_size=128,
            best_learning_rate=0.001,
        )

    def test_lstm_regressor(self) -> None:
        """The regressor mirrors the classifier's narrowing."""
        space = LSTMRegressorBackend().get_focused_search_space(
            best_int_params=_LSTM_BEST_INT,
            best_float_params=_BEST_FLOAT,
        )

        assert space == make_lstm_focused_space(
            best_hidden_size=128,
            best_num_layers=2,
            best_learning_rate=0.001,
        )

    def test_focused_differs_from_default(self) -> None:
        """Focusing actually narrows, rather than returning the default.

        If these were equal, a second optimization pass would just repeat the
        first and "focused" would be a no-op.
        """
        backend = MLPBackend()

        default = backend.get_default_search_space()
        focused = backend.get_focused_search_space(
            best_int_params=_MLP_BEST_INT,
            best_float_params=_BEST_FLOAT,
        )

        assert focused != default
