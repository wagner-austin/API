"""Round-trip tests for the public inference loaders.

These let the trainer write a real checkpoint, then reload it through the
published loader. That full path is what catches a mismatch between the writer
and the reader; asserting against a hand-built state dict cannot, because the
fixture and the bug can agree with each other.

They exist because that is exactly what happened. A caller reimplemented the
LSTM architecture and derived the state-dict prefixes from the wrapper's
attribute names, so it stripped nothing, loaded no weights, and every unit
test still passed. The architectures now live here, beside the trainer.

Training happens once per module in a fixture: these are tests of loading, not
of training, and retraining per test would say nothing extra. The assertion
that carries the weight is discrimination -- the labels are a deterministic
function of two features, so a model whose weights arrived intact separates
them well above chance, while one loaded with partial or default weights
scores about 0.5.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.types import LSTMConfig, MLPConfig
from numpy.typing import NDArray

from covenant_nn.backends.lstm.backend import LSTMBackend, load_lstm_for_inference
from covenant_nn.backends.mlp.backend import MLPBackend, load_mlp_for_inference

_N_FEATURES = 13
_SEQ_LEN = 4
_HIDDEN = 8
_LAYERS = 2
_MLP_HIDDEN = (16, 8)

# A loaded-correctly model must clear this on separable data. Weights that did
# not arrive leave the model at roughly 0.5.
_MIN_AUC = 0.7


def _dataset() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a small linearly separable dataset.

    Returns:
        Feature matrix and binary labels determined by two of the features.
    """
    rng = np.random.default_rng(1234)
    x: NDArray[np.float64] = rng.normal(size=(400, _N_FEATURES)).astype(np.float64)
    y: NDArray[np.int64] = (x[:, 0] + x[:, 3] > 0).astype(np.int64)
    return x, y


def _auc(probs: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    """Area under the ROC curve for the positive-class column.

    Computed from rank statistics rather than imported from sklearn, which
    ships no type information: its returns are untyped and this package
    forbids Any. Ties are not averaged, which is immaterial for continuous
    probabilities.

    Args:
        probs: Predicted probabilities, shape (n_samples, 2).
        y: Binary labels, shape (n_samples,).

    Returns:
        AUC in [0, 1], where 0.5 is chance.
    """
    scores: NDArray[np.float64] = np.asarray(probs[:, 1], dtype=np.float64)
    n_samples = int(scores.shape[0])
    order: NDArray[np.int64] = np.argsort(scores).astype(np.int64)
    ranks: NDArray[np.float64] = np.empty(n_samples, dtype=np.float64)
    ranks[order] = np.arange(1, n_samples + 1, dtype=np.float64)

    positive: NDArray[np.bool_] = y == 1
    n_pos = int(np.count_nonzero(positive))
    n_neg = n_samples - n_pos

    # Summed through .flat rather than .sum(), which is untyped and would
    # introduce Any. This matches the indexing style used in sequences.py.
    positive_ranks: NDArray[np.float64] = np.asarray(ranks[positive], dtype=np.float64)
    rank_sum = 0.0
    for i in range(int(positive_ranks.shape[0])):
        rank_sum += float(positive_ranks.flat[i])

    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _lstm_config() -> LSTMConfig:
    """Build an LSTM training config."""
    return {
        "device": "cpu",
        "precision": "fp32",
        "hidden_size": _HIDDEN,
        "num_layers": _LAYERS,
        "dropout": 0.0,
        "bidirectional": False,
        "sequence_length": _SEQ_LEN,
        "learning_rate": 0.01,
        "batch_size": 32,
        "n_epochs": 30,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 7,
        "early_stopping_patience": 30,
    }


def _mlp_config(dropout: float) -> MLPConfig:
    """Build an MLP training config.

    Args:
        dropout: Dropout rate, which changes the Sequential layer indices.
    """
    return {
        "device": "cpu",
        "precision": "fp32",
        "optimizer": "adam",
        "hidden_sizes": _MLP_HIDDEN,
        "learning_rate": 0.01,
        "batch_size": 32,
        "n_epochs": 30,
        "dropout": dropout,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 5,
        "early_stopping_patience": 30,
    }


@pytest.fixture(scope="module")
def lstm_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train one LSTM and return the path the trainer wrote.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        Path to the written checkpoint.
    """
    x, y = _dataset()
    out_dir: Path = tmp_path_factory.mktemp("lstm")
    outcome = LSTMBackend().train(
        x_features=x,
        y_labels=y,
        feature_names=None,
        config=_lstm_config(),
        output_dir=out_dir,
        progress=None,
    )
    return outcome["model_path"]


@pytest.fixture(scope="module")
def mlp_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[float, str]:
    """Train one MLP per dropout layout and return their checkpoint paths.

    Dropout occupies a Sequential slot, so its presence shifts every later
    layer index in the state dict. Both layouts are built.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        Mapping of dropout rate to checkpoint path.
    """
    x, y = _dataset()
    paths: dict[float, str] = {}
    for dropout in (0.0, 0.2):
        out_dir: Path = tmp_path_factory.mktemp(f"mlp{int(dropout * 10)}")
        outcome = MLPBackend().train(
            x_features=x,
            y_labels=y,
            feature_names=None,
            config=_mlp_config(dropout),
            output_dir=out_dir,
            progress=None,
        )
        paths[dropout] = outcome["model_path"]
    return paths


class TestLSTMInferenceRoundTrip:
    """A trained LSTM reloads, predicts, and still discriminates.

    n_features is 13 over sequence_length 4. 13 is not divisible by 4, so the
    ceiling and floor divisions disagree and an input_size computed either way
    builds a differently shaped model.
    """

    def test_reloaded_model_discriminates(self, lstm_checkpoint: str) -> None:
        """Weights survive the round trip, so the model still separates."""
        x, y = _dataset()

        predictor = load_lstm_for_inference(
            path=lstm_checkpoint,
            n_features=_N_FEATURES,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            dropout=0.0,
            bidirectional=False,
            sequence_length=_SEQ_LEN,
        )
        probs = predictor.predict_proba(x)

        assert probs.shape == (x.shape[0], 2)
        assert bool(np.isfinite(probs).all())
        assert _auc(probs, y) > _MIN_AUC

    def test_reload_is_deterministic(self, lstm_checkpoint: str) -> None:
        """Loading the same checkpoint twice yields identical predictions."""
        x, _ = _dataset()

        first = load_lstm_for_inference(
            path=lstm_checkpoint,
            n_features=_N_FEATURES,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            dropout=0.0,
            bidirectional=False,
            sequence_length=_SEQ_LEN,
        ).predict_proba(x[:16])
        second = load_lstm_for_inference(
            path=lstm_checkpoint,
            n_features=_N_FEATURES,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            dropout=0.0,
            bidirectional=False,
            sequence_length=_SEQ_LEN,
        ).predict_proba(x[:16])

        assert np.array_equal(first, second)

    def test_gradients_are_finite(self, lstm_checkpoint: str) -> None:
        """The reloaded model supports the gradient explainers."""
        x, _ = _dataset()

        predictor = load_lstm_for_inference(
            path=lstm_checkpoint,
            n_features=_N_FEATURES,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            dropout=0.0,
            bidirectional=False,
            sequence_length=_SEQ_LEN,
        )
        grads = predictor.compute_gradients(x[:4], 1)

        assert grads.shape == (4, _N_FEATURES)
        assert bool(np.isfinite(grads).all())

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """A path with no checkpoint fails rather than returning an empty model."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_lstm_for_inference(
                path=str(tmp_path / "absent.pt"),
                n_features=_N_FEATURES,
                hidden_size=_HIDDEN,
                num_layers=_LAYERS,
                dropout=0.0,
                bidirectional=False,
                sequence_length=_SEQ_LEN,
            )


class TestMLPInferenceRoundTrip:
    """A trained MLP reloads and still discriminates, under both layouts."""

    @pytest.mark.parametrize("dropout", [0.0, 0.2])
    def test_reloaded_model_discriminates(
        self,
        mlp_checkpoints: dict[float, str],
        dropout: float,
    ) -> None:
        """Both dropout layouts round-trip with their weights intact.

        A misaligned stack would load keys onto the wrong layers, or none at
        all, and the model would sit near chance.
        """
        x, y = _dataset()

        predictor = load_mlp_for_inference(
            path=mlp_checkpoints[dropout],
            n_features=_N_FEATURES,
            hidden_sizes=_MLP_HIDDEN,
            dropout=dropout,
        )
        probs = predictor.predict_proba(x)

        assert probs.shape == (x.shape[0], 2)
        assert bool(np.isfinite(probs).all())
        assert _auc(probs, y) > _MIN_AUC

    def test_gradients_are_finite(self, mlp_checkpoints: dict[float, str]) -> None:
        """The reloaded model supports the gradient explainers."""
        x, _ = _dataset()

        predictor = load_mlp_for_inference(
            path=mlp_checkpoints[0.0],
            n_features=_N_FEATURES,
            hidden_sizes=_MLP_HIDDEN,
            dropout=0.0,
        )
        grads = predictor.compute_gradients(x[:4], 1)

        assert grads.shape == (4, _N_FEATURES)
        assert bool(np.isfinite(grads).all())

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """A path with no checkpoint fails rather than returning an empty model."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_mlp_for_inference(
                path=str(tmp_path / "absent.pt"),
                n_features=_N_FEATURES,
                hidden_sizes=_MLP_HIDDEN,
                dropout=0.0,
            )
