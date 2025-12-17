"""LSTM backend import and type smoke tests.

These tests validate that exported symbols exist and basic methods return
the correct types for preparation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.lstm import (
    LSTM_CAPABILITIES,
    SequenceData,
    build_sequences,
    create_lstm_backend,
    reshape_flat_to_pseudo_sequences,
)


def test_lstm_backend_exports_and_prepare() -> None:
    """Factory returns backend and prepare yields a prepared model wrapper."""
    backend = create_lstm_backend()
    assert backend.backend_name() == "lstm"
    caps = backend.capabilities()
    assert caps["model_format"] == "pt"
    assert caps["supports_gpu"] is True
    assert caps["supports_early_stopping"] is True
    assert caps["supports_feature_importance"] is False

    prepared = backend.prepare(n_features=4, n_classes=2, feature_names=["a", "b", "c", "d"])

    # Verify prepared object exists by checking its type name
    assert type(prepared).__name__ == "_LSTMPrepared"


def test_lstm_capabilities_exported() -> None:
    """LSTM_CAPABILITIES is correctly exported."""
    assert LSTM_CAPABILITIES["supports_train"] is True
    assert LSTM_CAPABILITIES["model_format"] == "pt"


def test_sequence_data_exported() -> None:
    """SequenceData TypedDict is exported from module."""
    # SequenceData should be importable and usable as a type
    from covenant_ml.backends.lstm import __all__

    assert "SequenceData" in __all__


def test_build_sequences_exported() -> None:
    """build_sequences function is exported from module."""
    from covenant_ml.backends.lstm import __all__

    assert "build_sequences" in __all__

    # Verify it's callable with correct signature
    x: NDArray[np.float64] = np.ones((6, 2), dtype=np.float64)
    y_list: list[int] = [0, 1, 0, 1, 0, 1]
    y: NDArray[np.int64] = np.array(y_list, dtype=np.int64)
    entity_list: list[int] = [1, 1, 1, 2, 2, 2]
    entity_ids: NDArray[np.int64] = np.array(entity_list, dtype=np.int64)
    years_list: list[int] = [2015, 2016, 2017, 2015, 2016, 2017]
    years: NDArray[np.int64] = np.array(years_list, dtype=np.int64)

    seq_data: SequenceData = build_sequences(
        x_features=x,
        y_labels=y,
        entity_ids=entity_ids,
        years=years,
        sequence_length=2,
    )

    assert seq_data["n_sequences"] > 0


def test_reshape_flat_to_pseudo_sequences_exported() -> None:
    """reshape_flat_to_pseudo_sequences function is exported from module."""
    from covenant_ml.backends.lstm import __all__

    assert "reshape_flat_to_pseudo_sequences" in __all__

    # Verify it's callable with correct signature
    x: NDArray[np.float64] = np.ones((4, 8), dtype=np.float64)
    result = reshape_flat_to_pseudo_sequences(x, sequence_length=4)
    assert result.shape == (4, 4, 2)


def test_create_lstm_backend_exported() -> None:
    """create_lstm_backend factory is exported from module."""
    from covenant_ml.backends.lstm import __all__

    assert "create_lstm_backend" in __all__
