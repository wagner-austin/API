from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


def make_probs(*vals: float) -> NDArray[np.float64]:
    """Create a typed numpy array for test probability values.

    This helper ensures that numpy arrays in tests have explicit types
    to satisfy strict mypy checks.
    """
    float_vals: list[np.float64] = [np.float64(v) for v in vals]
    result: NDArray[np.float64] = np.array(float_vals, dtype=np.float64)
    return result


@pytest.fixture(autouse=True)
def _require_data_bank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide mandatory env vars for tests that build settings from the environment."""
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "test-key")
    monkeypatch.setenv("TURKIC_DATA_BANK_API_URL", "http://db")
