from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _require_data_bank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide mandatory env vars for tests that build settings from the environment."""
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "test-key")
    monkeypatch.setenv("TURKIC_DATA_BANK_API_URL", "http://db")
