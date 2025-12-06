from __future__ import annotations

import pytest
from platform_core.errors import AppError

from qr_api.settings import load_default_options_from_env


def test_load_defaults_from_env_respects_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QR_DEFAULT_ERROR_CORRECTION", "H")
    monkeypatch.setenv("QR_DEFAULT_BOX_SIZE", "12")
    monkeypatch.setenv("QR_DEFAULT_BORDER", "3")
    monkeypatch.setenv("QR_DEFAULT_FILL_COLOR", "#123456")
    monkeypatch.setenv("QR_DEFAULT_BACK_COLOR", "#FEFEFE")

    defaults = load_default_options_from_env()

    assert defaults["ecc"] == "H"
    assert defaults["box_size"] == 12
    assert defaults["border"] == 3
    assert defaults["fill_color"] == "#123456"
    assert defaults["back_color"] == "#FEFEFE"


def test_load_defaults_from_env_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QR_DEFAULT_ERROR_CORRECTION", "Z")
    monkeypatch.delenv("QR_DEFAULT_BOX_SIZE", raising=False)
    monkeypatch.delenv("QR_DEFAULT_BORDER", raising=False)
    monkeypatch.delenv("QR_DEFAULT_FILL_COLOR", raising=False)
    monkeypatch.delenv("QR_DEFAULT_BACK_COLOR", raising=False)

    with pytest.raises(AppError):
        load_default_options_from_env()


def test_load_defaults_from_env_returns_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QR_DEFAULT_ERROR_CORRECTION", raising=False)
    monkeypatch.delenv("QR_DEFAULT_BOX_SIZE", raising=False)
    monkeypatch.delenv("QR_DEFAULT_BORDER", raising=False)
    monkeypatch.delenv("QR_DEFAULT_FILL_COLOR", raising=False)
    monkeypatch.delenv("QR_DEFAULT_BACK_COLOR", raising=False)

    defaults = load_default_options_from_env()

    assert defaults["ecc"] == "M"
    assert defaults["box_size"] == 10
    assert defaults["border"] == 1
    assert defaults["fill_color"] == "#000000"
    assert defaults["back_color"] == "#FFFFFF"
