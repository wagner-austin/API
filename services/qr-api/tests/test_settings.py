from __future__ import annotations

import pytest
from platform_core.errors import AppError

from qr_api.api import _test_hooks
from qr_api.settings import load_default_options_from_env


def test_load_defaults_from_env_respects_overrides() -> None:
    env_values = {
        "QR_DEFAULT_ERROR_CORRECTION": "H",
        "QR_DEFAULT_BOX_SIZE": "12",
        "QR_DEFAULT_BORDER": "3",
        "QR_DEFAULT_FILL_COLOR": "#123456",
        "QR_DEFAULT_BACK_COLOR": "#FEFEFE",
    }
    _test_hooks.get_env = lambda key: env_values.get(key)

    defaults = load_default_options_from_env()

    assert defaults["ecc"] == "H"
    assert defaults["box_size"] == 12
    assert defaults["border"] == 3
    assert defaults["fill_color"] == "#123456"
    assert defaults["back_color"] == "#FEFEFE"


def test_load_defaults_from_env_rejects_invalid() -> None:
    env_values = {"QR_DEFAULT_ERROR_CORRECTION": "Z"}
    _test_hooks.get_env = lambda key: env_values.get(key)

    with pytest.raises(AppError):
        load_default_options_from_env()


def test_load_defaults_from_env_returns_defaults() -> None:
    _test_hooks.get_env = lambda key: None

    defaults = load_default_options_from_env()

    assert defaults["ecc"] == "M"
    assert defaults["box_size"] == 10
    assert defaults["border"] == 1
    assert defaults["fill_color"] == "#000000"
    assert defaults["back_color"] == "#FFFFFF"
