from __future__ import annotations

from .api import _test_hooks
from .types import ECCLevel
from .validators import (
    Defaults,
    _validate_border,
    _validate_box_size,
    _validate_ecc,
    _validate_hex_color,
)

_DEFAULT_ECC: ECCLevel = "M"
_DEFAULT_BOX_SIZE = 10
_DEFAULT_BORDER = 1
_DEFAULT_FILL = "#000000"
_DEFAULT_BACK = "#FFFFFF"


def _get_str(key: str, default: str) -> str:
    """Get string config value via hook, with default."""
    val = _test_hooks.get_env(key)
    return val if val is not None else default


def _get_int(key: str, default: int) -> int:
    """Get int config value via hook, with default."""
    val = _test_hooks.get_env(key)
    if val is None:
        return default
    return int(val)


def load_default_options_from_env() -> Defaults:
    """Load and validate QR defaults from environment variables."""
    ecc_raw = _get_str("QR_DEFAULT_ERROR_CORRECTION", _DEFAULT_ECC)
    box_raw = _get_int("QR_DEFAULT_BOX_SIZE", _DEFAULT_BOX_SIZE)
    border_raw = _get_int("QR_DEFAULT_BORDER", _DEFAULT_BORDER)
    fill_raw = _get_str("QR_DEFAULT_FILL_COLOR", _DEFAULT_FILL)
    back_raw = _get_str("QR_DEFAULT_BACK_COLOR", _DEFAULT_BACK)

    ecc = _validate_ecc(ecc_raw, _DEFAULT_ECC)
    box = _validate_box_size(box_raw, _DEFAULT_BOX_SIZE)
    border = _validate_border(border_raw, _DEFAULT_BORDER)
    fill = _validate_hex_color(fill_raw, _DEFAULT_FILL)
    back = _validate_hex_color(back_raw, _DEFAULT_BACK)

    return {
        "ecc": ecc,
        "box_size": box,
        "border": border,
        "fill_color": fill,
        "back_color": back,
    }


__all__ = ["load_default_options_from_env"]
