from __future__ import annotations

from typing import Final

from .types import (
    Scalar,
    ToneMappingConfig,
    ToneMappingConfigExposureGamma,
    ToneMappingConfigFilmic,
    ToneMappingConfigReinhard,
)

_SUPPORTED_TONE_TYPES: Final[set[str]] = {"exposure_gamma", "reinhard", "filmic"}


def _as_float(value: Scalar, *, name: str) -> float:
    # Accept ints/floats only; reject bools and strings to keep semantics strict.
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{name} must be a number")


def decode_tone_mapping(raw: dict[str, Scalar]) -> ToneMappingConfig:
    """Decode a raw Python dict into a strict ToneMappingConfig.

    Args:
        raw: Untyped mapping parsed at the boundary (e.g., Python builder or JSON).

    Returns:
        ToneMappingConfig: A strictly typed tone mapping configuration.

    Raises:
        ValueError: If required keys are missing, values have wrong types,
            or the tone mapping type is unsupported.
    """
    if "type" not in raw:
        raise ValueError("tone_mapping.type is required")
    type_val = raw["type"]
    if not isinstance(type_val, str):
        raise ValueError("tone_mapping.type must be a string")
    if type_val not in _SUPPORTED_TONE_TYPES:
        raise ValueError(f"unsupported tone mapping type: {type_val}")

    if type_val == "exposure_gamma":
        if "exposure" not in raw or "gamma" not in raw:
            raise ValueError("exposure_gamma requires 'exposure' and 'gamma'")
        cfg: ToneMappingConfigExposureGamma = {
            "type": "exposure_gamma",
            "exposure": _as_float(raw["exposure"], name="exposure"),
            "gamma": _as_float(raw["gamma"], name="gamma"),
        }
        return cfg

    if type_val == "reinhard":
        if "exposure" not in raw:
            raise ValueError("reinhard requires 'exposure'")
        cfg2: ToneMappingConfigReinhard = {
            "type": "reinhard",
            "exposure": _as_float(raw["exposure"], name="exposure"),
        }
        return cfg2

    # type_val == "filmic"
    if "exposure" not in raw:
        raise ValueError("filmic requires 'exposure'")
    cfg3: ToneMappingConfigFilmic = {
        "type": "filmic",
        "exposure": _as_float(raw["exposure"], name="exposure"),
    }
    return cfg3


__all__ = ["decode_tone_mapping"]
