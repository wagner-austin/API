from __future__ import annotations

from typing import Final, Protocol

from .color import apply_tone_map
from .math_backend import FloatArray
from .types import (
    ToneMappingConfigExposureGamma,
    ToneMappingConfigFilmic,
    ToneMappingConfigReinhard,
)


class ToneMapper(Protocol):
    """Apply tone mapping in-place style returning a new array.

    Args:
        rgb_linear: FloatArray-like, values >= 0 in linear space.

    Returns:
        Same-shape float array with values in [0, 1].
    """

    def __call__(self, rgb_linear: FloatArray) -> FloatArray: ...


_NAMES: Final[tuple[str, ...]] = ("exposure_gamma", "reinhard", "filmic")


def list_available_tone_mappers() -> list[str]:
    return list(_NAMES)


def get_tone_mapper(name: str) -> ToneMapper:
    # Build a closure around apply_tone_map with a fixed config discriminator
    def _tm(rgb_linear: FloatArray) -> FloatArray:
        if name == "exposure_gamma":
            cfg: ToneMappingConfigExposureGamma = {
                "type": "exposure_gamma",
                "exposure": 1.0,
                "gamma": 2.2,
            }
            return apply_tone_map(rgb_linear, cfg)
        if name == "reinhard":
            cfg2: ToneMappingConfigReinhard = {"type": "reinhard", "exposure": 1.0}
            return apply_tone_map(rgb_linear, cfg2)
        cfg3: ToneMappingConfigFilmic = {"type": "filmic", "exposure": 1.0}
        return apply_tone_map(rgb_linear, cfg3)

    if name in _NAMES:
        return _tm
    raise ValueError(f"unknown tone mapper: {name}")


__all__ = [
    "ToneMapper",
    "get_tone_mapper",
    "list_available_tone_mappers",
]
