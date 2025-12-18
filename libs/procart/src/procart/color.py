from __future__ import annotations

from .math_backend import BACKEND, FloatArray
from .types import (
    ToneMappingConfig,
    ToneMappingConfigExposureGamma,
    ToneMappingConfigFilmic,
    ToneMappingConfigReinhard,
)


def hsv_to_rgb(h: float, s: float, v: float) -> FloatArray:
    """Convert HSV to linear RGB.

    Args:
        h: Hue in [0, 1].
        s: Saturation in [0, 1].
        v: Value/brightness (linear). May exceed 1.0.

    Returns:
        np.ndarray: RGB vector of shape (3,), dtype float32, in linear space.

    Raises:
        ValueError: If hue or saturation is outside [0, 1].
    """
    if not (0.0 <= h <= 1.0 and 0.0 <= s <= 1.0):
        raise ValueError("h and s must be within [0,1]")
    h6 = (h * 6.0) % 6.0
    i = int(h6)
    f = h6 - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return BACKEND.array3(r, g, b)


def _apply_tone_map_exposure_gamma(
    rgb_linear: FloatArray, cfg: ToneMappingConfigExposureGamma
) -> FloatArray:
    exposure = float(cfg["exposure"])  # scale
    gamma = float(cfg["gamma"])  # gamma exponent
    scaled = rgb_linear * exposure
    compressed = scaled / (1.0 + BACKEND.maximum_scalar(scaled, 0.0))
    clamped = BACKEND.clip(compressed, 0.0, 1.0)
    inv_gamma = 1.0 / gamma
    return BACKEND.power(clamped, inv_gamma)


def _apply_tone_map_reinhard(rgb_linear: FloatArray, cfg: ToneMappingConfigReinhard) -> FloatArray:
    exposure = float(cfg["exposure"])  # scale
    x = rgb_linear * exposure
    out = x / (1.0 + BACKEND.maximum_scalar(x, 0.0))
    return BACKEND.clip(out, 0.0, 1.0)


def _apply_tone_map_filmic(rgb_linear: FloatArray, cfg: ToneMappingConfigFilmic) -> FloatArray:
    exposure = float(cfg["exposure"])  # scale
    x = BACKEND.clip(rgb_linear * exposure, 0.0, 1e9)
    a = BACKEND.clip(x * 0.9 + 0.1, 0.0, 1.0)
    out = a * (2.0 - a)
    return BACKEND.clip(out, 0.0, 1.0)


def apply_tone_map(rgb_linear: FloatArray, config: ToneMappingConfig) -> FloatArray:
    """Dispatch tone mapping based on the tagged config type.

    Args:
        rgb_linear: Float32 RGB array (H,W,3) or (3,), linear domain.
        config: Tone mapping configuration union with "type" discriminator.

    Returns:
        np.ndarray: Same shape as input, clamped to [0,1] and gamma-corrected when applicable.
    """
    if config["type"] == "exposure_gamma":
        cfg: ToneMappingConfigExposureGamma = config
        return _apply_tone_map_exposure_gamma(rgb_linear, cfg)
    if config["type"] == "reinhard":
        cfg2: ToneMappingConfigReinhard = config
        return _apply_tone_map_reinhard(rgb_linear, cfg2)
    # Remaining supported type is filmic (validated upstream by decoders).
    cfg3: ToneMappingConfigFilmic = config
    return _apply_tone_map_filmic(rgb_linear, cfg3)


__all__ = ["apply_tone_map", "hsv_to_rgb"]
