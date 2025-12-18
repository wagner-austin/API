from __future__ import annotations

import pytest

from procart.color import apply_tone_map
from procart.config_decode import decode_tone_mapping
from procart.math_backend import BACKEND
from procart.types import (
    ToneMappingConfigFilmic,
    ToneMappingConfigReinhard,
)


def _flatten_list(vals: list[float] | list[list[float]]) -> list[float]:
    out: list[float] = []
    for item in vals:
        if isinstance(item, list):
            for v in item:
                out.append(float(v))
        else:
            out.append(float(item))
    return out


def test_tone_map_reinhard() -> None:
    cfg: ToneMappingConfigReinhard = {"type": "reinhard", "exposure": 1.0}
    arr = BACKEND.from_list([0.0, 1.0, 10.0])
    out = apply_tone_map(arr, cfg)
    vals = _flatten_list(out.tolist())
    assert vals[0] == 0.0
    assert vals[1] < 1.0
    assert vals[2] < 1.0


def test_tone_map_filmic() -> None:
    cfg: ToneMappingConfigFilmic = {"type": "filmic", "exposure": 1.0}
    arr = BACKEND.from_list([0.1, 0.5, 0.9])
    out = apply_tone_map(arr, cfg)
    vals = _flatten_list(out.tolist())
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_tone_map_unsupported_raises() -> None:
    # Exercise invalid selector via decoder (properly typed path without ignores)
    with pytest.raises(ValueError):
        decode_tone_mapping({"type": "unknown", "exposure": 1.0})
