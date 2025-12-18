from __future__ import annotations

import pytest

from procart.config_decode import decode_tone_mapping
from procart.types import Scalar


def test_decode_tone_exposure_gamma_ok() -> None:
    raw: dict[str, Scalar] = {"type": "exposure_gamma", "exposure": 1.2, "gamma": 2.2}
    cfg = decode_tone_mapping(raw)
    assert cfg["type"] == "exposure_gamma"
    assert cfg["exposure"] == pytest.approx(1.2)


def test_decode_tone_reinhard_ok() -> None:
    raw: dict[str, Scalar] = {"type": "reinhard", "exposure": 0.8}
    cfg = decode_tone_mapping(raw)
    assert cfg["type"] == "reinhard"
    assert cfg["exposure"] == pytest.approx(0.8)


def test_decode_tone_filmic_ok() -> None:
    raw: dict[str, Scalar] = {"type": "filmic", "exposure": 1.0}
    cfg = decode_tone_mapping(raw)
    assert cfg["type"] == "filmic"
    assert cfg["exposure"] == pytest.approx(1.0)


def test_decode_tone_missing_type_raises() -> None:
    with pytest.raises(ValueError):
        decode_tone_mapping({"exposure": 1.0})


def test_decode_tone_filmic_missing_exposure_raises() -> None:
    with pytest.raises(ValueError):
        decode_tone_mapping({"type": "filmic"})


def test_decode_tone_type_wrong_type_raises() -> None:
    with pytest.raises(ValueError):
        bad: dict[str, Scalar] = {"type": 123, "exposure": 1.0}
        decode_tone_mapping(bad)


def test_decode_tone_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        decode_tone_mapping({"type": "unknown", "exposure": 1.0})


def test_decode_tone_exposure_gamma_missing_keys_raises() -> None:
    with pytest.raises(ValueError):
        decode_tone_mapping({"type": "exposure_gamma", "exposure": 1.0})


def test_decode_tone_reinhard_missing_exposure_raises() -> None:
    with pytest.raises(ValueError):
        decode_tone_mapping({"type": "reinhard"})


def test_decode_tone_filmic_exposure_non_number_raises() -> None:
    with pytest.raises(ValueError):
        bad: dict[str, Scalar] = {"type": "filmic", "exposure": "x"}
        decode_tone_mapping(bad)


def test_decode_tone_exposure_gamma_rejects_bool_as_number() -> None:
    with pytest.raises(ValueError):
        bad: dict[str, Scalar] = {"type": "exposure_gamma", "exposure": True, "gamma": 2.2}
        decode_tone_mapping(bad)


def test_decode_tone_exposure_gamma_gamma_non_number_raises() -> None:
    bad: dict[str, Scalar] = {"type": "exposure_gamma", "exposure": 1.0, "gamma": "x"}
    with pytest.raises(ValueError):
        decode_tone_mapping(bad)
