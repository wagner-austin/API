from __future__ import annotations

from datetime import UTC, datetime

import pytest
from platform_core.json_utils import JSONTypeError

from handwriting_ai.inference.manifest import _decode_manifest

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_manifest_from_dict_valid() -> None:
    d: dict[str, UnknownJson] = {
        "schema_version": "v1.1",
        "model_id": "mnist_resnet18_v1",
        "arch": "resnet18_cifar",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    man = _decode_manifest(d)
    assert man["model_id"] == "mnist_resnet18_v1"
    assert man["n_classes"] == 10


def test_manifest_from_dict_missing_raises() -> None:
    bad: dict[str, UnknownJson] = {
        "schema_version": "v1.1",
        # missing model_id
        "arch": "resnet18_cifar",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 0.98,
        "temperature": 1.0,
    }
    with pytest.raises(JSONTypeError):
        _ = _decode_manifest(bad)
