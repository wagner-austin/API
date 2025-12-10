from __future__ import annotations

from datetime import UTC, datetime

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from platform_ml.manifest import MANIFEST_SCHEMA_VERSION, from_json_manifest_v2


def _base_manifest_dict() -> dict[str, JSONValue]:
    now = datetime.now(UTC).isoformat()
    tr: dict[str, JSONValue] = {
        "run_id": "r-1",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "seed": 123,
        "device": "cpu",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "augment": False,
    }
    d: dict[str, JSONValue] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_type": "resnet18",
        "model_id": "m-1",
        "created_at": now,
        "arch": "resnet18",
        "n_classes": 10,
        "val_acc": 0.9,
        "preprocess_hash": "abc",
        "file_id": "m-1-r-1.tar.gz",
        "file_size": 1234,
        "file_sha256": "deadbeef",
        "training": tr,
    }
    return d


def test_manifest_v2_roundtrip() -> None:
    d = _base_manifest_dict()
    s = dump_json_str(d)
    m = from_json_manifest_v2(s)
    assert type(m) is dict
    assert m["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert m["model_type"] == "resnet18"
    assert m["model_id"] == "m-1"
    assert m["arch"] == "resnet18"
    assert m["file_id"] == "m-1-r-1.tar.gz"
    assert m["file_size"] == 1234
    assert m["file_sha256"] == "deadbeef"
    assert m["training"]["run_id"] == "r-1"


@pytest.mark.parametrize(
    "key,new_val",
    [
        ("schema_version", "v1.1"),
        ("model_type", "unknown"),
        ("model_id", ""),
        ("created_at", "not-iso"),
        ("arch", ""),
        ("file_id", ""),
        ("file_size", 0),
        ("file_sha256", ""),
    ],
)
def test_manifest_v2_field_validation_errors(key: str, new_val: JSONValue) -> None:
    d = _base_manifest_dict()
    assert key in d
    d[key] = new_val
    with pytest.raises(JSONTypeError):
        from_json_manifest_v2(dump_json_str(d))


def test_training_metadata_validation() -> None:
    d = _base_manifest_dict()
    # Replace training with a malformed object (epochs as string)
    d["training"] = {
        "run_id": "r-1",
        "epochs": "5",  # invalid type (str)
        "batch_size": 32,
        "learning_rate": 0.001,
        "seed": 123,
        "device": "cpu",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "augment": False,
    }
    with pytest.raises(JSONTypeError):
        from_json_manifest_v2(dump_json_str(d))
