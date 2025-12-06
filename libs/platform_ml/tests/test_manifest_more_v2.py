from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, dump_json_str

from platform_ml.manifest import from_json_manifest_v2, from_path_manifest_v2


def _make_valid_training() -> dict[str, JSONValue]:
    return {
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


def _make_valid_manifest() -> dict[str, JSONValue]:
    return {
        "schema_version": "v2.0",
        "model_type": "resnet18",
        "model_id": "m-1",
        "created_at": "2024-01-01T00:00:00+00:00",
        "arch": "resnet18",
        "n_classes": 10,
        "val_acc": 0.9,
        "preprocess_hash": "abc",
        "file_id": "m-1-r-1.tar.gz",
        "file_size": 1234,
        "file_sha256": "deadbeef",
        "training": _make_valid_training(),
    }


def test_manifest_v2_from_json_requires_object() -> None:
    with pytest.raises(ValueError):
        from_json_manifest_v2("[]")


# Coverage for line 31: _optional_int_ge raises when key is not int
def test_manifest_v2_n_classes_must_be_int() -> None:
    d = _make_valid_manifest()
    d["n_classes"] = "not-an-int"
    with pytest.raises(ValueError, match="n_classes must be int"):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for line 33: _optional_int_ge raises when int < min_value
def test_manifest_v2_n_classes_below_minimum() -> None:
    d = _make_valid_manifest()
    d["n_classes"] = 1  # min is 2
    with pytest.raises(ValueError, match="n_classes must be >="):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for line 44: _optional_float_range raises when value not number
def test_manifest_v2_val_acc_must_be_number() -> None:
    d = _make_valid_manifest()
    d["val_acc"] = "not-a-number"
    with pytest.raises(ValueError, match="val_acc must be number"):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for lines 47-48: _optional_float_range raises when val < min (unbounded max)
def test_manifest_v2_val_loss_below_minimum() -> None:
    d = _make_valid_manifest()
    d["val_loss"] = -0.5  # val_loss has min=0.0, max=None (unbounded)
    with pytest.raises(ValueError, match="val_loss must be >="):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for line 51: _optional_float_range raises when out of [min, max] range
def test_manifest_v2_val_acc_out_of_range() -> None:
    d = _make_valid_manifest()
    d["val_acc"] = 1.5  # val_acc must be in [0.0, 1.0]
    with pytest.raises(ValueError, match="val_acc must be within"):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for line 60: _optional_nonempty_str raises when present but empty
def test_manifest_v2_preprocess_hash_non_empty_if_present() -> None:
    d = _make_valid_manifest()
    d["preprocess_hash"] = ""
    with pytest.raises(ValueError, match="preprocess_hash must be non-empty if present"):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for line 152: training must be object
def test_manifest_v2_training_must_be_object() -> None:
    d = _make_valid_manifest()
    d["training"] = []  # not an object
    with pytest.raises(ValueError, match="training must be object"):
        from_json_manifest_v2(dump_json_str(d))


# Coverage for lines 174, 178: vocab_size and val_loss branches (gpt2 model)
def test_manifest_v2_gpt2_with_vocab_size_and_val_loss() -> None:
    d: dict[str, JSONValue] = {
        "schema_version": "v2.0",
        "model_type": "gpt2",
        "model_id": "gpt2-m",
        "created_at": "2024-01-01T00:00:00+00:00",
        "arch": "gpt2-small",
        "vocab_size": 50257,  # covers line 174
        "val_loss": 2.5,  # covers line 178
        "file_id": "gpt2-m.tar.gz",
        "file_size": 999,
        "file_sha256": "abc123",
        "training": _make_valid_training(),
    }
    m = from_json_manifest_v2(dump_json_str(d))
    assert m["model_type"] == "gpt2"
    assert m["vocab_size"] == 50257
    assert m["val_loss"] == 2.5


# Additional: vocab_size must be int type
def test_manifest_v2_vocab_size_must_be_int() -> None:
    d: dict[str, JSONValue] = {
        "schema_version": "v2.0",
        "model_type": "gpt2",
        "model_id": "gpt2-m",
        "created_at": "2024-01-01T00:00:00+00:00",
        "arch": "gpt2-small",
        "vocab_size": "not-int",
        "file_id": "gpt2-m.tar.gz",
        "file_size": 999,
        "file_sha256": "abc123",
        "training": _make_valid_training(),
    }
    with pytest.raises(ValueError, match="vocab_size must be int"):
        from_json_manifest_v2(dump_json_str(d))


def test_manifest_v2_from_path(tmp_path: Path) -> None:
    d = {
        "schema_version": "v2.0",
        "model_type": "resnet18",
        "model_id": "m",
        "created_at": "2024-01-01T00:00:00+00:00",
        "arch": "resnet18",
        "n_classes": 10,
        "file_id": "m.tgz",
        "file_size": 1,
        "file_sha256": "x",
        "training": {
            "run_id": "2024-01-01T00:00:00+00:00",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "seed": 1,
            "device": "cpu",
            "optimizer": "adamw",
            "scheduler": "cosine",
            "augment": False,
        },
    }
    p = tmp_path / "man.json"
    p.write_text(dump_json_str(d), encoding="utf-8")
    m = from_path_manifest_v2(p)
    assert m["model_id"] == "m"


def test_manifest_v2_from_path_requires_object(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = from_path_manifest_v2(p)
