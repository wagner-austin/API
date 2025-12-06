from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from handwriting_ai.inference.manifest import (
    _decode_manifest,
    from_json_manifest,
    from_path_manifest,
)

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_manifest_from_json_invalid_json_raises() -> None:
    with pytest.raises(ValueError):
        _ = from_json_manifest("{invalid}")


def test_manifest_from_json_non_object_raises() -> None:
    with pytest.raises(ValueError):
        _ = from_json_manifest("[]")


def test_manifest_invalid_created_at_raises() -> None:
    d: dict[str, UnknownJson] = {
        "schema_version": "v1.1",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": "not-a-date",
        "preprocess_hash": "abc",
        "val_acc": 0.5,
        "temperature": 1.0,
    }
    with pytest.raises(ValueError):
        _ = _decode_manifest(d)


def test_manifest_invalid_n_classes_bound_raises() -> None:
    d: dict[str, UnknownJson] = {
        "schema_version": "v1.1",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 1,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 0.5,
        "temperature": 1.0,
    }
    with pytest.raises(ValueError):
        _ = _decode_manifest(d)


def test_manifest_invalid_val_acc_raises() -> None:
    d: dict[str, UnknownJson] = {
        "schema_version": "v1.1",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 2.0,
        "temperature": 1.0,
    }
    with pytest.raises(ValueError):
        _ = _decode_manifest(d)


def test_manifest_invalid_temperature_raises() -> None:
    d: dict[str, UnknownJson] = {
        "schema_version": "v1.1",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 0.5,
        "temperature": 0.0,
    }
    with pytest.raises(ValueError):
        _ = _decode_manifest(d)


def test_manifest_required_fields_missing_schema_raises() -> None:
    d: dict[str, UnknownJson] = {
        # "schema_version" missing
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 0.5,
        "temperature": 1.0,
    }
    with pytest.raises(ValueError):
        _ = _decode_manifest(d)


def test_manifest_unsupported_schema_version_raises() -> None:
    d: dict[str, UnknownJson] = {
        "schema_version": "v9",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 0.5,
        "temperature": 1.0,
    }
    with pytest.raises(ValueError):
        _ = _decode_manifest(d)


def test_manifest_from_path_invalid_json_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.json"
        p.write_text("{bad}", encoding="utf-8")
        with pytest.raises(ValueError):
            _ = from_path_manifest(p)


def test_manifest_from_path_non_object_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.json"
        # Write a valid JSON that is not an object
        p.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            _ = from_path_manifest(p)
