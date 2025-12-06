from __future__ import annotations

from pathlib import Path

from handwriting_ai.inference.manifest import from_json_manifest, from_path_manifest


def _v2_training_block() -> str:
    """Return a valid v2 training metadata block."""
    return (
        '"training":{"run_id":"run-test","epochs":10,"batch_size":32,'
        '"learning_rate":0.001,"seed":42,"device":"cpu","optimizer":"adam",'
        '"scheduler":"none","augment":false}'
    )


def test_from_path_manifest_v2(tmp_path: Path) -> None:
    """Cover manifest.py:33-34 - v2 schema handling in from_path_manifest."""
    # Create a valid v2 manifest with all required fields
    v2_manifest = (
        '{"schema_version":"v2.0","model_type":"resnet18","model_id":"test-model",'
        '"created_at":"2025-01-01T00:00:00+00:00","arch":"resnet18","n_classes":10,'
        '"val_acc":0.95,"preprocess_hash":"abc123","file_id":"file-xyz",'
        f'"file_size":1024,"file_sha256":"a1b2c3d4e5f6",{_v2_training_block()}}}'
    )
    path = tmp_path / "manifest.json"
    path.write_text(v2_manifest, encoding="utf-8")

    result = from_path_manifest(path)
    assert result["schema_version"] == "v2.0"
    assert result["model_id"] == "test-model"
    assert result["arch"] == "resnet18"
    assert result["n_classes"] == 10
    assert result["val_acc"] == 0.95
    assert result["preprocess_hash"] == "abc123"


def test_from_json_manifest_v2() -> None:
    """Cover manifest.py:45-46 - v2 schema handling in from_json_manifest."""
    v2_manifest = (
        '{"schema_version":"v2.0","model_type":"resnet18","model_id":"json-model",'
        '"created_at":"2025-06-15T12:30:00+00:00","arch":"resnet18","n_classes":10,'
        '"val_acc":0.88,"preprocess_hash":"hash456","file_id":"file-abc",'
        f'"file_size":2048,"file_sha256":"d4e5f6a1b2c3",{_v2_training_block()}}}'
    )

    result = from_json_manifest(v2_manifest)
    assert result["schema_version"] == "v2.0"
    assert result["model_id"] == "json-model"
    assert result["arch"] == "resnet18"
    assert result["n_classes"] == 10
    assert result["val_acc"] == 0.88
    assert result["preprocess_hash"] == "hash456"


def test_map_v2_to_v1_optional_fields() -> None:
    """Cover manifest.py:89-94 - optional field defaults in _map_v2_to_v1."""
    # Create v2 manifest without optional n_classes, val_acc, preprocess_hash
    v2_minimal = (
        '{"schema_version":"v2.0","model_type":"resnet18","model_id":"minimal",'
        '"created_at":"2025-01-01T00:00:00+00:00","arch":"resnet18","file_id":"f",'
        f'"file_size":512,"file_sha256":"abc123",{_v2_training_block()}}}'
    )

    result = from_json_manifest(v2_minimal)
    assert result["schema_version"] == "v2.0"
    assert result["model_id"] == "minimal"
    # These should use defaults from _map_v2_to_v1
    assert result["n_classes"] == 10  # default
    assert result["val_acc"] == 0.0  # default
    assert result["preprocess_hash"] == ""  # default
    assert result["temperature"] == 1.0  # always default in v2
