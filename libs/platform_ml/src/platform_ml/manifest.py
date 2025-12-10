from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final, Literal, NotRequired, TypedDict

from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str


def _require_nonempty_str(d: dict[str, JSONValue], key: str) -> str:
    v = str(d.get(key, "")).strip()
    if v == "":
        raise JSONTypeError(f"{key} must be non-empty str")
    return v


def _require_iso8601(d: dict[str, JSONValue], key: str) -> str:
    s = _require_nonempty_str(d, key)
    try:
        datetime.fromisoformat(s)
    except ValueError as exc:
        raise JSONTypeError(f"{key} must be ISO 8601") from exc
    return s


def _optional_int_ge(d: dict[str, JSONValue], key: str, *, min_value: int) -> int | None:
    if key not in d:
        return None
    raw = d.get(key)
    if not isinstance(raw, int):
        raise JSONTypeError(f"{key} must be int")
    if raw < min_value:
        raise JSONTypeError(f"{key} must be >= {min_value}")
    return int(raw)


def _optional_float_range(
    d: dict[str, JSONValue], key: str, *, min_value: float, max_value: float | None
) -> float | None:
    if key not in d:
        return None
    raw = d.get(key)
    if not isinstance(raw, (int, float)):
        raise JSONTypeError(f"{key} must be number")
    val = float(raw)
    if max_value is None:
        if val < min_value:
            raise JSONTypeError(f"{key} must be >= {min_value}")
    else:
        if not (min_value <= val <= max_value):
            raise JSONTypeError(f"{key} must be within [{min_value},{max_value}]")
    return val


def _optional_nonempty_str(d: dict[str, JSONValue], key: str) -> str | None:
    if key not in d:
        return None
    s = str(d.get(key, "")).strip()
    if s == "":
        raise JSONTypeError(f"{key} must be non-empty if present")
    return s


def _require_file_info(d: dict[str, JSONValue]) -> tuple[str, int, str]:
    file_id = _require_nonempty_str(d, "file_id")
    size_raw = d.get("file_size")
    sha = _require_nonempty_str(d, "file_sha256")
    if not isinstance(size_raw, int) or size_raw <= 0:
        raise JSONTypeError("file_size must be positive int")
    return file_id, int(size_raw), sha


MANIFEST_SCHEMA_VERSION: Final[Literal["v2.0"]] = "v2.0"


class TrainingRunMetadata(TypedDict):
    """Training run configuration metadata."""

    run_id: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    optimizer: str
    scheduler: str
    augment: bool


class ModelManifestV2(TypedDict):
    """Standardized model manifest schema v2.

    file_id and file_sha256 reference the remote artifact stored in data-bank-api.
    created_at is an ISO 8601 timestamp string for portability across languages.
    """

    schema_version: Literal["v2.0"]
    model_type: Literal["resnet18", "gpt2"]
    model_id: str
    created_at: str
    arch: str
    n_classes: NotRequired[int]
    vocab_size: NotRequired[int]
    val_acc: NotRequired[float]
    val_loss: NotRequired[float]
    preprocess_hash: NotRequired[str]
    file_id: str
    file_size: int
    file_sha256: str
    training: TrainingRunMetadata


def from_path_manifest_v2(path: Path) -> ModelManifestV2:
    raw: JSONValue = load_json_str(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise JSONTypeError("manifest must be a JSON object")
    data: dict[str, JSONValue] = {str(k): v for k, v in raw.items()}
    return _decode_manifest_v2(data)


def from_json_manifest_v2(s: str) -> ModelManifestV2:
    raw: JSONValue = load_json_str(s)
    if not isinstance(raw, dict):
        raise JSONTypeError("manifest must be a JSON object")
    data: dict[str, JSONValue] = {str(k): v for k, v in raw.items()}
    return _decode_manifest_v2(data)


def _decode_manifest_v2(d: dict[str, JSONValue]) -> ModelManifestV2:
    schema_version = str(d.get("schema_version", "")).strip()
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise JSONTypeError("unsupported manifest schema version")

    model_type_str = str(d.get("model_type", "")).strip()
    if model_type_str not in ("resnet18", "gpt2"):
        raise JSONTypeError("unsupported model_type")

    model_id = _require_nonempty_str(d, "model_id")
    created_at = _require_iso8601(d, "created_at")
    arch = _require_nonempty_str(d, "arch")

    n_classes = _optional_int_ge(d, "n_classes", min_value=2)
    vocab_size = _optional_int_ge(d, "vocab_size", min_value=1)
    val_acc = _optional_float_range(d, "val_acc", min_value=0.0, max_value=1.0)
    val_loss = _optional_float_range(d, "val_loss", min_value=0.0, max_value=None)
    preprocess_hash = _optional_nonempty_str(d, "preprocess_hash")

    file_id, file_size, file_sha256 = _require_file_info(d)

    t_raw = d.get("training")
    if not isinstance(t_raw, dict):
        raise JSONTypeError("training must be object")
    training = _decode_training_metadata({str(k): v for (k, v) in t_raw.items()})

    # Narrow model_type to the expected Literal for strict typing
    model_type_lit: Literal["resnet18", "gpt2"] = (
        "resnet18" if model_type_str == "resnet18" else "gpt2"
    )

    out: ModelManifestV2 = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_type": model_type_lit,
        "model_id": model_id,
        "created_at": created_at,
        "arch": arch,
        "file_id": file_id,
        "file_size": file_size,
        "file_sha256": file_sha256,
        "training": training,
    }
    if n_classes is not None:
        out["n_classes"] = n_classes
    if vocab_size is not None:
        out["vocab_size"] = vocab_size
    if val_acc is not None:
        out["val_acc"] = val_acc
    if val_loss is not None:
        out["val_loss"] = val_loss
    if preprocess_hash is not None:
        out["preprocess_hash"] = preprocess_hash
    return out


def _decode_training_metadata(d: dict[str, JSONValue]) -> TrainingRunMetadata:
    run_id = str(d.get("run_id", "")).strip()
    epochs = d.get("epochs")
    batch_size = d.get("batch_size")
    learning_rate = d.get("learning_rate")
    seed = d.get("seed")
    device = str(d.get("device", "")).strip()
    optimizer = str(d.get("optimizer", "")).strip()
    scheduler = str(d.get("scheduler", "")).strip()
    augment = d.get("augment")

    if (
        not run_id
        or not isinstance(epochs, int)
        or not isinstance(batch_size, int)
        or not isinstance(learning_rate, (int, float))
        or not isinstance(seed, int)
        or not device
        or not optimizer
        or not scheduler
        or not isinstance(augment, bool)
    ):
        raise JSONTypeError("invalid training metadata")

    return {
        "run_id": run_id,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": device,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "augment": bool(augment),
    }


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ModelManifestV2",
    "TrainingRunMetadata",
    "from_json_manifest_v2",
    "from_path_manifest_v2",
]
