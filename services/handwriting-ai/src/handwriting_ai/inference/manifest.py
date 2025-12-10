from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypedDict

from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str
from platform_ml.manifest import MANIFEST_SCHEMA_VERSION as V2_SCHEMA
from platform_ml.manifest import ModelManifestV2
from platform_ml.manifest import from_json_manifest_v2 as _from_json_manifest_v2


class ModelManifest(TypedDict):
    schema_version: str
    model_id: str
    arch: str
    n_classes: int
    version: str
    created_at: datetime
    preprocess_hash: str
    val_acc: float
    temperature: float


def from_path_manifest(path: Path) -> ModelManifest:
    text = path.read_text(encoding="utf-8")
    raw: JSONValue = load_json_str(text)
    if not isinstance(raw, dict):
        raise JSONTypeError("manifest must be a JSON object")
    # Support v2 manifests by strictly decoding then mapping to v1 shape
    schema = str(raw.get("schema_version", "")).strip()
    if schema == V2_SCHEMA:
        v2 = _from_json_manifest_v2(text)
        return _map_v2_to_v1(v2)
    data: dict[str, JSONValue] = {str(k): v for k, v in raw.items()}
    return _decode_manifest(data)


def from_json_manifest(s: str) -> ModelManifest:
    raw: JSONValue = load_json_str(s)
    if not isinstance(raw, dict):
        raise JSONTypeError("manifest must be a JSON object")
    schema = str(raw.get("schema_version", "")).strip()
    if schema == V2_SCHEMA:
        v2 = _from_json_manifest_v2(s)
        return _map_v2_to_v1(v2)
    data: dict[str, JSONValue] = {str(k): v for k, v in raw.items()}
    return _decode_manifest(data)


def _decode_manifest(d: dict[str, JSONValue]) -> ModelManifest:
    allowed_schema_versions: Final[tuple[str, ...]] = ("v1", "v1.1")
    created_at_str = str(d["created_at"]) if "created_at" in d else ""
    try:
        created = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
    except ValueError as exc:
        raise JSONTypeError(f"invalid created_at date: {exc}") from exc
    n_classes = int(str(d.get("n_classes", 10)))
    val_acc = float(str(d.get("val_acc", 0.0)))
    temperature = float(str(d.get("temperature", 1.0)))
    if n_classes < 2:
        raise JSONTypeError("n_classes must be >= 2")
    if not (0.0 <= val_acc <= 1.0):
        raise JSONTypeError("val_acc must be within [0,1]")
    if temperature <= 0.0:
        raise JSONTypeError("temperature must be > 0")
    schema_version = str(d.get("schema_version", "")).strip()
    model_id = str(d.get("model_id", "")).strip()
    arch = str(d.get("arch", "")).strip()
    version = str(d.get("version", "")).strip()
    preprocess_hash = str(d.get("preprocess_hash", "")).strip()
    if not schema_version or not model_id or not arch or not version or not preprocess_hash:
        raise JSONTypeError("manifest is missing required fields")
    if schema_version not in allowed_schema_versions:
        raise JSONTypeError("unsupported manifest schema version")
    return {
        "schema_version": schema_version,
        "model_id": model_id,
        "arch": arch,
        "n_classes": n_classes,
        "version": version,
        "created_at": created,
        "preprocess_hash": preprocess_hash,
        "val_acc": val_acc,
        "temperature": temperature,
    }


def _map_v2_to_v1(mv2: ModelManifestV2) -> ModelManifest:
    # Use strict v2 parsing to guarantee types, then map fields to v1 structure
    # with defaults where necessary.
    created = datetime.fromisoformat(mv2["created_at"]).astimezone(UTC).replace(tzinfo=None)
    n_classes = int(mv2["n_classes"]) if "n_classes" in mv2 else 10
    val_acc = float(mv2["val_acc"]) if "val_acc" in mv2 else 0.0
    temperature = 1.0
    preprocess_hash = str(mv2["preprocess_hash"]) if "preprocess_hash" in mv2 else ""
    return {
        "schema_version": "v2.0",
        "model_id": mv2["model_id"],
        "arch": mv2["arch"],
        "n_classes": n_classes,
        "version": "2.0",
        "created_at": created,
        "preprocess_hash": preprocess_hash,
        "val_acc": val_acc,
        "temperature": temperature,
    }
