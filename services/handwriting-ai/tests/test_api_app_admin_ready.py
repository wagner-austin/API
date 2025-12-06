from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, dump_json_str, load_json_bytes
from torch import Tensor, nn

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import (
    InferenceEngine,
    LoadStateResult,
    TorchModel,
    build_fresh_state_dict,
)
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.preprocess import preprocess_signature


def _settings(tmp: Path, *, api_key: str) -> Settings:
    app_cfg: AppConfig = {
        "data_root": tmp / "data",
        "artifacts_root": tmp / "artifacts",
        "logs_root": tmp / "logs",
        "threads": 0,
        "port": 8081,
    }
    dig_cfg: DigitsConfig = {
        "model_dir": tmp / "models",
        "active_model": "m",
        "tta": False,
        "uncertain_threshold": 0.5,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 2,
        "visualize_max_kb": 16,
        "retention_keep_runs": 1,
    }
    sec_cfg: SecurityConfig = {"api_key": api_key, "api_key_enabled": api_key != ""}
    return {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}


class _ReadyEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.try_load_calls = 0
        man = ModelManifest(
            schema_version="v1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash=preprocess_signature(),
            val_acc=0.1,
            temperature=1.0,
        )
        self._manifest = man

        class _StubModel:
            def __init__(self) -> None:
                self._linear = nn.Linear(1, 1)

            def eval(self) -> _StubModel:
                return self

            def __call__(self, x: Tensor) -> Tensor:
                return x

            def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
                return LoadStateResult((), ())

            def train(self, mode: bool = True) -> _StubModel:
                return self

            def state_dict(self) -> dict[str, Tensor]:
                return {}

            def parameters(self) -> tuple[nn.Parameter, ...]:
                return tuple(self._linear.parameters())

        model: TorchModel = _StubModel()
        self._model = model

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def load_state_dict(self, sd: dict[str, Tensor]) -> None:
        return None

    def train(self, mode: bool = True) -> _ReadyEngine:
        return self

    def eval(self) -> _ReadyEngine:
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def parameters(self) -> tuple[Tensor, ...]:
        return ()

    def try_load_active(self) -> None:
        self.try_load_calls += 1
        super().try_load_active()


def test_models_active_ready(tmp_path: Path) -> None:
    settings = _settings(tmp_path, api_key="")
    engine = _ReadyEngine(settings)
    app = create_app(settings, engine_provider=lambda: engine, enforce_api_key=False)
    client = TestClient(app)
    resp = client.get("/v1/models/active")

    class ActiveModelResponse(TypedDict):
        model_loaded: bool
        model_id: str
        schema_version: str

    def _decode_active_model_response(value: JSONValue) -> ActiveModelResponse:
        if not isinstance(value, dict):
            raise AssertionError("response is not an object")
        ml = value.get("model_loaded")
        mid = value.get("model_id")
        sv = value.get("schema_version")
        if not isinstance(ml, bool):
            raise AssertionError("model_loaded must be bool")
        if not isinstance(mid, str) or mid.strip() == "":
            raise AssertionError("model_id must be non-empty str")
        if not isinstance(sv, str) or sv.strip() == "":
            raise AssertionError("schema_version must be non-empty str")
        return {"model_loaded": ml, "model_id": mid, "schema_version": sv}

    parsed: JSONValue = load_json_bytes(resp.content)
    raw = _decode_active_model_response(parsed)
    assert raw["model_loaded"] is True
    assert raw["model_id"] == "m"
    assert raw["schema_version"] == "v1"


def test_admin_upload_activate_true_and_false_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path, api_key="k")
    engine = _ReadyEngine(settings)
    app = create_app(settings, engine_provider=lambda: engine, enforce_api_key=None)
    client = TestClient(app)

    manifest = {
        "schema_version": "v1",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.1,
        "temperature": 1.0,
    }
    manifest_bytes = dump_json_str(manifest).encode("utf-8")
    state_dict = build_fresh_state_dict("resnet18", 10)
    model_bytes = io.BytesIO()
    import torch

    torch.save(state_dict, model_bytes)
    files = {
        "manifest": ("manifest.json", manifest_bytes, "application/json"),
        "model": ("model.pt", model_bytes.getvalue(), "application/octet-stream"),
    }
    resp = client.post(
        "/v1/admin/models/upload",
        headers={"X-Api-Key": "k"},
        files=files,
        data={"model_id": "m", "activate": "true"},
    )
    assert resp.status_code == 200
    assert engine.try_load_calls >= 1

    files_no_activate = {
        "manifest": ("manifest.json", manifest_bytes, "application/json"),
        "model": ("model.pt", b"nonempty", "application/octet-stream"),
    }
    resp2 = client.post(
        "/v1/admin/models/upload",
        headers={"X-Api-Key": "k"},
        files=files_no_activate,
        data={"model_id": "m", "activate": "false"},
    )
    assert resp2.status_code == 200
