from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import torch
from fastapi.testclient import TestClient
from torch import Tensor

from handwriting_ai.api.main import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, LoadStateResult, TorchModel
from handwriting_ai.inference.manifest import ModelManifest

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def _make_test_settings(tmp_path: Path) -> Settings:
    app_cfg: AppConfig = {"threads": 0, "port": 8081}
    dig_cfg: DigitsConfig = {
        "model_dir": tmp_path / "models",
        "active_model": "test_model",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec_cfg: SecurityConfig = {"api_key": ""}
    return {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}


class _StubModel:
    def eval(self) -> Self:
        return self

    def __call__(self, x: Tensor) -> Tensor:
        return x  # not used in these tests

    def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
        return LoadStateResult((), ())

    def train(self, mode: bool = True) -> Self:
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []


class _ReadyEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # Mark engine as ready by filling both _manifest and _model
        self._manifest = ModelManifest(
            schema_version="v1.1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
            val_acc=0.5,
            temperature=1.0,
        )
        model: TorchModel = _StubModel()
        self._model = model


def test_provider_closures_exposed_and_invoked(tmp_path: Path) -> None:
    s = _make_test_settings(tmp_path)
    app = create_app(s)
    # Invoke provider closures to cover create_app provider definitions (no type checks)
    app.state.provide_engine()  # coverage only
    app.state.provide_settings()
    app.state.provide_limits()


def test_readyz_ready_branch(tmp_path: Path) -> None:
    s = _make_test_settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _ReadyEngine(s))
    client = TestClient(app)
    # readyz "ready" branch
    rr = client.get("/readyz")
    assert rr.status_code == 200 and '"status":"ready"' in rr.text.replace(" ", "")


def test_models_active_when_no_manifest(tmp_path: Path) -> None:
    # Default engine without artifacts -> model not loaded branch (app.py:156-159)
    app = create_app(_make_test_settings(tmp_path))
    client = TestClient(app)
    r = client.get("/v1/models/active")
    assert r.status_code == 200
    body = r.text.replace(" ", "").lower()
    assert '"model_loaded":false' in body and '"model_id":null' in body
