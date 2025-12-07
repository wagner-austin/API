from __future__ import annotations

import io
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutTimeout
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from platform_core.errors import ErrorCode, HandwritingErrorCode
from platform_core.json_utils import JSONValue
from torch import Tensor

from handwriting_ai.api.app import create_app
from handwriting_ai.config import Settings
from handwriting_ai.inference.engine import (
    InferenceEngine,
    LoadStateResult,
    TorchModel,
    build_fresh_state_dict,
)
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput, PreprocessOutput
from handwriting_ai.preprocess import PreprocessOptions, preprocess_signature


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (28, 28), color=0).save(buf, format="PNG")
    return buf.getvalue()


def _settings(
    tmp: Path, *, max_mb: int = 2, api_key: str = "", seed_root: Path | None = None
) -> Settings:
    return {
        "app": {"threads": 0, "port": 8081},
        "digits": {
            "model_dir": tmp / "models",
            "seed_root": seed_root if seed_root is not None else tmp / "seed",
            "active_model": "m",
            "tta": False,
            "uncertain_threshold": 0.70,
            "max_image_mb": max_mb,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 1,
            "visualize_max_kb": 16,
            "retention_keep_runs": 1,
        },
        "security": {"api_key": api_key, "api_key_enabled": api_key != ""},
    }


class _ReadyEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

        class _TorchReadyModel:
            def eval(self) -> _TorchReadyModel:
                return self

            def __call__(self, x: Tensor) -> Tensor:
                return x

            def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
                return LoadStateResult((), ())

            def train(self, mode: bool = True) -> _TorchReadyModel:
                return self

            def state_dict(self) -> dict[str, Tensor]:
                return {}

            def parameters(self) -> tuple[torch.nn.Parameter, ...]:
                return ()

        man = ModelManifest(
            schema_version="v1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
            val_acc=0.0,
            temperature=1.0,
        )
        self._manifest = man
        ready_model: TorchModel = _TorchReadyModel()
        self._model = ready_model

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        f: Future[PredictOutput] = Future()
        probs = tuple(0.1 for _ in range(10))
        f.set_result(
            PredictOutput(digit=1, confidence=0.8, probs=probs, model_id="m"),
        )
        return f


def test_ready_and_active_endpoints_when_model_loaded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    eng = _ReadyEngine(settings)
    app = create_app(settings, engine_provider=lambda: eng, enforce_api_key=False)
    client = TestClient(app)

    r_ready = client.get("/readyz")
    body_ready: JSONValue = r_ready.json()
    assert r_ready.status_code == 200
    if type(body_ready) is not dict:
        raise AssertionError("expected dict")
    assert body_ready["status"] == "ready"

    r_active = client.get("/v1/models/active")
    body_active: JSONValue = r_active.json()
    assert r_active.status_code == 200
    if type(body_active) is not dict:
        raise AssertionError("expected dict")
    assert body_active["model_loaded"] is True
    assert body_active["model_id"] == "m"
    assert body_active["schema_version"] == "v1"


def test_read_rejects_extra_form_field(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(
        settings,
        engine_provider=lambda: _ReadyEngine(settings),
        enforce_api_key=False,
    )
    client = TestClient(app)
    files = {"file": ("img.png", _png_bytes(), "image/png")}
    data = {"unexpected": "x"}

    r = client.post("/v1/read", files=files, data=data)
    assert r.status_code == 400
    body: JSONValue = r.json()
    if type(body) is not dict:
        raise AssertionError("expected dict")
    assert body["code"] == HandwritingErrorCode.malformed_multipart.value


def test_read_raises_on_timeout_and_cancels_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    class _TimeoutEngine(_ReadyEngine):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self.last_future: _TimeoutFuture | None = None

        def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
            fut = _TimeoutFuture()
            self.last_future = fut
            return fut

    class _TimeoutFuture(Future[PredictOutput]):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_called = False

        def result(self, timeout: float | None = None) -> PredictOutput:
            raise FutTimeout()

        def cancel(self) -> bool:
            self.cancel_called = True
            return True

    def _fake_preprocess(_: Image.Image, opts: PreprocessOptions) -> PreprocessOutput:
        return {"tensor": Tensor(), "visual_png": b""}

    monkeypatch.setattr("handwriting_ai.api.routes.read.run_preprocess", _fake_preprocess)

    engine = _TimeoutEngine(settings)
    app = create_app(settings, engine_provider=lambda: engine, enforce_api_key=False)
    client = TestClient(app)
    files = {"file": ("img.png", _png_bytes(), "image/png")}

    r = client.post("/v1/read", files=files)
    assert r.status_code == 504  # TIMEOUT default status
    body: JSONValue = r.json()
    if type(body) is not dict:
        raise AssertionError("expected dict")
    assert body["code"] == ErrorCode.TIMEOUT
    if not isinstance(engine.last_future, _TimeoutFuture):
        raise AssertionError("expected _TimeoutFuture")
    assert engine.last_future.cancel_called is True


def test_seed_startup_copies_seed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use tmp_path for seed_root to isolate from other tests
    seed_base = tmp_path / "seed"
    settings = _settings(tmp_path, seed_root=seed_base)

    seed_model_dir = seed_base / "m"
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    valid_manifest = (
        '{"schema_version":"v1","model_id":"m","arch":"resnet18","n_classes":10,'
        '"version":"1.0.0","created_at":"2025-01-01T00:00:00Z","preprocess_hash":"h",'
        '"val_acc":0.9,"temperature":1.0}'
    )
    (seed_model_dir / "model.pt").write_bytes(b"model-bytes")
    (seed_model_dir / "manifest.json").write_text(valid_manifest, encoding="utf-8")

    app = create_app(
        settings,
        engine_provider=lambda: _ReadyEngine(settings),
        enforce_api_key=False,
    )
    dest = settings["digits"]["model_dir"] / "m"
    if dest.exists():
        for f in dest.iterdir():
            f.unlink()

    with TestClient(app):
        pass

    assert (dest / "model.pt").read_bytes() == b"model-bytes"
    assert (dest / "manifest.json").read_text(encoding="utf-8") == valid_manifest
    # No manual cleanup needed - tmp_path is automatically cleaned up by pytest


def test_seed_eager_then_engine_loads_and_ready(tmp_path: Path) -> None:
    # Use tmp_path for seed_root to isolate from other tests
    seed_base = tmp_path / "seed"
    settings = _settings(tmp_path, seed_root=seed_base)

    # Prepare valid seed artifacts for model id "m"
    seed_model_dir = seed_base / "m"
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    sd = build_fresh_state_dict("resnet18", 10)
    import torch

    torch.save(sd, (seed_model_dir / "model.pt").as_posix())
    manifest = {
        "schema_version": "v1.1",
        "model_id": "m",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.1,
        "temperature": 1.0,
    }
    from platform_core.json_utils import dump_json_str, load_json_bytes

    (seed_model_dir / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")

    app = create_app(settings, enforce_api_key=False)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r_ready = client.get("/readyz")
        body_ready = load_json_bytes(r_ready.content)
        assert r_ready.status_code == 200
        assert isinstance(body_ready, dict) and body_ready.get("status") == "ready"

        r_active = client.get("/v1/models/active")
        body_active = load_json_bytes(r_active.content)
        assert r_active.status_code == 200
        if type(body_active) is not dict:
            raise AssertionError("expected dict")
        assert body_active.get("model_loaded") is True and body_active.get("model_id") == "m"
    # No manual cleanup needed - tmp_path is automatically cleaned up by pytest


def test_seed_noop_when_destination_exists(tmp_path: Path) -> None:
    # Use tmp_path for seed_root to isolate from other tests
    seed_base = tmp_path / "seed"
    settings = _settings(tmp_path, seed_root=seed_base)

    # Create both seed and destination files; helper should be a no-op
    seed_model_dir = seed_base / "m"
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    dest = settings["digits"]["model_dir"] / "m"
    dest.mkdir(parents=True, exist_ok=True)

    valid_manifest = (
        '{"schema_version":"v1","model_id":"m","arch":"resnet18","n_classes":10,'
        '"version":"1.0.0","created_at":"2025-01-01T00:00:00Z","preprocess_hash":"h",'
        '"val_acc":0.9,"temperature":1.0}'
    )
    (seed_model_dir / "model.pt").write_bytes(b"seed")
    (seed_model_dir / "manifest.json").write_text(valid_manifest, encoding="utf-8")

    (dest / "model.pt").write_bytes(b"dest")
    (dest / "manifest.json").write_text(valid_manifest, encoding="utf-8")

    app = create_app(settings, enforce_api_key=False)
    from fastapi.testclient import TestClient

    # Should not raise and should not overwrite existing destination
    with TestClient(app):
        pass

    assert (dest / "model.pt").read_bytes() == b"dest"
    assert (dest / "manifest.json").read_text(encoding="utf-8").startswith("{")
    # No manual cleanup needed - tmp_path is automatically cleaned up by pytest
