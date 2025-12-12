from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.json_utils import dump_json_str

from handwriting_ai.api.main import create_app
from handwriting_ai.config import (
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.preprocess import preprocess_signature


class _NoReloadEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.reloaded: bool = False

    def try_load_active(self) -> None:
        self.reloaded = True


def _settings(tmp: Path, api_key: str = "k") -> Settings:
    return {
        "app": {
            "data_root": tmp / "data",
            "artifacts_root": tmp / "artifacts",
            "logs_root": tmp / "logs",
            "threads": 0,
            "port": 8081,
        },
        "digits": {
            "model_dir": tmp / "models",
            "active_model": "mnist_resnet18_v1",
            "tta": False,
            "uncertain_threshold": 0.70,
            "max_image_mb": 2,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 5,
            "visualize_max_kb": 16,
            "retention_keep_runs": 3,
        },
        "security": {"api_key": api_key},
    }


def test_admin_upload_activate_true_but_non_active_model_id_does_not_reload(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    eng = _NoReloadEngine(s)
    app = create_app(s, engine_provider=lambda: eng)
    client = TestClient(app)

    man = {
        "schema_version": "v1.1",
        "model_id": "other_model",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.9,
        "temperature": 1.0,
    }
    # Build a valid state dict to satisfy strict validation when activate=true
    import torch

    from handwriting_ai.inference.engine import build_fresh_state_dict

    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(dump_json_str(man).encode("utf-8")),
            "application/json",
        ),
        "model": ("model.pt", buf, "application/octet-stream"),
    }
    data = {"model_id": "other_model", "activate": "true"}
    r = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert r.status_code == 200 and '"ok":true' in r.text.replace(" ", "").lower()
    # Engine should not have been reloaded because uploaded model_id != active_model
    assert eng.reloaded is False
