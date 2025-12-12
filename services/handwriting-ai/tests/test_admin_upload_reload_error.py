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


class _RaiseReloadEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def try_load_active(self) -> None:
        raise OSError("reload failed")


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


def test_admin_upload_reload_failure_raises_after_logging(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _RaiseReloadEngine(s))
    client = TestClient(app, raise_server_exceptions=False)

    man = {
        "schema_version": "v1.1",
        "model_id": s["digits"]["active_model"],
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.99,
        "temperature": 1.0,
        "run_id": "t",
    }
    # Provide a valid state dict buffer to satisfy strict validation
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
    data = {"model_id": s["digits"]["active_model"], "activate": "true"}
    res = client.post(
        "/v1/admin/models/upload",
        headers={"X-Api-Key": "k"},
        files=files,
        data=data,
    )
    # With strict exception handling, reload failure now raises and endpoint returns 500
    assert res.status_code == 500
    # Files should still be saved before the reload attempt
    dest = s["digits"]["model_dir"] / s["digits"]["active_model"]
    assert (dest / "model.pt").exists()
    assert (dest / "manifest.json").exists()
