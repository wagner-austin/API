from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
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
from handwriting_ai.inference.types import PredictOutput

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def _img_bytes() -> bytes:
    import io

    from PIL import Image

    img = Image.new("L", (28, 28), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _StubModel:
    def eval(self) -> Self:
        return self

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
        return LoadStateResult((), ())

    def train(self, mode: bool = True) -> Self:
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []


class _OkEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._manifest = ModelManifest(
            schema_version="v1.1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
            val_acc=0.1,
            temperature=1.0,
        )
        model: TorchModel = _StubModel()
        self._model = model

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        f: Future[PredictOutput] = Future()
        # Confidence high to make uncertain False against default threshold 0.70
        probs_base = tuple(0.0 for _ in range(10))
        probs = (1.0, *probs_base[1:])
        out: PredictOutput = {"digit": 0, "confidence": 0.95, "probs": probs, "model_id": "m"}
        f.set_result(out)
        return f


def test_read_success_with_content_length_header_and_no_visualize(tmp_path: Path) -> None:
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
    s: Settings = {"app": app_cfg, "digits": dig_cfg, "security": sec_cfg}
    app = create_app(s, engine_provider=lambda: _OkEngine(s))
    client = TestClient(app)
    files = {"file": ("img.png", _img_bytes(), "image/png")}
    # Provide a small Content-Length to exercise the header-based branch without triggering 413
    headers = {"Content-Length": "10"}
    r = client.post("/v1/read", files=files, headers=headers)
    assert r.status_code == 200
    txt = r.text.replace(" ", "").lower()
    assert '"digit":0' in txt
    assert '"uncertain":false' in txt
    assert '"visual_png_b64":null' in txt
    assert '"model_id":"m"' in txt
