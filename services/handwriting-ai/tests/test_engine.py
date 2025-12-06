from __future__ import annotations

from pathlib import Path

import pytest
import torch

from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
)
from handwriting_ai.inference.engine import InferenceEngine, LoadStateResult

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_engine_submit_with_zero_model() -> None:
    app: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
        "tta": False,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    s: Settings = {"app": app, "digits": dig, "security": sec}
    eng = InferenceEngine(s)
    # Engine without a loaded model should reject predict.
    # Calling the internal implementation raises RuntimeError.
    with pytest.raises(RuntimeError):
        _ = eng._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))


def test_engine_tta_batching() -> None:
    from collections.abc import Sequence
    from typing import Self

    from handwriting_ai.inference.engine import TorchModel

    # Dummy model that records batch size and returns zero logits
    class _M:
        def __init__(self) -> None:
            self.last_batch: int = 0

        def eval(self) -> Self:
            return self

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            self.last_batch = int(x.shape[0]) if x.ndim == 4 else 1
            return torch.zeros((self.last_batch, 10), dtype=torch.float32)

        def load_state_dict(self, sd: dict[str, torch.Tensor]) -> LoadStateResult:
            return LoadStateResult((), ())

        # Satisfy TorchModel protocol
        def train(self, mode: bool = True) -> Self:
            return self

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {}

        def parameters(self) -> Sequence[torch.nn.Parameter]:
            return []

    def _mk_engine(tta: bool) -> InferenceEngine:
        app: AppConfig = {
            "data_root": Path("/tmp/data"),
            "artifacts_root": Path("/tmp/artifacts"),
            "logs_root": Path("/tmp/logs"),
            "threads": 0,
            "port": 8081,
        }
        dig: DigitsConfig = {
            "model_dir": Path("/tmp/models"),
            "active_model": "mnist_resnet18_v1",
            "tta": tta,
            "uncertain_threshold": 0.70,
            "max_image_mb": 2,
            "max_image_side_px": 1024,
            "predict_timeout_seconds": 5,
            "visualize_max_kb": 16,
            "retention_keep_runs": 3,
        }
        sec: SecurityConfig = {"api_key": ""}
        s: Settings = {"app": app, "digits": dig, "security": sec}
        eng = InferenceEngine(s)
        # Inject manifest and model directly
        from datetime import UTC, datetime

        from handwriting_ai.inference.manifest import ModelManifest

        eng._manifest = ModelManifest(
            schema_version="v1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1",
            created_at=datetime.now(UTC),
            preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
            val_acc=0.0,
            temperature=1.0,
        )
        model: TorchModel = _M()
        eng._model = model  # assign dummy model implementing protocol
        return eng

    e0 = _mk_engine(tta=False)
    _ = e0._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    assert type(e0._model) is _M
    assert e0._model.last_batch == 1

    e1 = _mk_engine(tta=True)
    _ = e1._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    assert type(e1._model) is _M
    assert e1._model.last_batch > 1
