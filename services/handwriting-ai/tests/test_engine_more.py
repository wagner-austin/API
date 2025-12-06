from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import torch
from platform_core.json_utils import dump_json_str
from torch import Tensor

from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
)
from handwriting_ai.inference.engine import (
    InferenceEngine,
    LoadStateResult,
    TorchModel,
    build_fresh_state_dict,
)
from handwriting_ai.inference.manifest import ModelManifest

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_load_state_dict_file_nested_and_flat(tmp_path: Path) -> None:
    from handwriting_ai.preprocess import preprocess_signature

    sd = build_fresh_state_dict("resnet18", 10)
    for mode in ("flat", "nested"):
        active = f"{mode}"
        active_dir = tmp_path / active
        active_dir.mkdir(parents=True, exist_ok=True)
        man = {
            "schema_version": "v1.1",
            "model_id": active,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (active_dir / "manifest.json").write_text(dump_json_str(man), encoding="utf-8")
        if mode == "flat":
            torch.save(sd, (active_dir / "model.pt").as_posix())
        else:
            torch.save({"state_dict": sd}, (active_dir / "model.pt").as_posix())
        app_cfg: AppConfig = {
            "data_root": Path("/tmp/data"),
            "artifacts_root": Path("/tmp/artifacts"),
            "logs_root": Path("/tmp/logs"),
            "threads": 0,
            "port": 8081,
        }
        dig_cfg: DigitsConfig = {
            "model_dir": tmp_path,
            "active_model": active,
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
        eng = InferenceEngine(s)
        eng.try_load_active()
        assert eng.ready is True


def test_softmax_output_shape_and_sum() -> None:
    class _ZeroModel:
        def eval(self) -> Self:
            return self

        def __call__(self, x: Tensor) -> Tensor:
            b = int(x.shape[0]) if x.ndim == 4 else 1
            return torch.zeros((b, 10), dtype=torch.float32)

        def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
            return LoadStateResult((), ())

        def train(self, mode: bool = True) -> Self:
            return self

        def state_dict(self) -> dict[str, Tensor]:
            return {}

        def parameters(self) -> Sequence[torch.nn.Parameter]:
            return []

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
    model: TorchModel = _ZeroModel()
    eng._model = model
    out = eng._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    assert len(out["probs"]) == 10 and abs(sum(out["probs"]) - 1.0) < 1e-6


def test_tta_changes_score_for_asymmetric_model() -> None:
    class _AsymModel:
        def eval(self) -> Self:
            return self

        def __call__(self, x: Tensor) -> Tensor:
            b = int(x.shape[0])
            logits = torch.zeros((b, 10), dtype=torch.float32)
            # Make later batch entries slightly higher on class 1
            for i in range(b):
                logits[i, 1] = float(i)
            return logits

        def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
            return LoadStateResult((), ())

        def train(self, mode: bool = True) -> Self:
            return self

        def state_dict(self) -> dict[str, Tensor]:
            return {}

        def parameters(self) -> Sequence[torch.nn.Parameter]:
            return []

    app: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 0,
        "port": 8081,
    }
    dig0: DigitsConfig = {
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
    dig1: DigitsConfig = {
        "model_dir": Path("/tmp/models"),
        "active_model": "mnist_resnet18_v1",
        "tta": True,
        "uncertain_threshold": 0.70,
        "max_image_mb": 2,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 5,
        "visualize_max_kb": 16,
        "retention_keep_runs": 3,
    }
    sec: SecurityConfig = {"api_key": ""}
    s0: Settings = {"app": app, "digits": dig0, "security": sec}
    s1: Settings = {"app": app, "digits": dig1, "security": sec}
    eng0 = InferenceEngine(s0)
    eng1 = InferenceEngine(s1)
    from handwriting_ai.preprocess import preprocess_signature

    man = ModelManifest(
        schema_version="v1",
        model_id="m",
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at=datetime.now(UTC),
        preprocess_hash=preprocess_signature(),
        val_acc=0.0,
        temperature=1.0,
    )
    eng0._manifest = man
    eng1._manifest = man
    model0: TorchModel = _AsymModel()
    model1: TorchModel = _AsymModel()
    eng0._model = model0
    eng1._model = model1

    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    out0 = eng0._predict_impl(x)
    out1 = eng1._predict_impl(x)
    # With rotation+shift TTA, confidence should not decrease
    assert out1["probs"][1] >= out0["probs"][1]
