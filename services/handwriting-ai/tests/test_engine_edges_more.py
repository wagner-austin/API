from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from torch import Tensor

from handwriting_ai.config import (
    AppConfig,
    DigitsConfig,
    SecurityConfig,
    Settings,
)
from handwriting_ai.inference.engine import (
    InferenceEngine,
    _as_torch_tensor,
    _augment_for_tta,
    _build_model,
    _load_state_dict_file,
    _validate_state_dict,
    build_fresh_state_dict,
)

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_make_pool_uses_threads_when_nonzero() -> None:
    app: AppConfig = {
        "data_root": Path("/tmp/data"),
        "artifacts_root": Path("/tmp/artifacts"),
        "logs_root": Path("/tmp/logs"),
        "threads": 2,
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
    _ = InferenceEngine(s)


def test_build_model_not_callable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    class _FakeModelsModule(types.ModuleType):
        resnet18: int = 123

    fake_models = _FakeModelsModule("torchvision.models")

    def _import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "torchvision.models":
            return fake_models
        return importlib.import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module, raising=True)
    with pytest.raises(RuntimeError):
        _ = _build_model("resnet18", 10)


def test_build_fresh_state_dict_non_dict_from_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _M:
        def state_dict(self) -> list[int]:
            return [1, 2, 3]

    def _bm(arch: str, n_classes: int) -> _M:
        return _M()

    target_mod = sys.modules["handwriting_ai.inference.engine"]
    monkeypatch.setattr(target_mod, "_build_model", _bm, raising=False)
    with pytest.raises(RuntimeError):
        _ = build_fresh_state_dict("resnet18", 10)


def test_build_fresh_state_dict_invalid_entries_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    class _M:
        def state_dict(self) -> dict[str, int]:
            return {"fc.weight": 5}

    def _bm(arch: str, n_classes: int) -> _M:
        return _M()

    target_mod = sys.modules["handwriting_ai.inference.engine"]
    monkeypatch.setattr(target_mod, "_build_model", _bm, raising=False)
    with pytest.raises(RuntimeError):
        _ = build_fresh_state_dict("resnet18", 10)


def test_as_torch_tensor_unsqueezes_3d() -> None:
    t3 = torch.zeros((1, 28, 28), dtype=torch.float32)
    t4 = _as_torch_tensor(t3)
    assert tuple(t4.shape) == (1, 1, 28, 28)


def test_augment_for_tta_noop_for_non4d() -> None:
    t3 = torch.zeros((1, 28, 28), dtype=torch.float32)
    out = _augment_for_tta(t3)
    assert out is t3


def test_load_state_dict_file_not_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Create a real file that torch.load will read as an int
    p = tmp_path / "model.pt"
    torch.save(123, p.as_posix())
    with pytest.raises(ValueError):
        _ = _load_state_dict_file(p)


def test_load_state_dict_file_invalid_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Create a real file that torch.load will read as a dict with invalid values
    p = tmp_path / "model.pt"
    torch.save({"fc.weight": 5}, p.as_posix())
    with pytest.raises(ValueError):
        _ = _load_state_dict_file(p)


def test_validate_state_dict_invalid_dims_raises() -> None:
    sd: dict[str, Tensor] = {
        "fc.weight": torch.zeros((10,), dtype=torch.float32),
        "fc.bias": torch.zeros((10,), dtype=torch.float32),
        "conv1.weight": torch.zeros((64, 1, 3, 3), dtype=torch.float32),
        "bn1.weight": torch.zeros((64,), dtype=torch.float32),
        "bn1.bias": torch.zeros((64,), dtype=torch.float32),
        "layer1.0.weight": torch.zeros((1,), dtype=torch.float32),
        "layer2.0.weight": torch.zeros((1,), dtype=torch.float32),
        "layer3.0.weight": torch.zeros((1,), dtype=torch.float32),
        "layer4.0.weight": torch.zeros((1,), dtype=torch.float32),
    }
    with pytest.raises(ValueError):
        _validate_state_dict(sd, "resnet18", 10)


def test_validate_state_dict_missing_blocks_raises() -> None:
    sd: dict[str, Tensor] = {
        "fc.weight": torch.zeros((10, 512), dtype=torch.float32),
        "fc.bias": torch.zeros((10,), dtype=torch.float32),
        "conv1.weight": torch.zeros((64, 1, 3, 3), dtype=torch.float32),
        "bn1.weight": torch.zeros((64,), dtype=torch.float32),
        "bn1.bias": torch.zeros((64,), dtype=torch.float32),
    }
    with pytest.raises(ValueError):
        _validate_state_dict(sd, "resnet18", 10)
