from __future__ import annotations

import types
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from platform_core.json_utils import JSONTypeError
from torch import Tensor

from handwriting_ai import _test_hooks
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


def test_build_model_not_callable_raises() -> None:
    import importlib

    class _FakeModelsModule(types.ModuleType):
        resnet18: int = 123

    fake_models = _FakeModelsModule("torchvision.models")

    def _import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "torchvision.models":
            return fake_models
        return importlib.import_module(name, package)

    _test_hooks.import_module = _import_module
    with pytest.raises(RuntimeError):
        _ = _build_model("resnet18", 10)


class _FakeModelNonDict:
    """Fake model returning list from state_dict - tests runtime type validation."""

    def eval(self) -> _FakeModelNonDict:
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> _test_hooks.LoadStateResultProtocol:
        _ = sd

        class _Res:
            def __init__(self) -> None:
                self.missing_keys: list[str] = []
                self.unexpected_keys: list[str] = []

        return _Res()

    def train(self, mode: bool = True) -> _FakeModelNonDict:
        _ = mode
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Runtime: returns list to test "state_dict() did not return a dict"
        # Use _test_hooks injection point to return bad data
        return _test_hooks.inject_bad_state_dict_list()

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []


class _FakeModelInvalidEntries:
    """Fake model returning non-Tensor dict values - tests runtime type validation."""

    def eval(self) -> _FakeModelInvalidEntries:
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> _test_hooks.LoadStateResultProtocol:
        _ = sd

        class _Res:
            def __init__(self) -> None:
                self.missing_keys: list[str] = []
                self.unexpected_keys: list[str] = []

        return _Res()

    def train(self, mode: bool = True) -> _FakeModelInvalidEntries:
        _ = mode
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Runtime: returns ints instead of Tensors to test validation
        # Use _test_hooks injection point to return bad data
        return _test_hooks.inject_bad_state_dict_values()

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return []


def test_build_fresh_state_dict_non_dict_from_model_raises() -> None:
    def _bm(arch: str, n_classes: int) -> _test_hooks.InferenceTorchModelProtocol:
        _ = (arch, n_classes)
        return _FakeModelNonDict()

    _test_hooks.build_model = _bm
    with pytest.raises(RuntimeError):
        _ = build_fresh_state_dict("resnet18", 10)


def test_build_fresh_state_dict_invalid_entries_raise() -> None:
    def _bm(arch: str, n_classes: int) -> _test_hooks.InferenceTorchModelProtocol:
        _ = (arch, n_classes)
        return _FakeModelInvalidEntries()

    _test_hooks.build_model = _bm
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


def test_load_state_dict_file_not_dict(tmp_path: Path) -> None:
    # Create a real file that torch.load will read as an int
    p = tmp_path / "model.pt"
    torch.save(123, p.as_posix())
    with pytest.raises(JSONTypeError):
        _ = _load_state_dict_file(p)


def test_load_state_dict_file_invalid_entry(tmp_path: Path) -> None:
    # Create a real file that torch.load will read as a dict with invalid values
    p = tmp_path / "model.pt"
    torch.save({"fc.weight": 5}, p.as_posix())
    with pytest.raises(JSONTypeError):
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
    with pytest.raises(JSONTypeError):
        _validate_state_dict(sd, "resnet18", 10)


def test_validate_state_dict_missing_blocks_raises() -> None:
    sd: dict[str, Tensor] = {
        "fc.weight": torch.zeros((10, 512), dtype=torch.float32),
        "fc.bias": torch.zeros((10,), dtype=torch.float32),
        "conv1.weight": torch.zeros((64, 1, 3, 3), dtype=torch.float32),
        "bn1.weight": torch.zeros((64,), dtype=torch.float32),
        "bn1.bias": torch.zeros((64,), dtype=torch.float32),
    }
    with pytest.raises(JSONTypeError):
        _validate_state_dict(sd, "resnet18", 10)
