from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
import torch
from platform_core.json_utils import JSONTypeError
from torch import Tensor

from handwriting_ai import _test_hooks
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig
from handwriting_ai.inference.engine import (
    InferenceEngine,
    _build_model,
    _load_state_dict_file,
)


class _FakeStat:
    """Fake stat result with st_mtime and st_size attributes."""

    def __init__(self, st_mtime: float, st_size: int) -> None:
        self.st_mtime = st_mtime
        self.st_size = st_size


def _engine(tmp: Path) -> InferenceEngine:
    app: AppConfig = {
        "data_root": tmp,
        "artifacts_root": tmp,
        "logs_root": tmp,
        "threads": 0,
        "port": 8081,
    }
    dig: DigitsConfig = {
        "model_dir": tmp / "models",
        "active_model": "a",
        "tta": False,
        "uncertain_threshold": 0.5,
        "max_image_mb": 1,
        "max_image_side_px": 1024,
        "predict_timeout_seconds": 1,
        "visualize_max_kb": 16,
        "retention_keep_runs": 1,
    }
    sec: SecurityConfig = {"api_key": ""}
    return InferenceEngine({"app": app, "digits": dig, "security": sec})


def test_reload_if_changed_rejects_zero_and_flapping_sizes(tmp_path: Path) -> None:
    # Prepare artifacts dir and set last mtimes older so change detection triggers
    engine = _engine(tmp_path)
    art = tmp_path / "models" / "a"
    art.mkdir(parents=True, exist_ok=True)
    man = art / "manifest.json"
    model = art / "model.pt"
    man.write_text("{}", encoding="utf-8")
    model.write_bytes(b"x")
    engine._artifacts_dir = art
    engine._last_manifest_mtime = 0.0
    engine._last_model_mtime = 0.0

    # Preserve original hook
    orig_path_stat = _test_hooks.path_stat

    # Case 1: size1 <= 0 returns False
    def _stat_zero(path: Path, *, follow_symlinks: bool = True) -> _test_hooks.StatResultProtocol:
        if path.as_posix().endswith("manifest.json"):
            return _FakeStat(st_mtime=999999.0, st_size=0)
        if path.as_posix().endswith("model.pt"):
            return _FakeStat(st_mtime=999999.0, st_size=1)
        return orig_path_stat(path, follow_symlinks=follow_symlinks)

    _test_hooks.path_stat = _stat_zero
    assert engine.reload_if_changed() is False

    # Case 2: second size read differs -> returns False
    calls = {"n": 0}

    def _stat_flap(path: Path, *, follow_symlinks: bool = True) -> _test_hooks.StatResultProtocol:
        if path.as_posix().endswith("manifest.json"):
            # Many stat() calls happen inside _collect_artifact_mtimes();
            # trigger flapping exactly for the two size reads that follow that loop.
            calls["n"] += 1
            if calls["n"] == 17:
                return _FakeStat(st_mtime=999999.0, st_size=10)
            if calls["n"] == 18:
                return _FakeStat(st_mtime=999999.0, st_size=20)
            return _FakeStat(st_mtime=999999.0, st_size=10)
        if path.as_posix().endswith("model.pt"):
            return _FakeStat(st_mtime=999999.0, st_size=1)
        return orig_path_stat(path, follow_symlinks=follow_symlinks)

    _test_hooks.path_stat = _stat_flap
    assert engine.reload_if_changed() is False


def test_build_model_missing_maxpool() -> None:
    import importlib

    class _HasConvOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Provide conv1 but no maxpool
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return x

    def _fake_resnet18(*, weights: None, num_classes: int) -> torch.nn.Module:
        _ = (weights, num_classes)
        return _HasConvOnly()

    class _Models(ModuleType):
        def __init__(self) -> None:
            super().__init__("torchvision.models")
            self.resnet18 = _fake_resnet18

    fake = _Models()

    def _import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "torchvision.models":
            return fake
        return importlib.import_module(name, package)

    _test_hooks.import_module = _import_module
    with pytest.raises(RuntimeError):
        _ = _build_model("resnet18", 10)


def test_load_state_dict_file_invalid_wrapper_and_key(tmp_path: Path) -> None:
    # Wrapper present but nested is not a dict
    p1 = tmp_path / "model1.pt"
    torch.save({"state_dict": 123}, p1.as_posix())
    with pytest.raises(JSONTypeError):
        _ = _load_state_dict_file(p1)

    # Invalid non-string key in state dict
    p2 = tmp_path / "model2.pt"
    torch.save({1: torch.zeros((1,), dtype=torch.float32)}, p2.as_posix())
    with pytest.raises(JSONTypeError):
        _ = _load_state_dict_file(p2)


def test_load_state_dict_file_unreachable_else_via_hooks(tmp_path: Path) -> None:
    p = tmp_path / "model.pt"
    # Valid dict but guards will be forced to both return False to hit the else branch
    torch.save({}, p.as_posix())

    def _always_false_wrapped(
        value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
    ) -> bool:
        _ = value
        return False

    def _always_false_flat(
        value: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
    ) -> bool:
        _ = value
        return False

    _test_hooks.is_wrapped_state_dict = _always_false_wrapped
    _test_hooks.is_flat_state_dict = _always_false_flat
    with pytest.raises(JSONTypeError):
        _ = _load_state_dict_file(p)
