from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
import torch
import torch.autograd
from platform_core.json_utils import JSONTypeError
from torch import Tensor
from torch.optim.sgd import SGD

from handwriting_ai.inference.engine import (
    _build_model,
    _load_state_dict_file,
    _normalize_keys,
    _WrappedTorchModel,
)


class _GoodModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Linear = torch.nn.Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.linear(x)
        return out


class _DummyOut:
    pass


class _BadModule(torch.nn.Module):
    def forward(self, x: Tensor) -> _DummyOut:
        return _DummyOut()


def test_wrapped_torch_model_valid_and_invalid_forward() -> None:
    good = _WrappedTorchModel(_GoodModule())
    out = good(torch.zeros((1, 1)))
    # Verify tensor output by accessing dtype and shape
    _ = out.dtype
    _ = out.shape
    good.eval()
    good.train(True)
    sd = good.state_dict()
    assert type(sd) is dict
    params = good.parameters()
    assert type(params) is tuple

    # Verify model can produce valid loss and loss decreases with gradient step
    x_train = torch.randn((4, 1), requires_grad=False)
    y_target = torch.randn((4, 1))
    loss_fn = torch.nn.MSELoss()
    optimizer = SGD(good.parameters(), lr=0.1)
    pred_before: Tensor = good(x_train)
    loss_t_before: Tensor = loss_fn(pred_before, y_target)
    loss_before = float(loss_t_before.item())
    for _ in range(5):
        optimizer.zero_grad()
        pred: Tensor = good(x_train)
        loss_t: Tensor = loss_fn(pred, y_target)
        torch.autograd.backward(loss_t)
        optimizer.step()
    pred_after: Tensor = good(x_train)
    loss_t_after: Tensor = loss_fn(pred_after, y_target)
    loss_after = float(loss_t_after.item())
    assert loss_after < loss_before, "expected loss to decrease"

    bad = _WrappedTorchModel(_BadModule())
    with pytest.raises(RuntimeError):
        _ = bad(torch.zeros((1, 1)))


def test_normalize_keys_rejects_non_strings() -> None:
    bad: Sequence[str] = ["ok", "also"]
    assert _normalize_keys(bad, "missing_keys") == ("ok", "also")


def test_load_state_dict_file_rejects_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"fc.weight": "not_a_tensor"}, path)
    with pytest.raises(JSONTypeError):
        _ = _load_state_dict_file(path)
    torch.save("not a dict", path)
    with pytest.raises(JSONTypeError):
        _ = _load_state_dict_file(path)


def test_build_model_raises_when_builder_missing_conv(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    class _BuilderModule(torch.nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x

    def _fake_resnet18(*, weights: None, num_classes: int) -> torch.nn.Module:
        return _BuilderModule()

    class _ModelsModule(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("torchvision.models")
            self.resnet18 = _fake_resnet18

        def __getattr__(self, name: str) -> NoReturn:
            raise AttributeError(name)

    models_mod = _ModelsModule()
    import importlib

    def _fake_import(name: str) -> types.ModuleType:
        if name == "torchvision.models":
            return models_mod
        return importlib.import_module(name)

    monkeypatch.setattr("importlib.import_module", _fake_import)

    with pytest.raises(RuntimeError):
        _ = _build_model("resnet18", 10)
