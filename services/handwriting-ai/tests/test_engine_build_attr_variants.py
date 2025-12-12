from __future__ import annotations

import importlib
import types

import pytest

import handwriting_ai.inference.engine as eng
from handwriting_ai import _test_hooks

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def test_build_model_raises_on_missing_attrs() -> None:
    class _M:
        """Model stub deliberately missing conv1/maxpool attributes."""

        pass

    class _FakeModelsModule(types.ModuleType):
        @staticmethod
        def resnet18(weights: None, num_classes: int) -> _M:
            _ = weights
            return _M()

    fake_models = _FakeModelsModule("torchvision.models")
    real_import = importlib.import_module

    def _import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "torchvision.models":
            return fake_models
        return real_import(name, package)

    _test_hooks.import_module = _import_module
    with pytest.raises(RuntimeError):
        eng._build_model("resnet18", 10)
