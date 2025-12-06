from __future__ import annotations

import pytest
import torch

import handwriting_ai.inference.engine as eng


def test_wrapped_state_dict_invalid_value_type_raises() -> None:
    m = torch.nn.Module()

    # Assign a custom state_dict that returns a wrong value type
    def _bad_state_dict_value() -> dict[str, str]:
        return {"ok": "not-a-tensor"}

    object.__setattr__(m, "state_dict", _bad_state_dict_value)
    wrapped = eng._WrappedTorchModel(m)
    with pytest.raises(RuntimeError, match="state_dict value must be Tensor"):
        _ = wrapped.state_dict()


def test_wrapped_state_dict_invalid_key_type_raises() -> None:
    m = torch.nn.Module()

    # Assign a custom state_dict that returns a non-string key
    def _bad_state_dict_key() -> dict[int, torch.Tensor]:
        return {1: torch.zeros((1,), dtype=torch.float32)}

    object.__setattr__(m, "state_dict", _bad_state_dict_key)
    wrapped = eng._WrappedTorchModel(m)
    with pytest.raises(RuntimeError, match="state_dict key must be str"):
        _ = wrapped.state_dict()
