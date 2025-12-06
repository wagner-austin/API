from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.core.services.model.backends.gpt2.io import get_model_max_seq_len
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)


class _MockModelWithPositions:
    """Mock model with n_positions=42 in config."""

    class _Cfg:
        n_positions = 42

    config = _Cfg()

    @classmethod
    def from_pretrained(cls: type[_MockModelWithPositions], path: str) -> LMModelProto:
        return cls()

    def train(self: _MockModelWithPositions) -> None:
        pass

    def eval(self: _MockModelWithPositions) -> None:
        pass

    def forward(
        self: _MockModelWithPositions, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        raise NotImplementedError

    def parameters(self: _MockModelWithPositions) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _MockModelWithPositions) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _MockModelWithPositions, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _MockModelWithPositions, out_dir: str) -> None:
        pass


class _MockModelNoPositions:
    """Mock model without n_positions in config."""

    class _Cfg:
        pass

    config = _Cfg()

    @classmethod
    def from_pretrained(cls: type[_MockModelNoPositions], path: str) -> LMModelProto:
        return cls()

    def train(self: _MockModelNoPositions) -> None:
        pass

    def eval(self: _MockModelNoPositions) -> None:
        pass

    def forward(
        self: _MockModelNoPositions, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        raise NotImplementedError

    def parameters(self: _MockModelNoPositions) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _MockModelNoPositions) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _MockModelNoPositions, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _MockModelNoPositions, out_dir: str) -> None:
        pass


class _MockModelNonIntPositions:
    """Mock model with n_positions as non-int."""

    class _Cfg:
        n_positions = "not-an-int"

    config = _Cfg()

    @classmethod
    def from_pretrained(cls: type[_MockModelNonIntPositions], path: str) -> LMModelProto:
        return cls()

    def train(self: _MockModelNonIntPositions) -> None:
        pass

    def eval(self: _MockModelNonIntPositions) -> None:
        pass

    def forward(
        self: _MockModelNonIntPositions, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        raise NotImplementedError

    def parameters(self: _MockModelNonIntPositions) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _MockModelNonIntPositions) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _MockModelNonIntPositions, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _MockModelNonIntPositions, out_dir: str) -> None:
        pass


class _MockModelConfigProperty:
    """Mock model with config as property."""

    class _EmptyCfg:
        pass

    @classmethod
    def from_pretrained(cls: type[_MockModelConfigProperty], path: str) -> LMModelProto:
        return cls()

    def train(self: _MockModelConfigProperty) -> None:
        pass

    def eval(self: _MockModelConfigProperty) -> None:
        pass

    def forward(
        self: _MockModelConfigProperty, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        raise NotImplementedError

    def parameters(self: _MockModelConfigProperty) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _MockModelConfigProperty) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _MockModelConfigProperty, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _MockModelConfigProperty, out_dir: str) -> None:
        pass

    @property
    def config(self: _MockModelConfigProperty) -> ConfigLike:
        return self._EmptyCfg()


def test_get_model_max_seq_len_branches() -> None:
    assert get_model_max_seq_len(_MockModelWithPositions()) == 42
    assert get_model_max_seq_len(_MockModelNoPositions()) == 512


def test_get_model_max_seq_len_config_non_int() -> None:
    # When n_positions exists but is not int, fallback is 512
    assert get_model_max_seq_len(_MockModelNonIntPositions()) == 512


def test_get_model_max_seq_len_no_config_attr() -> None:
    assert get_model_max_seq_len(_MockModelConfigProperty()) == 512
