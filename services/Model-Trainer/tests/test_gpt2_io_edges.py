from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch

from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.backends.gpt2.io import (
    load_prepared_gpt2_from_handle,
    save_prepared_gpt2,
    token_ids,
)
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)


class _FakeTokenInfo(Protocol):
    def token_to_id(self: _FakeTokenInfo, token: str) -> int | None: ...
    def get_vocab_size(self: _FakeTokenInfo) -> int: ...


class _Fwd(ForwardOutProto):
    @property
    def loss(self: _Fwd) -> torch.Tensor:
        return torch.tensor(0.0)


class _FakeModelConfig(ConfigLike):
    n_positions = 8


class FakeModel(LMModelProto):
    @classmethod
    def from_pretrained(cls: type[FakeModel], path: str) -> LMModelProto:
        return cls()

    def train(self: FakeModel) -> None:
        pass

    def eval(self: FakeModel) -> None:
        pass

    def forward(
        self: FakeModel, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        return _Fwd()

    def parameters(self: FakeModel) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: FakeModel) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: FakeModel, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: FakeModel, out_dir: str) -> None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights.bin").write_bytes(b"ok")

    @property
    def config(self: FakeModel) -> ConfigLike:
        return _FakeModelConfig()


class Enc:
    def __init__(self: Enc, ids: list[int]) -> None:
        self._ids = ids

    @property
    def ids(self: Enc) -> list[int]:
        return self._ids


class Tok:
    def encode(self: Tok, text: str) -> Enc:
        return Enc([1, 2, 3])

    def token_to_id(self: Tok, token: str) -> int | None:
        return 0

    def get_vocab_size(self: Tok) -> int:
        return 10

    def decode(self: Tok, ids: list[int]) -> str:
        return "".join(str(i) for i in ids)


# Helper classes for load tests - defined at module level to reduce complexity


class _TokH:
    """Tokenizer handle with EOS/PAD token support."""

    def encode(self: _TokH, text: str) -> list[int]:
        return [1]

    def token_to_id(self: _TokH, token: str) -> int | None:
        if token == "[EOS]":
            return 2
        if token == "[PAD]":
            return 0
        return None

    def get_vocab_size(self: _TokH) -> int:
        return 16

    def decode(self: _TokH, ids: list[int]) -> str:
        return "".join(str(i) for i in ids)


class _TokHNoTokens:
    """Tokenizer handle without special tokens."""

    def encode(self: _TokHNoTokens, text: str) -> list[int]:
        return [1]

    def token_to_id(self: _TokHNoTokens, token: str) -> int | None:
        return None

    def get_vocab_size(self: _TokHNoTokens) -> int:
        return 16

    def decode(self: _TokHNoTokens, ids: list[int]) -> str:
        return "".join(str(i) for i in ids)


class _CfgWithPositions(ConfigLike):
    """Config with n_positions set."""

    n_positions: int = 64


class _CfgNoPositions(ConfigLike):
    """Config without n_positions to trigger default."""

    pass


class _FakeModel(LMModelProto):
    """Fake model with configurable n_positions."""

    def __init__(self: _FakeModel, cfg: ConfigLike) -> None:
        self._config = cfg

    @classmethod
    def from_pretrained(cls: type[_FakeModel], path: str) -> LMModelProto:
        return cls(_CfgWithPositions())

    def train(self: _FakeModel) -> None:
        pass

    def eval(self: _FakeModel) -> None:
        pass

    def forward(
        self: _FakeModel, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        return _Fwd()

    def parameters(self: _FakeModel) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _FakeModel) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _FakeModel, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _FakeModel, out_dir: str) -> None:
        pass

    @property
    def config(self: _FakeModel) -> ConfigLike:
        return self._config


def test_token_ids_defaults_when_missing() -> None:
    class FakeTok:
        def token_to_id(self: FakeTok, token: str) -> int | None:
            return None

        def get_vocab_size(self: FakeTok) -> int:
            return 100

    eos_id, pad_id, vocab = token_ids(FakeTok())
    assert eos_id == 0
    assert pad_id == 0
    assert vocab == 100


def test_save_prepared_gpt2_writes(tmp_path: Path) -> None:
    prepared = PreparedLMModel(
        model=FakeModel(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=Tok(),
    )
    out_dir = str(tmp_path / "m")
    save_prepared_gpt2(prepared, out_dir)
    assert (tmp_path / "m" / "weights.bin").exists()


def test_load_prepared_gpt2_from_handle_uses_n_positions() -> None:
    from model_trainer.core import _test_hooks

    def fake_load(path: str) -> LMModelProto:
        _ = path
        return _FakeModel(_CfgWithPositions())

    _test_hooks.load_gpt2_model = fake_load

    prepared = load_prepared_gpt2_from_handle("/does/not/matter", _TokH())
    assert prepared.max_seq_len == 64


def test_load_prepared_gpt2_from_handle_defaults_when_missing() -> None:
    from model_trainer.core import _test_hooks

    def fake_load2(path: str) -> LMModelProto:
        _ = path
        return _FakeModel(_CfgNoPositions())

    _test_hooks.load_gpt2_model = fake_load2

    prepared = load_prepared_gpt2_from_handle("/no/file", _TokHNoTokens())
    assert prepared.max_seq_len == 512
