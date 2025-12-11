from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings
from model_trainer.core.contracts.dataset import DatasetBuilder, DatasetConfig
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.model.backend_factory import create_gpt2_backend
from model_trainer.core.types import ForwardOutProto, NamedParameter, ParameterLike


class _FakeConfigLike:
    """Fake config for LMModelProto."""

    n_positions: int = 64


class _FakeForwardOut:
    """Fake forward output."""

    @property
    def loss(self) -> torch.Tensor:
        return torch.tensor(0.0)


class _FakeLMModel:
    """Fake language model for testing."""

    def __init__(self) -> None:
        self.config = _FakeConfigLike()

    @classmethod
    def from_pretrained(cls, path: str) -> _FakeLMModel:
        return cls()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        return _FakeForwardOut()

    def parameters(self) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self, device: str) -> _FakeLMModel:
        return self

    def save_pretrained(self, out_dir: str) -> None:
        pass


class _FakeEncoder:
    """Fake encoder for testing."""

    def encode(self, text: str) -> ListEncoded:
        return ListEncoded([ord(c) for c in text])

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def token_to_id(self, token: str) -> int | None:
        if len(token) == 1:
            return ord(token)
        return None

    def get_vocab_size(self) -> int:
        return 256


def _make_fake_prepared() -> PreparedLMModel:
    """Create a fake PreparedLMModel for testing."""
    return PreparedLMModel(
        model=_FakeLMModel(),
        tokenizer_id="fake_tok",
        eos_id=0,
        pad_id=1,
        max_seq_len=64,
        tok_for_dataset=_FakeEncoder(),
    )


def test_gpt2_backend_load_calls_helper() -> None:
    called: dict[str, str] = {}

    def _fake_load(artifact_path: str, tokenizer: TokenizerHandle) -> PreparedLMModel:
        _ = tokenizer
        called["path"] = artifact_path
        called["tok"] = "called"
        return _make_fake_prepared()

    _test_hooks.load_prepared_gpt2_from_handle = _fake_load

    class _Tok(TokenizerHandle):
        def encode(self: _Tok, text: str) -> list[int]:
            return [1]

        def decode(self: _Tok, ids: list[int]) -> str:
            return "x"

        def token_to_id(self: _Tok, token: str) -> int | None:
            return 0

        def get_vocab_size(self: _Tok) -> int:
            return 1

    class _DS(DatasetBuilder):
        def split(self: _DS, cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
            _ = cfg
            return ([], [], [])

    dataset: DatasetBuilder = _DS()
    backend = create_gpt2_backend(dataset)
    _ = backend.load("/x", load_settings(), tokenizer=_Tok())
    # Function was invoked; validate captured path as proxy for behavior
    assert called["path"] == "/x"
