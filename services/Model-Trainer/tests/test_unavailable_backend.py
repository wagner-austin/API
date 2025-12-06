from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from platform_core.errors import AppError

from model_trainer.core.config.settings import load_settings
from model_trainer.core.contracts.model import (
    GenerateConfig,
    ModelTrainConfig,
    PreparedLMModel,
    ScoreConfig,
)
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.model.unavailable_backend import UnavailableBackend
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)


class _FakeForwardOut(ForwardOutProto):
    @property
    def loss(self: _FakeForwardOut) -> torch.Tensor:
        return torch.tensor(0.0)


class _FakeLM(LMModelProto):
    @classmethod
    def from_pretrained(cls: type[_FakeLM], path: str) -> LMModelProto:
        return cls()

    def train(self: _FakeLM) -> None:
        pass

    def eval(self: _FakeLM) -> None:
        pass

    def forward(self: _FakeLM, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        return _FakeForwardOut()

    def parameters(self: _FakeLM) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _FakeLM) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _FakeLM, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _FakeLM, out_dir: str) -> None:
        pass

    @property
    def config(self: _FakeLM) -> ConfigLike:
        class _C(ConfigLike):
            n_positions = 8

        return _C()


class _FakeEnc(Encoder):
    def encode(self: _FakeEnc, text: str) -> ListEncoded:
        return ListEncoded([1])

    def token_to_id(self: _FakeEnc, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _FakeEnc) -> int:
        return 1

    def decode(self: _FakeEnc, ids: list[int]) -> str:
        return ""


def _make_dummy_prepared() -> PreparedLMModel:
    return PreparedLMModel(
        model=_FakeLM(),
        tokenizer_id="tok",
        eos_id=0,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_FakeEnc(),
    )


class _TokHandle:
    def encode(self: _TokHandle, text: str) -> list[int]:
        return [1]

    def decode(self: _TokHandle, ids: list[int]) -> str:
        return ""

    def token_to_id(self: _TokHandle, token: str) -> int | None:
        return None

    def get_vocab_size(self: _TokHandle) -> int:
        return 1


def test_unavailable_backend_all_methods_raise() -> None:
    loss_initial = 0.0
    ub = UnavailableBackend("llama")
    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "s",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": "tok",
        "corpus_path": "/c",
        "holdout_fraction": 0.01,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
    }
    s = load_settings()

    dummy = _make_dummy_prepared()

    with pytest.raises(AppError):
        _ = ub.prepare(cfg, s, tokenizer=_TokHandle())
    with pytest.raises(AppError):
        _ = ub.save(dummy, "/out")
    with pytest.raises(AppError):
        _ = ub.load("/in", s, tokenizer=_TokHandle())
    with pytest.raises(AppError):
        _ = ub.train(
            cfg,
            s,
            run_id="r1",
            heartbeat=lambda t: None,
            cancelled=lambda: False,
            prepared=dummy,
        )
    with pytest.raises(AppError):
        _ = ub.evaluate(run_id="r1", cfg=cfg, settings=s)

    score_cfg = ScoreConfig(text="hello", path=None, detail_level="summary", top_k=None, seed=None)
    with pytest.raises(AppError):
        _ = ub.score(prepared=dummy, cfg=score_cfg, settings=s)

    gen_cfg = GenerateConfig(
        prompt_text="hello",
        prompt_path=None,
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    with pytest.raises(AppError):
        _ = ub.generate(prepared=dummy, cfg=gen_cfg, settings=s)

    loss_final = 0.0
    assert loss_final <= loss_initial
